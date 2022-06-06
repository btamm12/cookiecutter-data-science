
import csv
from enum import Enum
from re import X
import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.csv_info import STANDARDIZED_CSV_INFO
from {{cookiecutter.pkg_name}}.data.split import Split
from {{cookiecutter.pkg_name}}.model.input import Input
from {{cookiecutter.pkg_name}}.utils.full_path import full_path


def _decode_non_mp3_file_like(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio

    array, sampling_rate = sf.read(file)
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=new_sr,
            res_type="kaiser_best"
        )
        sampling_rate = new_sr
    return array, sampling_rate


def load_audio(file_path: str, sampling_rate: int) -> torch.Tensor:
    array, _ = _decode_non_mp3_file_like(file_path, sampling_rate)
    array = np.float32(array)
    return torch.from_numpy(array)

class CropType(Enum):
    NONE = 0
    RANDOM = 1
    CENTER = 2

class MyCrop(torch.nn.Module):
    def __init__(self, crop_type: CropType, input: Input, feat_seq_len: int = None):
        super().__init__()
        self.crop_type = crop_type
        self.input = input
        self.feat_seq_len = feat_seq_len
        if crop_type != CropType.NONE:
            if feat_seq_len is None:
                raise Exception("crop_type is specified but feat_seq_len is not!")

        if crop_type == CropType.CENTER:
            if input == Input.AUDIO:
                center_crop = transforms.CenterCrop((feat_seq_len, 1,))
            elif input == Input.MFCC:
                center_crop = transforms.CenterCrop((feat_seq_len, 40,))
            elif input == Input.XLSR:
                center_crop = transforms.CenterCrop((feat_seq_len, 1024,))
            self.crop = center_crop
        elif crop_type == CropType.RANDOM:
            if input == Input.AUDIO:
                random_crop = transforms.RandomCrop(
                    (feat_seq_len, 1,),
                    pad_if_needed=True,
                )
            elif input == Input.MFCC:
                random_crop = transforms.RandomCrop(
                    (feat_seq_len, 40,),
                    pad_if_needed=True,
                )
            elif input == Input.XLSR:
                random_crop = transforms.RandomCrop(
                    (feat_seq_len, 1024,),
                    pad_if_needed=True,
                )
            self.crop = random_crop
    def forward(self, x):
        if self.crop_type == CropType.NONE:
            return x
        else:
            return self.crop(x)


class MyNormalize(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = var.sqrt()

    def forward(self, x: torch.Tensor):
        out = (x - self.mean) / self.std
        return out

def make_transform(
    crop_type: CropType,
    input: Input,
    feat_seq_len: int = None,
    mean: torch.Tensor = None,
    var: torch.Tensor = None,
):
    if mean is None or var is None:
        assert mean is None
        assert var is None
        return MyCrop(crop_type, input, feat_seq_len)
    else:
        return transforms.Compose([
        MyCrop(crop_type, input, feat_seq_len),
        MyNormalize(mean, var),
    ])


class CsvDataset(Dataset):

    def __init__(
        self,
        split: Split,
        example: bool,
        input: Input,
        get_label: bool = True,
        integer_label: bool = True,
        do_normalization: bool = True,
        crop_type: CropType = CropType.CENTER,
        feat_seq_len: int = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.example = example
        self.input = input
        self.get_label = get_label
        self.integer_label = integer_label
        self.dtype = torch.int64 if integer_label else torch.float32
        self.do_normalization = do_normalization
        self.crop_type = crop_type
        if crop_type != CropType.NONE:
            if feat_seq_len is None:
                raise Exception("crop_type is specified, but feat_seq_len is not!")

        # Select train, val or test dataset.
        dataset = constants.get_dataset(split, example)

        # Input type to CSV column.
        if input == Input.AUDIO:
            col_path = STANDARDIZED_CSV_INFO.col_audio_path
        elif input == Input.MFCC:
            col_path = STANDARDIZED_CSV_INFO.col_mfcc_path
        elif input == Input.XLSR:
            col_path = STANDARDIZED_CSV_INFO.col_xlsr_path
        else:
            raise Exception(f"Unknown input: {input}.")
        
        # Label column.
        col_label = STANDARDIZED_CSV_INFO.col_label

        # Load CSV.
        self.csv_data = []  # (feature_path [, label])
        with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):
                # Skip header row.
                if idx == 0:
                    continue
                if self.get_label:
                    self.csv_data.append((in_row[col_path], in_row[col_label]))
                else:
                    self.csv_data.append((in_row[col_path],))

        # Load normalization data.
        if do_normalization:
            train_dataset = constants.get_dataset(Split.TRAIN, example)
            mu_path = train_dataset.norm_dir.joinpath(f"{input}.mu.pt")
            var_path = train_dataset.norm_dir.joinpath(f"{input}.var.pt")
            if not mu_path.exists() or not var_path.exists():
                msg = f"Cannot find {input}.mu.pt and {input}.var.pt in {train_dataset.norm_dir}."
                raise Exception(msg)
            mean = torch.load(mu_path)
            var = torch.load(var_path)
        else:
            mean = None
            var = None

        # Create transform.
        self.transform = make_transform(crop_type, input, feat_seq_len, mean, var)


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[index][0]
        if self.input == Input.AUDIO:
            features = load_audio(full_path(file_path), sampling_rate=16_000)
        else:
            features = torch.load(full_path(file_path))
        features = self.transform(features)
        
        # Load label.
        if self.get_label:
            label = torch.tensor(self.csv_data[index][1], dtype=self.dtype)
        else:
            label = torch.tensor(0, dtype=self.dtype)

        return features, label
