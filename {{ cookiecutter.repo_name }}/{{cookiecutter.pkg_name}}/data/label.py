from {{cookiecutter.pkg_name}}.data.csv_info import STANDARDIZED_CSV_INFO

class Label:
    def __init__(self, name: str, is_integer: bool, column: int):
        self.name = name
        self.is_integer = is_integer
        self.column = column

    def __str__(self) -> str:
        return self.name


# TODO: define labels here
MOS = Label(name="mos", is_integer=False, column=STANDARDIZED_CSV_INFO.col_label)
NORM_MOS = Label(name="norm_mos", is_integer=False, column=STANDARDIZED_CSV_INFO.col_norm_label)
MOS_STATUS = Label(name="mos_status", is_integer=True, column=STANDARDIZED_CSV_INFO.col_status_label)

ALL_LABELS = [MOS, NORM_MOS, MOS_STATUS]
