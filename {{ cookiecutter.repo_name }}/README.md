{{cookiecutter.repo_name}}
==============================

## 1. Overview

{{cookiecutter.description}}

**RMSE Results (lower is better):**
|                                  | Baseline1 | Baseline2 |    Ours    |
|----------------------------------|:---------:|:---------:|:----------:|
| Validation RMSE (All Corpora)    |   0.8224  |   0.5475  | **0.5000** |
| Validation RMSE (PSTN & Tencent) |   0.6614  |   0.4965  | **0.4759** |
| Test RMSE (PSTN & Tencent)       |   0.745   |   0.543   |  **0.344** |


## 2. Installation
*Note: this software was developed for Linux.*

**Clone Repository**
```
git clone https://github.com/btamm12/{{cookiecutter.repo_name}}.git
```

**Google Drive Credentials**

To download the IU Bloomington from Google Drive, you need to download your
Google Drive API credentials.

1. Follow [these
   instructions](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
   to create a Google Cloud project and Service Key.
   After following these instructions, you will have downloaded a JSON file
   containing your Google credentials. Place this JSON file in the following
   location:
   ```
   {{cookiecutter.repo_name}}/gdrive_creds.json
   ```
2. Go to [this
   link](https://console.developers.google.com/apis/library/drive.googleapis.com)
   to enable Google Drive API for this project.
3. Wait 5 minutes for changes to propagate through Google systems.

## 3. Reproducing Results

Run the following commands to reproduce the results.

**1. Create Virtual Environment**
```
cd {{cookiecutter.repo_name}}
make create_environment
source venv/bin/activate
make requirements
```

**2. Download Datasets**
```
make download
```

**3. Perform Feature Extraction (GPU Recommended)**
```
make features
```

**4. Calculate Norm/Variance**
```
make norm
```

**6. Train Models (GPU Recommended)**
```
make train
```

**7. Predict Models on Validation Set**
```
make predict
```

**8. Follow the [README file](src/eval/README.md) in the `src/eval/` folder to copy
the ground-truth files and prediction files to the correct locations.**

**9. Evaluate Models on Validation Set**
```
make eval
```



## 4. Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make download` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final datasets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── submission     <- The postprocessed predictions that are ready for submission.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── {{cookiecutter.pkg_name}}
        │
        ├── data           <- Scripts to download and preprocess data, extract features and perform postprocessing.
        │
        ├── eval           <- Scripts to evaluate model performance on the validation set.
        │
        ├── model          <- Model definition and configurations.
        │
        ├── predict        <- Calculate model prediction on validation/test splits.
        │
        ├── train          <- Train model.
        │
        ├── utils          <- General utility functions.
        │
        └── visualization  <- Scripts to create exploratory and results-oriented visualizations.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
