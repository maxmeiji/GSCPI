# GSCPI Project


## Introduction

The github repository is for Time series foundation mdoel implementation. We currently have two foundation models to use: TimesFM (ICML 2024) and TimeGPT-1 (arXiv preprint).
The dataset is GSCPI (Glabal Supply Chain Pressure Index) from the Federal Reserve Bank.


## File Structure
```
GSCPI/
    data/
    data_provider/
    model/
        tfm_model.py
        timegpt1_model.py
    pic/
        TimeGPT1/
        TimesFM/
    FM.py
    TimeGPT1_GSCPI.sh
    TimesFM_GSCPI.sh

```
- `model/`: code for different foundation models
- `pic/`: save output figures for different models
- `FM.py`: main.py for data loading, model loading, and inferencing.
- `.sh files/`: running scripts.


## Installation

There are two ways to install required packages

1.Create a conda environment with specific python version, and activate it:
    
    ```
    conda create -n TFM python=3.10
    conda activate TFM
    ```

    Install necessary packages specified in requirement.txt:
    ```
    pip install -r requirement.txt
    ```

2. Refer to each foundation model's requirements

    - For TimesFM (ICML, 2024), install the neccessary packages from https://github.com/google-research/timesfm

    - For TimeGPT-1 (arXiv preprint, 2024), install the neccessary package from https://docs.nixtla.io/docs/getting-started-timegpt_quickstart



## Run

For running this code, please follow the instructions below

Change working directory to GSCPI:
```
cd GSCPI
```

Execute the inference file:
```
./{model_name}.sh
```


## Data Description

Refer to the data_provider directory for detailed information about how to load the dataset.

The input data should include a 'Date' column and at least one additional feature.
