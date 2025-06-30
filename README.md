# sip-deliverables

This repository is split up into all relevant steps in chronological order. Steps 0 (preprocessing) - 3 (testing).
Input, output data and models are stored in the `10_data` folder.
Satellite images are not provided via the git repository, only the output of the preprocessing due to size constraints.

1. Open a code editor of choice, for ease of use, the project provides a dev container
2. Run `uv venv && uv sync` if not executed automatically
3. (Optional) Upload tiff files (normalized satellite images) into the `/10_data/input/`

## How to run the training scripts
1. Open the virtual environment
2. Run python `1_training/train_yolo.py` or `1_training/train_resnet.py` using the provided environment

## How to run the trained models on the test dataset
1. Open the virtual environment
2. Run python `2_run/run_yolo.py` or `python 2_run/run_resnet.py`

## How to validate the trained models on the test dataset
1. Open the virtual environment
2. Run python `3_test/test_yolo.py` or python `3_test/test_resnet.py`

## Misc scripts
* Scripts used for plot generation can be found in 90_misc/plot_generation
* Script for geojson generation is stored in `90_misc/generate_geojson.py`

## Data
* 10_data/input contains tiff
* 10_data/labels contains all labeled images
* 10_data/models contains all models
* 10_data/output is the default output location for image split script
* 10_data/results is the default location for all generated csv, images and plots
* 10_data/runs is the default storage location for all training runs
* 10_data/validation-favorable contains the test-set for single trees
* 10_data/validation-random contains the test-set for groups of trees