# Data Directory

This directory contains the data required for reproducing the experiments in the `Reliability-Control-Streamflow-CP` project.

## 1. Raw Data (`raw/`)

Due to their large size, the raw datasets are **not included** in this repository. You must download them manually and place them in the appropriate subdirectories before running the experiments.

### CAMELS-US
- **Source**: [NCAR CAMELS Dataset](https://ral.ucar.edu/solutions/products/camels)
- **Directory**: `data/raw/CAMELS_US/`
- **Required Files**:
  - Download the basin timeseries (meteorological forcing and observed flow) and the basin attributes.
  - Extract the contents such that the daymet forcings and USGS streamflow data are accessible under this directory. The structure should be compatible with the `NeuralHydrology` CAMELS-US dataset loader.
  - Ensure the dataset is extracted here.

### CAMELS-GB
- **Source**: [NERC Environmental Data Centre](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9)
- **Directory**: `data/raw/CAMELS_GB/`
- **Required Files**:
  - The CAMELS-GB dataset files needed for the external generalization experiments (Exp 004 and CQR_GB).

## 2. Basin Lists

The `data/raw/` directory contains several `.txt` files which list the specific catchments used in our experiments:
- `exp002_basin_list_671.txt`: The primary 671 CAMELS-US basins used for the main models.
- `exp010_basin_list_clean.txt`: The clean subset used for specific ablations.

## 3. Preprocessing (`processed/` and `splits/`)

This project relies on the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework, which dynamically loads and preprocesses the CAMELS datasets during the training loop based on the configuration files (`config.yml`) located in the `experiments/` directory. 

You generally **do not need to manually run separate data preprocessing scripts**. Just ensure the raw data is placed in the correct directories as specified above, and NeuralHydrology will handle the rest (including standardizing features and creating the data splits defined by the water-year timelines in our configs).

## Git Ignore Policy

Please note that `data/raw/*`, `data/processed/*`, and `data/splits/*` are ignored by `.gitignore` to prevent accidentally committing massive multi-gigabyte files. Only the `.gitkeep` files and structural lists are tracked.
