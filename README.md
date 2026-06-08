# Reliability Control for Streamflow Prediction using Conformal Prediction (HSCC)

This repository contains the official replication package for the paper investigating Reliability Control in Streamflow Prediction via Heteroscedastic Stratified Conformal Prediction (HSCC) and other baseline methods (such as Mixture Density Networks and Conformalized Quantile Regression).

## Repository Structure

- `data/`: Contains instructions for downloading the CAMELS-US and CAMELS-GB datasets, as well as the lists of basins used in our experiments. (See `data/README.md` for download details).
- `experiments/`: Contains the configurations (`config.yml`), tracking code, and quantitative results (metrics) for each experimental setup discussed in the paper.
- `scripts/`: Shared data setup and execution scripts (e.g., environment setup, AI evaluation integrations used during analysis).
- `src/`: Shared source code module components if applicable.
- `paper/`: Contains the manuscript source files (`.qmd` Quarto format), outputs, and the aggregated data references used directly in the text and figures.

## Environment Setup

The models are built and evaluated using the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework. 
To reproduce the environment:
1. Ensure you have `conda` installed.
2. Run `conda env create -f environment.yml` (or `environment-cuda.yml` for GPU).
3. Activate the environment: `conda activate reliability_cp`

## Running Experiments

Refer to the configurations in the `experiments/` directory to run the exact training setups. After downloading the datasets into `data/raw/`, you can use standard NeuralHydrology CLI commands, referencing the `config.yml` files.

## License

Please refer to the manuscript details for dataset licensing and usage constraints.
