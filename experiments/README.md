# Experiments Directory

This directory contains the specific configurations and output metrics for all computational experiments presented in the manuscript.

## Structure

Each sub-directory corresponds to a specific experiment or ablation study (e.g., `exp002`, `exp003`, `exp004`, `exp010`, etc.). 

Inside each experiment directory, you will typically find:
- **`config.yml`**: The exact NeuralHydrology configuration file used to define the model architecture, data splits (water-years), features, and training hyperparameters.
- **`results/`**: The parsed evaluation metrics (such as `metrics.json` or aggregated `.csv` files) tracking the conformal coverage, intervals, AR1 components, and Continuous Ranked Probability Score (CRPS).

## Note on Model Checkpoints

To keep the repository size manageable for distribution (the "Replication Package" snapshot), large binary files such as model weights (`*.pt`), `.p` evaluation dumps, and full TensorBoard event logs (`.tfevents`) have been deliberately excluded. 

The provided `config.yml` files contain all necessary information to exactly reproduce the training runs from scratch on your own hardware using the standard NeuralHydrology pipeline.
