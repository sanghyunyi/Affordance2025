# Affordance2025

Computational and neural mechanisms underlying the influence of action affordances on value learning.

There are two directories for each behavioral and fMRI experiment where the structures are identical.

## Directories

### `bandit_task`
- The PsychoPy code for running the behavioral/fMRI experiment.

### `data`
- Raw `.pkl` files and processed `.json` files of choice and reaction times are provided.
- Also can be downloaded in [OSF](https://osf.io/cvukp/?view_only=f2bd9f66ad604de09368eeec59996ea1)

### `model_fitting`
- We provide a CBM-based model fitting file used for the computational model fitting.

### `model_fitting/models`
- Model implementations. Any model name with `_2` in it refers to the reduced model.

## How to Run

### 0. Install
- Install [CBM](https://github.com/payampiray/cbm).

### 1. `indiv_fit_par_all.m`
- Fits all models to the data.

### 2. `model_based_var_generate.m` or `model_based_var_generate_simulation.m`
- Conducts a posterior predictive check or generates model-based variables given the fit parameters.

### 3. `generate_simulated_data_using_models.m` and `model_recov.m`
- Simulates models using the fit parameters and performs model recovery analysis.
- Use `model_recov_confusion_matrix.m` to get the confusion matrix.

### 4. `param_recov.m`
- Conducts parameter recovery analysis.

### 5. `param_fit_comparison.m`
- Compares the fit parameters from the original performance-based model and the reduced performance-based models.


## fMRI data

WIP
