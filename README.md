# Computational and Neural Mechanisms Underlying the Influence of Action Affordances on Value Learning

This repository contains resources and materials related to our study on the computational and neural mechanisms by which action affordances shape value learning.

### Preprint

The preprint of our study is available on bioRxiv:

📄 [Affordance2025 Preprint](https://www.biorxiv.org/content/10.1101/2023.07.21.550102v3.abstract)

---

## Directories
There are two directories for each behavioral and fMRI experiment where the structures are identical.

### `analysis`
- Analysis codes used to generate the plots for the paper.
- The `lap_out` directory (for storing model fitting results) and the `model_based_var` directory (for storing model-based variables and simulation results) in `model_fitting`, generated from running `model_fitting`, are needed to perform computational model analysis.

### `bandit_task`
- The PsychoPy code for running the behavioral/fMRI experiment.

### `data`
- Raw `.pkl` files and processed `.json` files of choice and reaction times are provided.
- Can also be downloaded from [OSF](https://osf.io/cvukp/?view_only=f2bd9f66ad604de09368eeec59996ea1)

### `model_fitting`
- We provide a CBM-based model fitting file used for the computational model fitting.

### `model_fitting/models`
- Model implementations. Any model name with `_2` in it refers to the reduced model.

---

## How to Run

### 0. Install
- Install [CBM](https://github.com/payampiray/cbm).

### 1. `indiv_fit_par_all.m`
- Fits all models to the data.
- Saves results in `lap_out`

### 2. `model_based_var_generate.m` or `model_based_var_generate_simulation.m`
- Conducts a posterior predictive check or generates model-based variables given the fit parameters.
- Saves results in `model_based_var`

### 3. `generate_simulated_data_using_models.m` and `model_recov.m`
- Simulates models using the fit parameters and performs model recovery analysis.
- Use `model_recov_confusion_matrix.m` to get the confusion matrix.

### 4. `param_recov.m`
- Conducts parameter recovery analysis.

### 5. `param_fit_comparison.m`
- Compares the fit parameters from the original performance-based model and the reduced performance-based models.

---

## fMRI data

WIP

---

### Citation

If you use or reference this work, please cite the preprint appropriately.

```bibtex
@article {Yi2023.07.21.550102,
	author = {Yi, Sanghyun and O{\textquoteright}Doherty, John P.},
	title = {Computational and neural mechanisms underlying the influence of action affordances on value learning},
	elocation-id = {2023.07.21.550102},
	year = {2024},
	doi = {10.1101/2023.07.21.550102},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/03/06/2023.07.21.550102},
	eprint = {https://www.biorxiv.org/content/early/2024/03/06/2023.07.21.550102.full.pdf},
	journal = {bioRxiv}
}
```

### Contact

For any questions or collaborations, feel free to reach out.


