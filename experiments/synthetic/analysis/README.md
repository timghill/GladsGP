# Analysis contents

This directory contains the functions and scripts used to fit, evaluate, and analyze GP emulators, including putting together the final figures for the paper.

## Table of contents
File | Description
---- | ---------------
`fit/`                      | Directory for parallel emulator fitting
`test/`                     | Directory for parallel emulator predictions on the test set
`fit_all_models.py`         | Fit GP models for different subsets of training data and different choices for the number of principal components (called from `fit/` directory)
`fit_scalar_models.py`      | Fit GP models for different subsets of training data and different choices for the number of principal components for scalar variables make scalar variable performance boxplots
`compute_test_error.py`     | Compute prediction error for different numbers of simulations and different choices for the number of principal components (called from `test/` directory)
`assess_all_models.py`      | Plot and sssess prediction error for different numbers of simulations and different choices for the number of principal components given a common test set 
`plot_test_error.py`        | Evaluate GP for GlaDS ensembles: timeseries, width-averaged test errors
`plot_PC_RMSE.py`           | Plot singular value proportion of variance, RMSE, and basis vectors
`plot_PC_maps.py`           | Plot GlaDS output, PC low-rank representation, and GP predictions
`plot_integrated_RMSE.py`   | Compute and plot space- and/or time-integrated RMSE patterns
`sensitivity_indices.py`    | Compute Sobol' indices for functional and scalar outputs
`tabulate_performance.py`   | Statistics to summarize emulator performance
`mcmc_diagnostics.py`       | Assess MCMC sampling by viewing traces and cmputing Rhat, ESS diagnostics
`mean_response.py`          | Compute the mean effect of each model input on each scalar variable

## Workflow

The following steps can be followed to re-fit the emulators and make figures. To make figures corresponding to the already fit emulators, run each python script without the `--recompute` flag.

### Fit flotation-fraction emulators and predict

First, fit emulators (~6 h):

```bash
cd fit
rm -rf MISC OUTPUT STATUSES TMP
submit.run 6
```

Then evaluate the MCMC sampling to ensure that the chains have converged (~2 h):

```bash
sbatch mcmc_diagnostics.sh
```

Once those jobs are completed, compute test predictions and store RMSE, MAPE, etc (~6 h):

```bash
cd test
rm-rf MISC OUTPUT STATUSES TMP farm.log slurm-*
submit.run 12
```

### Fit scalar emulators and compute test predictions

At the same time as the flotation-fraction jobs, fit the scalar emulators (~3 h):

```bash
sbatch fit_scalar_models.sh
```

Then compute the mean response surfaces (~minutes):
```bash
sbatch mean_response.sh
```

### Run GP assessment/evaluation

The `runme.sh` script calls a few different python scipts and only takes a few minutes:

```
sbatch runme.sh
```

### Sensitivity analysis

Last, if all the assessments look good, compute the sensitivity indices (~6 h):

```
sbatch sensitivity_indices.sh
```
