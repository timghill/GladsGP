# Analysis contents

File | Description
---- | ---------------
`fit_all_models.py`         | Fit GP models for different subsets of training data and different choices for the number of principal components.
`fit_scalar_models.py`      | Fit GP models for different subsets of training data and different choices for the number of principal components for scalar variables make scalar variable performance boxplots.
`assess_all_models.py`      | Compute prediction error for different numbers of simulations and different choices for the number of principal components given a common test set
`plot_test_error.py`        | Evaluate GP for GlaDS ensembles: timeseries, width-averaged test errors
`plot_PC_RMSE.py`           | Plot singular value proportion of variance, RMSE, and basis vectors
`plot_PC_maps.py`           | Plot GlaDS output, PC low-rank representation, and GP predictions
`plot_integrated_RMSE.py`   | Compute and plot space- and/or time-integrated RMSE patterns
`sensitivity_indices.py`    | Compute Sobol' indices for functional and scalar outputs
`tabulate_performance.py`   | Statistics to summarize emulator performance
`mcmc_diagnostics.py`       | Assess MCMC sampling by viewing traces and cmputing Rhat, ESS diagnostics
`mean_response.py`          | Compute the mean effect of each model input on each scalar variable

## Workflow

### Fit flotation-fraction emulators and predict

First, fit emulators (~6 h):

```bash
cd fit
rm -rf MISC OUTPUT STATUSES TMP
submit.run 6
```

Then we need to evaluate the MCMC sampling to ensure that the chains have converged (~2 h):

```bash
sbatch mcmc_diagnostics.sh
```

Once those jobs are completed, we can make test predictions and store RMSE, MAPE, etc (~12 h):

```bash
cd test
rm-rf MISC OUTPUT STATUSES TMP farm.log slurm-*
submit.run 12
```

### Fit scalar emulators and compute test predictions

At the same time as the flotation-fraction jobs, we can fit the scalar emulators (~12 h):

```bash
sbatch fit_scalar_models.sh
```

Then compute the mean response surfaces:
```bash
python mean_response.py ../train_config.py --recompute
```

### Run GP assessment/evaluation

The `runme.sh` script calls a few different python scipts and only takes a few minutes:

```
sbatch runme.sh
```

### Sensitivity analysis

Last, if all the assessments look good, we can compute the sensitivity indices (~12 h):

```
sbatch sensitivity_indices.sh
```
