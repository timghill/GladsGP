# Analysis contents

File | Description
---- | ---------------
`fit_all_models.py`         | Fit GP models for different subsets of training data and different choices for the number of principal components.
`fit_scalar_models.py`      | Fit GP models for different subsets of training data and different choices for the number of principal components for scalar variables make scalar variable performance boxplots.
`assess_all_models.py`      | Compute prediction error for different numbers of simulations and different choices for the number of principal components given a common test set
`plot_PC_RMSE.py`           | Plot singular value proportion of variance, RMSE, and basis vectors
`plot_integrated_RMSE.py`   | Compute and plot space- and/or time-integrated RMSE patterns
`plot_test_error.py`        | Evaluate GP for GlaDS ensembles: timeseries, width-averaged test errors
`sensitivity_indices.py`    | Compute Sobol' indices for functional and scalar outputs
`summary_statistics.py`     | Statistics to summarize emulator performance
`time_predictions.py`       | Compute time to make predictions in a fair way
`mcmc_diagnostics.py`       | Assess MCMC sampling chain convergence, posterior sampling