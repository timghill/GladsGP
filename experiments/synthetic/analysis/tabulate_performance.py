"""
Tabulate emulator performance: Table X
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.utils import import_config

# Short functions to compute metrics
def rmse(z1, z2):
    dz = z2 - z1
    return np.sqrt(np.mean(dz**2, axis=1))

def mape(z1, z2):
    lq = np.quantile(z2, 0.1)
    z2 = z2.copy()
    z2[z2<lq] = np.nan
    inner = (z2-z1)/z2
    inner[z2<lq] = np.nan
    return np.nanmean(np.abs(inner), axis=1)

def bias(z1, z2):
    return np.mean(z2-z1, axis=1)

def coefdet(z1, z2):
    SS_resid = np.sum((z2-z1)**2, axis=1)
    SS_tot = np.sum((z1 - z1.mean(axis=1)[:,None])**2, axis=1)
    R2 = 1 - SS_resid/SS_tot
    return R2

# Compute metrics for whole and subsets of predictions
def main(test_config):
    test_config = import_config(test_config)

    mesh = np.load(test_config.mesh, allow_pickle=True)

    # Define all-year timesteps and DJF, JJA subsets
    tsteps = np.arange(365)
    tsteps_djf = np.hstack((np.arange(60), np.arange(335, 365)))
    tsteps_jja = np.arange(152, 244)

    nt = len(tsteps)
    nx = mesh['numberofvertices']
    indices = np.arange(int(nt*nx)).reshape((nx, nt))

    # Define spatial and temporal subsets of the data
    masks = [
        indices.flatten(),
        indices[mesh['x']<=30e3, :].flatten(),
        indices[mesh['x']>30e3, :].flatten(),
        indices[:, tsteps_djf].flatten(),
        indices[:, tsteps_jja].flatten(),
    ]
    mask_labels = ['All', 'Below 30 km', 'Above 30 km', 'DJF', 'JJA']

    # Load pre-computed GP predictions
    gp_pred = np.load('data/reference/pred_mean.npy', mmap_mode='r')
    y_test = np.load(test_config.Y_physical, mmap_mode='r').T
    gp_err = gp_pred - y_test

    # Set up list of labels and functions to compute
    statistics_labels = ['rmse', 'mape', 'bias', 'R2']
    statistics_funs = [rmse, mape, bias, coefdet]
    perf = {}

    # Median, 5th- and 95th-percentiles
    quantiles = [0.5, 0.05, 0.95]
    quantiles_labels = ['median', '0.05', '0.95']

    # Compute each statistic for all subsets
    for i,mask in enumerate(masks):
        print(mask_labels[i])
        print(len(mask))
        for j,statistic in enumerate(statistics_funs):
            statval = statistic(y_test[:, mask], gp_pred[:, mask])
            statq = np.quantile(statval, quantiles).round(decimals=3)
            print('\t' + statistics_labels[j] + '\t', statq)
            for k in range(len(statq)):
                key = statistics_labels[j] + '_' + quantiles_labels[k]
                if key not in perf.keys():
                    perf[key] = []
                perf[statistics_labels[j] + '_' + quantiles_labels[k]].append(statq[k])
    
    # Write as a DataFrame for nice formatting
    df = pd.DataFrame(perf, index=mask_labels)
    with open('tabulate_performance.txt', 'w') as table:
        table.writelines(df.to_string() + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_config')
    args = parser.parse_args()
    main(args.test_config)
