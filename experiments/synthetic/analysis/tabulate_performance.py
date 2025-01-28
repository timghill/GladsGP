"""
Tabulate emulator performance: Table X
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.utils import import_config

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

def main(train_config, test_config, m, p):
    train_config = import_config(train_config)
    test_config = import_config(test_config)

    mesh = np.load(train_config.mesh, allow_pickle=True)
    print(mesh)

    tsteps = np.arange(365)
    tsteps_djf = np.hstack((np.arange(60), np.arange(335, 365)))
    tsteps_jja = np.arange(152, 244)

    nt = len(tsteps)
    nx = mesh['numberofvertices']
    indices = np.arange(int(nt*nx)).reshape((nx, nt))

    masks = [
        indices.flatten(),
        indices[mesh['x']<=30e3].flatten(),
        indices[mesh['x']>30e3].flatten(),
        indices[:, tsteps_djf].flatten(),
        indices[:, tsteps_jja].flatten(),
    ]
    mask_labels = ['All', 'Below 30 km', 'Above 30 km', 'DJF', 'JJA']

    gp_pred = np.load('data/reference/cv_mean.npy', mmap_mode='r')
    y_test = np.load(test_config.Y_physical, mmap_mode='r').T
    gp_err = gp_pred - y_test

    print(gp_pred.shape)
    print(y_test.shape)

    # data = {'data1':np.random.random((100)), 'data2':np.random.random((100))}
    perf = {}
    # perf['all_rmse'] = np.sqrt(np.mean(gp_err**2, axis=1))
    # lq = np.quantile(y_test, 0.1)
    # mape = np.abs(gp_err/y_test)
    # mape[y_test<lq] = np.nan
    # perf['all_mape'] = np.nanmean(mape, axis=1)

    statistics_labels = ['rmse', 'mape', 'bias', 'R2']
    statistics_funs = [rmse, mape, bias, coefdet]
    # perf = {'rmse':[], 'mape':[], 'bias':[], 'R2':[]}
    perf = {}

    quantiles = [0.5, 0.05, 0.95]
    quantiles_labels = ['median', '0.05', '0.95']

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
    
    df = pd.DataFrame(perf, index=mask_labels)
    with open('table_summary.txt', 'w') as table:
        table.writelines(df.to_string() + '\n')

    # subsets
    # for i in range(len(statistics_labels)):

    # perf['all_rmse'] = rmse(gp_err)
    # perf['all_mape'] = mape(gp_pred, y_test)
    # perf['all_bias'] = bias(gp_err)
    # perf['all_R2'] = coefdet(gp_pred, y_test)
    # print(perf['all_rmse'].shape)
    # print(perf['all_mape'].shape)

    # # print('RMSE:', np.sqrt(np.mean(perf['all_rmse']**2)))
    # # print('MAPE:', np.mean(perf['all_mape']))
    # # print('bias:', perf['all_bias'])
    # # print('coefdet:', perf['all_R2'])

    # keys = ['all_rmse', 'all_mape', 'all_bias', 'all_R2']
    # for key in keys:
    #     print(key)
    #     vals = perf[key]
    #     print(np.median(vals), np.quantile(vals, 0.05), np.quantile(vals, 0.95))
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--nsim', nargs='+', type=int)
    parser.add_argument('--npc', nargs='+', type=int)
    args = parser.parse_args()
    main(args.train_config, args.test_config, args.nsim, args.npc)
