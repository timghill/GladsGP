"""
Tabulate emulator performance: Table X
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.utils import import_config

def rmse(z):
    return np.sqrt(np.mean(z**2, axis=1))

def mape(z1, z2):
    lq = np.quantile(z2, 0.1)
    z2 = z2.copy()
    z2[z2<lq] = np.nan
    inner = (z2-z1)/z2
    inner[z2<lq] = np.nan
    return np.nanmean(np.abs(inner), axis=1)

def bias(z):
    return np.mean(z, axis=1)

def coefdet(z1, z2):
    SS_resid = np.sum((z1-z2)**2, axis=1)
    SS_tot = np.sum((z2 - z2.mean(axis=1)[:,None])**2, axis=1)
    R2 = 1 - SS_resid/SS_tot
    return R2

def main(train_config, test_config, m, p):
    train_config = import_config(train_config)
    test_config = import_config(test_config)

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
    perf['all_rmse'] = rmse(gp_err)
    perf['all_mape'] = mape(gp_pred, y_test)
    perf['all_bias'] = bias(gp_err)
    perf['all_R2'] = coefdet(gp_pred, y_test)
    print(perf['all_rmse'].shape)
    print(perf['all_mape'].shape)

    print('RMSE:', np.sqrt(np.mean(perf['all_rmse']**2)))
    print('MAPE:', np.mean(perf['all_mape']))
    print('bias:', perf['all_bias'])
    print('coefdet:', perf['all_R2'])
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--nsim', nargs='+', type=int)
    parser.add_argument('--npc', nargs='+', type=int)
    args = parser.parse_args()
    main(args.train_config, args.test_config, args.nsim, args.npc)
