"""
Compute prediction error for different numbers of simulations and different
choices for the number of principal components given a common test set

usage: assess_all_models.py [-h] --npc NPC [NPC ...] --nsim NSIM [NSIM ...] [--recompute] [--test]
                            train_conf test_conf

"""

import os
import argparse
import time

fs = 8
import matplotlib
matplotlib.rc('font', size=fs)

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction

from src import utils
from src.model import load_model

target_p = [2, 5, 8]

def compute_test_error(train_config, test_config, n_sims, n_pcs, 
    quantile=0.025, dtype=np.float32, test=False):
    """
    Compute test error for specified GPs.

    Produces the CSV files read by the various plotting functions.

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration
                
    n_sims : array
             List of numbers of simulations
    
    n_pcs : array
             List of numbers of PCs
    
    quantile : float, optional
               Compute prediction uncertainty between [quantile, 1-quantile]
    
    dtype : type, optional
            Type to cast simulation outputs into, e.g. np.float32
    
    test : bool, optional
           Development only! Use only a few MCMC samples and integration points
           to enable faster development. Do not use for making real predictions!
    """

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(train_config.X_standard, delimiter=',', max_rows=1,
        dtype=str, comments=None)

    x_pred = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1,
        comments=None)[:test_config.m].astype(dtype)
    y_test = np.load(test_config.Y_physical).T.astype(dtype)
    
    data_dir = os.path.join(train_config.data_dir, 'architecture')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_pattern = os.path.join(data_dir, 'performance_n{:03d}_p{:02d}.csv')

    for i in range(len(n_sims)):
        for k in range(len(n_pcs)):
            m = n_sims[i]
            p = n_pcs[k]
            print('m={}, p={}'.format(m,p))
            ti_std = t_std[:m, :]
            # yi_phys = y_sim[:m, :]
            sepia_data, model = load_model(train_config, m, p)
            print(sepia_data)

            n = model.data.sim_data.y.shape[1]
            mu_y = np.mean(model.data.sim_data.y, axis=0)
            sd_y = np.std(model.data.sim_data.y, ddof=1, axis=0)
            sd_y[sd_y<1e-6] = 1e-6

            if test:
                samples = model.get_samples(16, nburn=2500)
            else:
                samples = model.get_samples(256, nburn=2500)
            
            for key in samples.keys():
                samples[key] = samples[key].astype(dtype)
            ypred_mean = np.zeros((test_config.m, n), dtype=dtype)
            ypred_lq = np.zeros((test_config.m, n), dtype=dtype)
            ypred_uq = np.zeros((test_config.m, n), dtype=dtype)

            # Loop over test points, make predictions in batches (reduce
            # memory usage, probably a little slower)
            n_per_batch = 1
            n_batches = int(np.ceil(len(x_pred)/n_per_batch))
            batch_indices = np.array_split(np.arange(len(x_pred)), n_batches)
            print('Using {} batches of ~{}'.format(n_batches, n_per_batch))
            dt = np.zeros(n_batches)
            for j in range(n_batches):
                print('Test Batch {}/{}'.format(j+1, n_batches), end='\t')
                tj_pred = x_pred[batch_indices[j],:]
                t0 = time.perf_counter()
                preds = SepiaEmulatorPrediction(t_pred=tj_pred, 
                    samples=samples, model=model)
                preds.w = preds.w.astype(np.float32)
                ypreds = preds.get_y()
                t1 = time.perf_counter()
                error_preds = np.zeros(ypreds.shape, dtype=np.float32)
                for l_pred in range(len(batch_indices[j])):
                    for l_sample in range(error_preds.shape[0]):
                        err_sd = 1/np.sqrt(samples['lamWOs'][l_sample])
                        error_preds[l_sample][l_pred] = sd_y*np.random.normal(scale=err_sd)
                dt[j] = t1 - t0
                print('dt = {:.2f}s'.format(dt[j]))
                ypred_mean[batch_indices[j]] = np.mean(ypreds, axis=0)
                ypred_lq[batch_indices[j]] = np.quantile(ypreds + error_preds, quantile, axis=0)
                ypred_uq[batch_indices[j]] = np.quantile(ypreds + error_preds, 1-quantile, axis=0)

            # Save full predictions + intervals for the reference emulator
            if m==train_config.m and p==train_config.p:
                ref_dir = 'data/reference/'
                if not os.path.exists(ref_dir):
                    os.makedirs(ref_dir)
                np.save(os.path.join(ref_dir, 'pred_mean.npy'), ypred_mean)
                np.save(os.path.join(ref_dir, 'pred_lower.npy'), ypred_lq)
                np.save(os.path.join(ref_dir, 'pred_upper.npy'), ypred_uq)
                print('REFERENCE MODEL:')
                print('Mean dt = {:.4f}s'.format(np.mean(dt)))

            # Compute statistics and save results
            pred_resid = ypred_mean - y_test
            pred_rmse = np.sqrt(np.mean(pred_resid**2, axis=1))

            if p in target_p and m==train_config.m:
                out_sp = 'data/architecture/rmse_spatial_n{}_p{}.npy'.format(m,p)
                out_ts = 'data/architecture/rmse_timeseries_n{}_p{}.npy'.format(m,p)

                nt = 365
                nx = int(n/nt)

                rmse = np.sqrt(np.mean(pred_resid**2, axis=0)).reshape((nx, nt))
                rmse_ts = np.sqrt(np.mean(rmse**2, axis=0))
                rmse_sp = np.sqrt(np.mean(rmse**2, axis=1))

                np.save(out_sp, rmse_sp)
                np.save(out_ts, rmse_ts)


            print('RMSE:', np.sqrt(np.mean(pred_rmse**2)))
            lq = np.quantile(y_test, 0.1)
            inner_mape = np.abs(pred_resid/y_test)
            inner_mape[y_test<lq] = np.nan
            pred_mape = np.nanmean(inner_mape, axis=1)

            is_covered = np.logical_and(
                y_test>=ypred_lq, y_test<=ypred_uq)
            frac_covered = is_covered.sum(axis=1)/is_covered.shape[1]

            pred_lq = np.mean(ypred_lq, axis=1)
            pred_uq = np.mean(ypred_uq, axis=1)

            pred_arr = np.array([
                pred_rmse, 
                pred_mape, 
                pred_lq, 
                pred_uq, 
                # confint, 
                frac_covered
            ]).T
            csv_file = csv_pattern.format(m, p)
            pred_header = 'RMSE,MAPE,Lower quantile,Upper quantile,Fraction covered'
            np.savetxt(csv_file, pred_arr, header=pred_header,
                delimiter=',', fmt='%.6e')
    return
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_conf')
    parser.add_argument('test_conf')
    parser.add_argument('--npc', nargs='+', type=int, required=True)
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    parser.add_argument('--test', '-t', action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_conf)
    test_config = utils.import_config(args.test_conf)
    compute_test_error(train_config, test_config, n_sims=args.nsim, n_pcs=args.npc,
        test=args.test)
    
