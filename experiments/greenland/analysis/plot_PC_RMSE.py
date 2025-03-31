"""
usage: plot_PC_maps.py [-h] train_config test_config npc nsim

Plot original GlaDS simulations, PC low-rank representation and GP predictions

positional arguments:
  train_config
  test_config
  npc
  nsim

options:
  -h, --help    show this help message and exit
"""

import os
import pickle
import argparse

import numpy as np
import scipy

fs = 8
import matplotlib
matplotlib.rc('font', size=fs)

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.tri import Triangulation
import cmocean

from src.utils import import_config, width_average
from src.svd import randomized_svd
from src.model import load_model
from src import utils

def compute_truncation_error(usv, y_mean, y_sd, y_sim):
    """
    Compute truncated SVD error.

    Parameters
    ----------
    usv: tuple (U, S, V)
         where U,S,V are (truncated) singular value decomposition of y_sim
    
    y_mean : array-like
             Mean of y_sim
    
    y_sd : array-like
           Standard devaition of y_sim
    
    y_sim : (m, n_x*n_t) array-like
            Simulation output matrix in physical units
    """
    U,S,V = usv
    y_error = y_sim - y_mean - y_sd*(U @ S @ V)
    y_rmse = np.linalg.norm(y_error, ord='fro')/np.sqrt(y_sim.size)
    return y_rmse

def plot_PC_RMSE_variance(train_config, n_sims, recompute=False):
    """
    Plot PC RMSE, cumulative proportion of variance and first 7 basis vectors.

    Parameters
    ----------
    recompute : bool, optional
                Force recompute PC error and overwrite on disk?
    """
    ## Part 1: plot RMSE and cumulative proportion of variance
    npcs = list(np.arange(1, 12)) + list((np.linspace(1, 10, 21)**2).astype(int))
    npcs = np.unique(npcs)
    pmax = 100

    y_full = np.load(train_config.Y_physical, mmap_mode='r')
    y_full = y_full[:, :n_sims[-1]].T.astype(np.float32)
    for j in range(len(n_sims)):
        cvar_fname = 'data/architecture/pca_cvar_n{}.csv'.format(n_sims[j])
        rmse_fname = 'data/architecture/pca_rmse_n{}.csv'.format(n_sims[j])
        if not os.path.exists(rmse_fname) or recompute:
            y_sim = y_full[:n_sims[j]]
            print('y_sim.shape', y_sim.shape)
            y_mean = np.mean(y_sim, axis=0)
            y_sd = np.std(y_sim, ddof=1, axis=0)
            y_sd[y_sd<1e-6] = 1e-6
            y_std = (y_sim - y_mean)/y_sd
            pj = min(pmax, n_sims[j])
            usv = randomized_svd(y_std, p=pj, k=0, q=1)
            U,S,Vh = usv
            cvar = np.cumsum(S**2)/np.sum(S**2)
            pca_rmse = np.zeros(npcs.shape)

            for k in range(len(npcs)):
                print('Computing for %d PCs' % npcs[k])
                U_ = U[:,:npcs[k]]
                S_ = np.diag(S[:npcs[k]])
                Vh_ = Vh[:npcs[k],:]
                yhat = U_ @ (S_ @ Vh_)
                pca_rmse[k] = compute_truncation_error((U_,S_,Vh_), y_mean, y_sd, y_sim)
            
            writedata = np.array([npcs, pca_rmse]).T
            np.savetxt(rmse_fname, writedata, fmt='%.6e')
            
            np.savetxt(cvar_fname, np.array([np.arange(1, pj+1), cvar]).T,
                fmt='%.6e')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=''
    'Plot singular value proportion of variance, RMSE, and basis vectors'
    )
    parser.add_argument('train_conf')
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    args = parser.parse_args()
    train_config = utils.import_config(args.train_conf)
    plot_PC_RMSE_variance(train_config, args.nsim, recompute=False)
