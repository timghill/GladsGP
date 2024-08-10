"""
Compute and plot space- and/or time-integrated RMSE patterns

usage: plot_integrated_RMSE.py [-h] [--recompute] train_config test_config

"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.tri import Triangulation
from matplotlib import colors
import cmocean

from sepia.SepiaPredict import SepiaEmulatorPrediction

from src.model import load_model


def main(train_config, test_config, n_pcs, recompute=False, dtype=np.float32):
    """
    Compute and plot space- and/or time-integrated RMSE patterns

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration
    
    recompute : bool, optional
                Force to recompute integrated RMSE values and overwrite on disk?

    dtype : type, optional
            Type to cast simulation outputs into, e.g. np.float32
        
    """
    # Load data and initialize model
    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1).astype(dtype)
    t_std = t_std[:train_config.m, :]
    t_phys = t_phys[:train_config.m, :]
    y_sim = np.load(train_config.Y_physical).T[:train_config.m, :].astype(dtype)
    exp_name = train_config.exp

    t_test_std = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)[:test_config.m :]
    t_test_phys = np.loadtxt(test_config.X_physical, delimiter=',',
        skiprows=1).astype(dtype)[:test_config.m, :]
    y_test_sim = np.load(test_config.Y_physical).T[:test_config.m, :].astype(dtype)

    fig = plt.figure(figsize=(6, 3))
    gs = GridSpec(len(n_pcs)+1, 2, height_ratios=[10] + len(n_pcs)*[100],
        width_ratios=(100, 100), left=0.1, bottom=0.125, 
        right=0.95, top=0.85, wspace=0.25, hspace=0.1)
    pcaxs = np.array([fig.add_subplot(gs[i+1,0]) for i in range(len(n_pcs))])
    tsax = fig.add_subplot(gs[1:, 1])
    cax = fig.add_subplot(gs[0, 0])
    alphabet = ['a', 'b', 'c', 'd',]
    colors = ['#000000', '#555555', '#aaaaaa']

    # Plot test error: width-averaged, space, and time error
    with open(os.path.join(train_config.sim_dir, train_config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements'] - 1
            
    data_dir = 'data/architecture'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_config.figures):
        os.makedirs(train_config.figures)

    for i,p in enumerate(n_pcs):
        timeseries_fname = os.path.join(data_dir, 
            'rmse_timeseries_n{}_p{}.npy'.format(train_config.m, p))
        spatial_fname = os.path.join(data_dir, 
            'rmse_spatial_n{}_p{}.npy'.format(train_config.m, p))
        if not os.path.exists(timeseries_fname) or not os.path.exists(spatial_fname) or recompute:
            data, model = load_model(train_config, train_config.m, p)

            # Compute test predictions and error
            samples = model.get_samples(numsamples=32, nburn=256)
            for key in samples.keys():
                samples[key] = samples[key].astype(dtype)
            Y_preds = np.zeros_like(y_test_sim)
            for k in range(test_config.m):
                print('k:', k)
                xi = t_test_std[k:k+1]
                pred = SepiaEmulatorPrediction(samples=samples, model=model,
                    t_pred=xi)
                pred.w = pred.w.astype(np.float32)
                y_pred = pred.get_y()
                y_pred = y_pred.mean(axis=0)
                Y_preds[k] = y_pred

            test_error = Y_preds - y_test_sim
            # Integrated over time and space
            nx = nodexy.shape[0]
            nt = int(test_error.shape[1]/nx)
            dim_separated_test_error = np.zeros((test_config.m, nx, nt), dtype=dtype)
            for k in range(test_config.m):
                dim_separated_test_error[k, :, :] = test_error[k, :].reshape((nx, nt))
            rmse_x = np.sqrt(np.nanmean(dim_separated_test_error**2, axis=(0,2)))
            rmse_t = np.sqrt(np.nanmean(dim_separated_test_error**2, axis=(0, 1)))

            np.save(timeseries_fname, rmse_t)
            np.save(spatial_fname, rmse_x)
        
        rmse_t = np.load(timeseries_fname)
        rmse_x = np.load(spatial_fname)


        mtri = Triangulation(nodexy[:, 0]/1e3, nodexy[:, 1]/1e3, connect)
        ax2 = pcaxs[i]
        tpc = ax2.tripcolor(mtri, rmse_x, vmin=0, vmax=0.15, 
            cmap=cmocean.cm.matter, rasterized=True)
        ax2.set_aspect('equal')
        ax2.set_xlim([0, 100])
        ax2.set_ylim([0, 25])
        ax2.set_yticks([0, 12.5, 25])
        ax2.text(0.95, 0.95, alphabet[i], transform=ax2.transAxes,
            fontweight='bold', ha='right', va='top')
        ax2.text(0.95, 0.05, 'p={}'.format(p), transform=ax2.transAxes,
            ha='right', va='bottom')

        t_month = np.arange(365) * 12/365
        ax1 = tsax
        ax1.plot(t_month, rmse_t, label='p={}'.format(p), color=colors[i], linewidth=1)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('RMSE')
        ax1.set_xlim([0, 12])
        ax1.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax1.grid(linestyle=':', linewidth=0.5)
    
    for ax in pcaxs[:-1]:
        ax.set_xticklabels([])
    pcaxs[-1].set_xlabel('Distance from terminus (km)')
    pcaxs[1].set_ylabel('Distance across (km)')

    cb = fig.colorbar(tpc, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cb.set_label('RMSE')
    ax1.text(0.025, 0.95, alphabet[len(n_pcs)], transform=ax1.transAxes,
        fontweight='bold', ha='left', va='top')
    ylim = ax1.get_ylim()
    ax1.set_ylim([0.0, ylim[1]])
    ax1.legend(loc='upper right', frameon=False)
    fig.savefig(os.path.join(train_config.figures, 
        'rmse_timeseries_spatial.png'), dpi=400)
    fig.savefig(os.path.join(train_config.figures, 
        'rmse_timeseries_spatial.pdf'), dpi=400)


if __name__=='__main__':
    import argparse
    from src.utils import import_config
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    train_config = import_config(args.train_config)
    test_config = import_config(args.test_config)
    npcs = [2, 5, 8]
    main(train_config, test_config, npcs, recompute=args.recompute)
