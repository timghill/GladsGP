"""
Compute and plot space- and/or time-integrated RMSE patterns

usage: plot_integrated_RMSE.py [-h]train_config test_config

"""

import os
import sys
import pickle
import argparse

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
from src.utils import import_config


def main(train_config, test_config, n_pcs):
    """
    Compute and plot space- and/or time-integrated RMSE patterns

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration
        
    """
    # Load data and initialize model
    fig = plt.figure(figsize=(6, 3))
    gs = GridSpec(len(n_pcs)+1, 2, height_ratios=[10] + len(n_pcs)*[100],
        width_ratios=(100, 100), left=0.1, bottom=0.125, 
        right=0.95, top=0.85, wspace=0.25, hspace=0.1)
    pcaxs = np.array([fig.add_subplot(gs[i+1,0]) for i in range(len(n_pcs))])
    tsax = fig.add_subplot(gs[1:, 1])
    cax = fig.add_subplot(gs[0, 0])
    alphabet = ['(a)', '(b)', '(c)', '(d)',]
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

        rmse_x = np.load(spatial_fname)
        rmse_t = np.load(timeseries_fname)

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
        'main/fig06.png'), dpi=400)
    fig.savefig(os.path.join(train_config.figures, 
        'main/fig06.pdf'), dpi=400)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    args = parser.parse_args()
    train_config = import_config(args.train_config)
    test_config = import_config(args.test_config)
    npcs = [2, 5, 8]
    main(train_config, test_config, npcs)
