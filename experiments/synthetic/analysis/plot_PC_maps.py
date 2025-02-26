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

from src.utils import import_config

def main(train_config, test_config, p, m):
    train_config = import_config(train_config)
    test_config = import_config(test_config)

    U = np.load('data/models/pca_synthetic_n{:03d}_U.npy'.format(m)).astype(np.float32)
    S = np.load('data/models/pca_synthetic_n{:03d}_S.npy'.format(m)).astype(np.float32)
    Vh= np.load('data/models/pca_synthetic_n{:03d}_Vh.npy'.format(m)).astype(np.float32)

    U = U[:, :p]
    Sinv = np.diag(1./S[:p])
    S = np.diag(S[:p])
    Vh = Vh[:p]
    print(U.shape)
    print(S.shape)
    print(Vh.shape)
    print(Vh.dtype)

    sim_num = 24 # median RMSE simulation

    Y_test = np.load(test_config.Y_physical, mmap_mode='r').T
    print(Y_test.shape)

    Y_train = np.load(train_config.Y_physical, mmap_mode='r').T[:m]
    print(Y_train.shape)

    mu = np.mean(Y_train, axis=0)
    sd = np.std(Y_train, axis=0)
    sd[sd<1e-6] = 1e-6

    Z_train = (Y_train-mu)/sd
    Z_test = (Y_test-mu)/sd

    U_test = Z_test @ Vh.T @ Sinv
    Z_test_lowrank = U_test @ S @ Vh
    Y_test_lowrank = mu + sd*Z_test_lowrank
    print(Y_test_lowrank.shape)

    mean_pred = np.load('data/reference/pred_mean.npy')
    print('mean:', mean_pred.shape)

    mesh = np.load('../issm/data/geom/synthetic_mesh.pkl', allow_pickle=True)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements'].astype(int)-1)

    fig = plt.figure(figsize=(8, 4))
    # fig,axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    gs = GridSpec(4, 2, height_ratios=(10, 100, 100, 100),
        bottom=0.1, left=0.065, right=0.95, top=0.9,
        hspace=0.05, wspace=0.05,
    )
    axs = np.array([[fig.add_subplot(gs[i+1,j]) for j in range(2)] for i in range(3)])
    caxs = np.array([fig.add_subplot(gs[0,j]) for j in range(2)])

    triargs = dict(
        cmap=cmocean.cm.dense,
        vmin=0,
        vmax=1,
        rasterized=True,
    )

    diffargs = dict(
        cmap=cmocean.cm.balance,
        vmin=-0.09, vmax=0.09)

    tstep = 210
    axs[0,0].tripcolor(mtri, Y_test[sim_num, tstep::365], **triargs)
    axs[1,0].tripcolor(mtri, Y_test_lowrank[sim_num, tstep::365], **triargs)
    pctri = axs[2,0].tripcolor(mtri, mean_pred[sim_num, tstep::365], **triargs)

    axs[1,1].tripcolor(mtri, Y_test_lowrank[sim_num, tstep::365] - Y_test[sim_num, tstep::365],
        **diffargs)
    difftri = axs[2,1].tripcolor(mtri, mean_pred[sim_num, tstep::365] - Y_test[sim_num, tstep::365],
        **diffargs)

    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 25])
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks([0, 12.5, 25])
    
    for ax in axs[:-1].flat:
        ax.set_xticklabels([])
    for ax in axs[:, 1].flat:
        ax.set_yticklabels([])

    axs[-1,0].set_xlabel('Distance from terminus (km)')
    axs[-1,1].set_xlabel('Distance from terminus (km)')
    axs[1,0].set_ylabel('Distance across (km)')

    cbar0 = fig.colorbar(pctri, cax=caxs[0], orientation='horizontal')
    cbar1 = fig.colorbar(difftri, cax=caxs[1], orientation='horizontal')

    cbar0.set_label('Flotation fraction')
    cbar1.set_label(r'$\Delta$Flotation fraction')

    for cax in caxs:
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
    
    textx = 98
    labelx = 1
    texty = 24
    textargs = dict(fontweight='bold', fontsize=fs, ha='right', va='top')
    labelargs = dict(fontweight='bold', fontsize=fs, ha='left', va='top')
    axs[0,0].text(labelx, texty, '(a)', color='k', **labelargs)
    axs[1,0].text(labelx, texty, '(b)', color='k', **labelargs)
    axs[2,0].text(labelx, texty, '(c)', color='k', **labelargs)

    axs[0,0].text(textx, texty, 'GlaDS', color='w', **textargs)
    axs[1,0].text(textx, texty, 'PCs 1-8', color='w', **textargs)
    axs[2,0].text(textx, texty, 'GP', color='w', **textargs)

    axs[1,1].text(labelx, texty, '(d)', **labelargs)
    axs[2,1].text(labelx, texty, '(e)', **labelargs)

    axs[1,1].text(textx, texty, 'PCs 1-8 $-$ GlaDS', **textargs)
    axs[2,1].text(textx, texty, 'GP $-$ GlaDS', **textargs)

    axs[0,1].set_visible(False)

    plt.subplots_adjust(bottom=0.1, left=0.065, right=0.95, top=0.9,
        hspace=0.05, wspace=0.05)

    fig.savefig('figures/appendix/B03.png', dpi=400)
    fig.savefig('figures/appendix/B03.pdf', dpi=400)
    return


if __name__=='__main__':

    parser = argparse.ArgumentParser(description=''
    'Plot original GlaDS simulations, PC low-rank representation and GP predictions'
    )
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('npc', type=int)
    parser.add_argument('nsim', type=int)
    args = parser.parse_args()
    main(args.train_config, args.test_config, p=args.npc, m=args.nsim)