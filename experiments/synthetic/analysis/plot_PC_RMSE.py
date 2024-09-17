"""
Plot singular value proportion of variance, RMSE, and basis vectors
"""

import os
import pickle

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

def plot_PC_RMSE_variance(recompute=False):
    """
    Plot PC RMSE, cumulative proportion of variance and first 7 basis vectors.

    This function (c.f. others) assumes config file positions.

    Parameters
    ----------
    recompute : bool, optional
                Force recompute PC error and overwrite on disk?
    """
    ## Part 1: plot RMSE and cumulative proportion of variance
    config = import_config('../train_config.py')
    y_fname = config.Y_physical
    nsims = [16, 32, 64, 128, 256, 512]
    npcs = list(np.arange(1, 12)) + list((np.linspace(1, 10, 21)**2).astype(int))
    npcs = np.unique(npcs)
    pmax = 100

    # colors = cmocean.cm.haline(np.linspace(0.2, 0.85, len(nsims)))
    colors = cmocean.cm.deep(np.linspace(0.15, 0.9, len(nsims)))
    fig = plt.figure(figsize=(6, 3.75))
    gs_global = GridSpec(1, 2, left=0.1, bottom=0.125, top=0.95, right=0.95,
        hspace=0.0, wspace=0.15,
        width_ratios=(60, 100))
    gs_left = GridSpecFromSubplotSpec(2, 1, gs_global[0,0], hspace=0.05)
    ax1 = fig.add_subplot(gs_left[0])
    ax2 = fig.add_subplot(gs_left[1])
    axs = np.array([ax1,ax2])

    y_full = np.load(y_fname, mmap_mode='r')
    y_full = y_full[:, :nsims[-1]].T.astype(np.float32)
    for j in range(len(nsims)):
        cvar_fname = 'data/architecture/pca_cvar_n{}.csv'.format(nsims[j])
        rmse_fname = 'data/architecture/pca_rmse_n{}.csv'.format(nsims[j])
        if not os.path.exists(rmse_fname) or recompute:
            y_sim = y_full[:nsims[j]]
            print('y_sim.shape', y_sim.shape)
            y_mean = np.mean(y_sim, axis=0)
            y_sd = np.std(y_sim, ddof=1, axis=0)
            y_sd[y_sd<1e-6] = 1e-6
            y_std = (y_sim - y_mean)/y_sd
            pj = min(pmax, nsims[j])
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

        writedata = np.loadtxt(rmse_fname)
        pca_rmse = writedata[:, 1]
        cvar = np.loadtxt(cvar_fname)
        nsim_axis = cvar[:, 0]
        cvar = cvar[:, 1]

        label =nsims[j]
        ax1.plot(npcs, pca_rmse, label=label, color=colors[j])
        ax2.plot(nsim_axis, cvar, label=label, color=colors[j])

    for ax in axs:
        ax.grid(linestyle=':')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim([1, pmax])
    
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('Cumulative proportion of variance')
    ax1.legend(frameon=False, ncols=2, loc='upper right')

    ax1.set_xticklabels([])
    ax2.set_xlabel('Number of PCs')

    ax1.text(0.025, 1.025, 'a1', transform=ax1.transAxes,
        fontweight='bold', ha='left', va='top')
    ax2.text(0.025, 1.025, 'a2', transform=ax2.transAxes,
        fontweight='bold', ha='left', va='top')
    
    ## Part 2: Plot basis vectors
    data, model = load_model(config, config.m, config.p, dtype=np.float32)
    K = data.sim_data.K.astype(np.float32)
    pc_cumulative_var = np.loadtxt('data/architecture/pca_cvar_n{}.csv'.format(config.m))[:, 1]
    pcvar = np.diff(pc_cumulative_var, prepend=0)
    nplot = 7
    ncols = 2
    nrows = int(np.ceil(nplot/ncols))
    gs_right = GridSpecFromSubplotSpec(nrows, ncols, gs_global[1],
        hspace=0.3)
    cgs = GridSpecFromSubplotSpec(5, 1, gs_right[3,1])
    cax = fig.add_subplot(cgs[-1])
    axs = np.array([[fig.add_subplot(gs_right[i,j]) for j in range(ncols)]
        for i in range(nrows)])
    
    with open(os.path.join(config.sim_dir, config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    
    for i in range(nplot):
        ax = axs.flat[i]
        nx = len(mesh['x'])
        nt = int(len(K[i])/nx)
        kavg, xe = width_average(mesh, K[i].reshape(nx, nt))
        t = np.arange(0, 365+1) * 12/365
        [xx,tt] = np.meshgrid(xe, t)
        pcol = ax.pcolormesh(xx, tt, kavg.T, cmap=cmocean.cm.delta, 
            vmin=-1.1, vmax=1.1, rasterized=True)

        # ax.set_aspect('equal')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 12])

        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 3, 6, 9, 12])
        ax.set_yticklabels(['Jan', 'Apr', 'Jul', 'Oct', ''])

        if i<nplot-2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Distance from terminus (km)')
        
        if i%2==1:
            ax.set_yticklabels([])
        
        ax.text(0., 1.025, 'b{:d}'.format(i+1), transform=ax.transAxes,
            fontweight='bold', ha='left', va='bottom')
        ax.text(0.5, 1.025, 'PC{:d} ({:.1f}%)'.format(i+1, 100*pcvar[i]), transform=ax.transAxes,
            ha='center', va='bottom')
    
    for ax in axs.flat[nplot:]:
        ax.set_visible(False)
    
    cbar = fig.colorbar(pcol, cax=cax, orientation='horizontal')
    cbar.set_label('PC coefficients')

    fig.savefig('figures/pca_rmse_var.png', dpi=600)
    fig.savefig('figures/pca_rmse_var.pdf', dpi=600)
    

if __name__=='__main__':
    plot_PC_RMSE_variance(recompute=False)

