"""
Evaluate GP for GlaDS ensembles
"""

import os
import sys
import argparse
import time
import pickle

sys.path.append(os.path.expanduser('~/SFU-code'))
from palettes.code import palettes, tools

from src.utils import import_config
from src.model import load_model

import numpy as np
from scipy import linalg
import matplotlib
matplotlib.rc('font', size=12)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction

def plot_error_samples(config, sim_y, ypred_mean, cv_error, ypred_lq, ypred_uq):
    figs = []

    with open(os.path.join(config.sim_dir,config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements']-1
    mtri = Triangulation(nodexy[:, 0]/1e3, nodexy[:, 1]/1e3, connect)

    surf = np.load('../issm/data/geom/IS_surface.npy')
    bed = np.load('../issm/data/geom/IS_bed.npy')
    zmax = 1850
    xmin = np.min(mesh['x'][surf<=zmax])/1e3
    xmax = np.max(mesh['x'][surf<=zmax])/1e3
    ymin = np.min(mesh['y'][surf<=zmax])/1e3
    ymax = np.max(mesh['y'][surf<=zmax])/1e3

    # Pick ensemble members, nodes, and time steps
    m_test = ypred_mean.shape[0]
    nx = len(mesh['x'])
    nt = int(ypred_mean.shape[1]/nx)
    dim_separated_cv_error = np.zeros((m_test, nx, nt), dtype=np.float32)
    for i in range(m_test):
        dim_separated_cv_error[i, :, :] = cv_error[i, :].reshape((nx, nt))
    rmse_m = np.sqrt(np.nanmean(dim_separated_cv_error**2, axis=(1,2)))
    rmse_x = np.sqrt(np.nanmean(dim_separated_cv_error**2, axis=(0,2)))
    rmse_t = np.sqrt(np.nanmean(dim_separated_cv_error**2, axis=(0, 1)))

    nodes = [4061, 2959, 2673]
    timestep = 205

    # Pick low (5%), median, and high (95%) ensemble members
    qntls = [0.95, 0.5, 0.05]
    sim_indices = np.zeros(len(qntls), dtype=int)
    for i,qi in enumerate(qntls):
        mi = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, qi)))
        sim_indices[i] = mi
    print('sim_indices:', sim_indices)

    # Pick logical time steps (winter, spring, summer)
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
    colors = cmocean.cm.algae([0.2, 0.5, 0.75])

    # Timeseries error
    cm2 = LinearSegmentedColormap.from_list('', cmocean.cm.gray(np.linspace(0.05, 1, 128)))
    cmap = tools.join_cmaps(cmocean.cm.dense, cm2, average=0, N1=128, N2=128)
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(len(sim_indices), len(nodes), wspace=0.1, hspace=0.1, left=0.08, right=0.98,
        bottom=0.1, top=0.985)
    lws = [1.5, 1]
    axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(len(nodes))]
                        for i in range(len(sim_indices))])
    months = np.array([4, 5, 6, 7, 8, 9, 10])
    ticklabels = ['May','','July','','Sept','','Nov']
    ticks = 365/12*months
    tt = np.arange(365)
    for j,node in enumerate(nodes):
        for i,mi in enumerate(sim_indices):
            yi_sim = sim_y[mi].reshape((nx, nt))[node, :]
            yi_pred = ypred_mean[mi].reshape((nx, nt))[node, :]
            yi_lq = ypred_lq[mi].reshape((nx, nt))[node, :]
            yi_uq = ypred_uq[mi].reshape((nx, nt))[node, :]
            ax = axs[i,j]
            ax.fill_between(tt, yi_lq, yi_uq,
                color=colors[j], alpha=0.5, edgecolor=colors[j])
            ax.plot(tt, yi_sim, color='#222222', label='GlaDS', linewidth=1.5)
            ax.plot(tt, yi_pred, color=colors[j], label='GP', linewidth=1)
            ax.grid(linestyle='dotted', linewidth=0.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, rotation=45)
            ax.set_xlim([4*365/12, 10*365/12])
            ax.axvline(timestep, linestyle='dashed', color='k', linewidth=0.5)
            
            ax.text(0.025, 0.8, alphabet[j] + str(i+1), transform=ax.transAxes,
                fontweight='bold', ha='left', va='bottom')
            
            ax.spines[['right', 'top']].set_visible(False)
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    for i in range(len(sim_indices)):
        axs[i,-1].text(0.975, 0.8, '$m_{{{}}}$={}'.format(qntls[i], sim_indices[i]), transform=axs[i,-1].transAxes,
            va='bottom', ha='right')
        ylims = np.array([ax.get_ylim() for ax in axs[i, :]])
        ylim_max = np.max(ylims)
        stepsize = 0.5 if ylim_max<2.1 else 1
        yt = np.arange(0, ylim_max, stepsize)
        for ax in axs[i, :]:
            ax.set_yticks(yt)
            ax.set_ylim([0, ylim_max])
        

    for ax in axs[:, 1:].flat:
        ax.set_yticklabels([])

    
    fig.text(0.01, 0.5, 'Flotation fraction', va='center', rotation=90)
    # axs[-1, -1].legend(loc='lower right', ncols=2)
    
    figs.append(fig)

    fig = plt.figure(figsize=(8, 4.2))
    gs = GridSpec(len(sim_indices)+1, 3, height_ratios=[8] + len(sim_indices)*[100],
        wspace=0.15, hspace=0.08, left=0.08, bottom=0.05, right=0.98, top=0.85)
    axs = np.array([[fig.add_subplot(gs[j+1,i]) for j in range(len(sim_indices))] for i in range(3)]).T
    caxs = np.array([fig.add_subplot(gs[0,i]) for i in range(3)])
    for i in range(len(sim_indices)):
        mi = sim_indices[i]
        (ax1,ax2,ax3) = axs[i, :]
        y_sim_spatial = sim_y[mi].reshape((nx, nt))[:, timestep]
        y_pred_spatial = ypred_mean[mi].reshape((nx, nt))[:, timestep]
        pc1 = ax1.tripcolor(mtri, y_sim_spatial, 
            vmin=0, vmax=2., cmap=cmap, rasterized=True)
        pc2 = ax2.tripcolor(mtri, y_pred_spatial, 
            vmin=0, vmax=2., cmap=cmap, rasterized=True)
        pc3 = ax3.tripcolor(mtri, y_pred_spatial - y_sim_spatial,
            vmin=-0.25, vmax=0.25, cmap=cmocean.cm.balance, rasterized=True)

        for ax in axs[i,:].flat:
            ax.set_aspect('equal')
            ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_xticks([])
            ax.set_yticks([])

            for k in range(len(nodes)):
                ax.plot(mesh['x'][nodes[k]]/1e3, mesh['y'][nodes[k]]/1e3, markersize=5,
                    marker='s', color=colors[k], markeredgecolor='w')
            


        cb1 = fig.colorbar(pc1, cax=caxs[0], orientation='horizontal')
        cb2 = fig.colorbar(pc2, cax=caxs[1], orientation='horizontal')
        cb3 = fig.colorbar(pc3, cax=caxs[2], orientation='horizontal')

        # lbl = '$m_{{{}}}$ = {}'.format(qntls[i], sim_indices[i])
        # ax1.text(0., 0.5, lbl, transform=ax1.transAxes,
        #     ha='right', va='center')
    
    for cax in caxs:
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.tick_params(axis='both', labelsize=9)

    axs[0,0].text(0.5, 1.45, r'GlaDS $f_{\rm{w}}$', rotation=0, ha='center',
        va='bottom', transform=axs[0,0].transAxes)
    axs[0,1].text(0.5, 1.45, r'GP emulator  $f_{\rm{w}}$', rotation=0, ha='center',
        va='bottom', transform=axs[0,1].transAxes)
    axs[0,2].text(0.5, 1.45, 'GP emulator $-$ GlaDS', rotation=0, ha='center',
        va='bottom', transform=axs[0,2].transAxes)
    
    for i in range(len(sim_indices)):
        axs[i,0].text(0., 0.3, '$m_{{{}}}$={}'.format(qntls[i], sim_indices[i]), transform=axs[i,0].transAxes,
            va='bottom', ha='center')
        for j in range(3):
            axs[i,j].text(0., 0.85, alphabet[j]+str((i+1)),
            fontweight='bold', transform=axs[i,j].transAxes, va='bottom')

    # scale = Rectangle(xy=(xmin, ymin), width=50, height=4, zorder=15)
    # spc = PatchCollection([scale], facecolor='k', clip_on=False)
    # axs[-1,-1].add_collection(spc)
    # axs[-1,-1].text(xmin+0.5*50, ymin+5, '50 km', ha='center', va='bottom')

    figs.append(fig)
    # fig.savefig(os.path.join(config.figures, 'IS_flot_frac_maps.png'), dpi=400)


    return figs


def plot_scatter(config, y_sim, ypred_mean):
    with open(os.path.join(config.sim_dir, config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    # Make a scatter plot
    fig = plt.figure(figsize=(8, 2.5))
    wspace = 10
    gs = GridSpec(1, 6, width_ratios=(100, wspace, 100, wspace, 100, 8),
        left=0.075, right=0.9, bottom=0.225, top=0.95, wspace=0.1)
    axs = np.array([fig.add_subplot(gs[2*i]) for i in range(3)])
    cax = fig.add_subplot(gs[-1])
    rng = np.random.default_rng()
    rng_inds = rng.choice(np.arange(int(np.prod(y_sim.shape)/2)), size=int(1e6), replace=False)
    y_sim_scatter = y_sim.flat[rng_inds]
    y_pred_scatter = ypred_mean.flat[rng_inds]

    # surf = 390 + 6*( (np.sqrt(nodexy[:, 0] + 5e3) - np.sqrt(5e3)))
    # bed = 350
    # thick = surf - bed
    surf = np.load('../issm/data/geom/IS_surface.npy')
    bed = np.load('../issm/data/geom/IS_bed.npy')
    thick = surf - bed
    thick[thick<10] = 10

    rho_w = 1000.
    rho_i = 910.
    g = 9.8

    p_i_spatial = g*rho_i*thick
    p_i = np.tile(np.vstack(p_i_spatial), (1, 365))
    p_i = np.tile(p_i, (config.m, 1, 1))
    p_i = p_i.reshape(y_sim.shape)

    bed = np.tile(np.vstack(bed), (1, 365))
    bed = np.tile(bed, (config.m, 1, 1))
    bed = bed.reshape(y_sim.shape)
    bed_scatter = bed.flat[rng_inds]

    print('y_sim.shape:', y_sim.shape)
    print('ypred_mean.shape:', ypred_mean.shape)
    print('p_i.shape:', p_i.shape)

    p_i_scatter = p_i.flat[rng_inds]
    N_sim_scatter = p_i_scatter*(1 - y_sim_scatter)
    N_pred_scatter = p_i_scatter*(1 - y_pred_scatter)

    phi_sim_scatter = rho_w*g*bed_scatter + p_i_scatter*y_sim_scatter
    phi_pred_scatter = rho_w*g*bed_scatter + p_i_scatter*y_pred_scatter

    # Define bounds
    ff_min = -0.1
    ff_max = 1.6
    ff_ticks = [0, 0.5, 1, 1.5]

    N_min = -0.5
    N_max = 5
    N_ticks = [0, 2, 4] 

    phi_min = 1
    phi_max = 2.5e7/1e6
    phi_ticks = [5, 10, 15, 20, 25]

    countnorm = colors.LogNorm(vmin=1e0, vmax=1e4, clip=True)

    axs[0].hexbin(y_sim_scatter, y_pred_scatter, norm=countnorm,
        cmap=cmocean.cm.rain, gridsize=100, edgecolors='none',
        extent=(ff_min, ff_max, ff_min, ff_max))
    axs[0].set_xlim([ff_min, ff_max])
    axs[0].set_ylim([ff_min, ff_max])
    R2 = np.corrcoef(y_sim_scatter.flatten(), y_pred_scatter.flatten())[0,1]**2
    axs[0].text(0.95, 0.025, '$r^2={:.3f}$'.format(R2),
        ha='right', va='bottom', transform=axs[0].transAxes)
    axs[0].set_aspect('equal')
    axs[0].set_xticks(ff_ticks)
    axs[0].set_yticks(ff_ticks)
    axs[0].set_xlabel('Flot frac')
    axs[0].text(0.025, 0.95, 'a', transform=axs[0].transAxes,
        fontweight='bold', ha='left', va='top')

    axs[1].hexbin(N_sim_scatter/1e6, N_pred_scatter/1e6, norm=countnorm,
        cmap=cmocean.cm.rain, gridsize=100, edgecolors='none',
        extent=(N_min, N_max, N_min, N_max))
    axs[1].set_xlim([N_min, N_max])
    axs[1].set_ylim([N_min, N_max])
    R2 = np.corrcoef(N_sim_scatter, N_pred_scatter)[0,1]**2
    axs[1].text(0.95, 0.025, '$r^2={:.3f}$'.format(R2),
        ha='right', va='bottom', transform=axs[1].transAxes)
    axs[1].set_aspect('equal')
    axs[1].set_xticks(N_ticks)
    axs[1].set_yticks(N_ticks)
    axs[1].set_xlabel('$N$ (MPa)')
    axs[1].text(0.025, 0.95, 'b', transform=axs[1].transAxes,
        fontweight='bold', ha='left', va='top')

    hb = axs[2].hexbin(phi_sim_scatter/1e6, phi_pred_scatter/1e6, norm=countnorm,
        cmap=cmocean.cm.rain, gridsize=100, edgecolors='none',
        extent=(phi_min, phi_max, phi_min, phi_max))
    axs[2].set_xlim([phi_min, phi_max])
    axs[2].set_ylim([phi_min, phi_max])
    R2 = np.corrcoef(phi_sim_scatter, phi_pred_scatter)[0,1]**2
    axs[2].text(0.95, 0.025, '$r^2={:.3f}$'.format(R2),
        ha='right', va='bottom', transform=axs[2].transAxes)
    axs[2].set_aspect('equal')
    axs[2].set_xticks(phi_ticks)
    axs[2].set_yticks(phi_ticks)
    axs[2].set_xlabel(r'$\phi$ (MPa)')
    axs[2].text(0.025, 0.95, 'c', transform=axs[2].transAxes,
        fontweight='bold', ha='left', va='top')

    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label('Count (n={})'.format(len(rng_inds)))

    for ax in axs:
        ax.grid(linestyle=':', linewidth=0.5)
    return fig

def main(config, test_config, recompute=False, dtype=np.float32):
    """
    Fit GP, compute and save CV prediction error, make basic figures

    Parameters
    ----------
    config : module
             Configuration file loaded as a module
    
    recompute : bool, optional
                Force recompute fields even if file already exists
    
    dtype : optional (np.float32)
            Data type for GP predictions and CV error calculations            
    """
    # Load data and initialize model
    t_std = np.loadtxt(config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(config.X_physical, delimiter=',', skiprows=1).astype(dtype)
    t_std = t_std[:config.m, :]
    t_phys = t_phys[:config.m, :]
    y_train_sim = np.load(config.Y_physical).T[:config.m, :].astype(dtype)

    t_test_std = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)[:test_config.m :]
    t_test_phys = np.loadtxt(test_config.X_physical, delimiter=',',
        skiprows=1).astype(dtype)[:test_config.m, :]
    y_test_sim = np.load(test_config.Y_physical).T[:test_config.m, :].astype(dtype)

    ypred_mean = np.load(cv_y_file, mmap_mode='r')[:test_config.m, :]
    ypred_lq = np.load(cv_lq_file, mmap_mode='r')[:test_config.m :]
    ypred_uq = np.load(cv_uq_file, mmap_mode='r')[:test_config.m, :]

    print('ypred_mean.shape:', ypred_mean.shape)
    rmse_ts, rmse_map = plot_error_samples(config, 
        sim_y=y_test_sim, ypred_mean=ypred_mean, cv_error=ypred_mean-y_test_sim, ypred_lq=ypred_lq, ypred_uq=ypred_uq)
    rmse_ts.savefig(os.path.join(
        config.figures, 'test_error_timeseries.png'), dpi=400)
    rmse_ts.savefig(os.path.join(
        config.figures, 'test_error_timeseries.pdf'), dpi=400)
    
    rmse_map.savefig(os.path.join(
        config.figures, 'test_error_map.png'), dpi=400)
    rmse_map.savefig(os.path.join(
        config.figures, 'test_error_map.pdf'), dpi=400)


    scatter_fig = plot_scatter(test_config, y_test_sim, ypred_mean)
    scatter_fig.savefig(os.path.join(
        config.figures, 'test_error_scatter.png'), dpi=400)

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file')
    parser.add_argument('test_file')
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    config = import_config(args.conf_file)
    test_config = import_config(args.test_file)
    main(config, test_config, recompute=args.recompute)
