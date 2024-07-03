"""
Fit and evaluate GP for GlaDS ensembles

TODO
 [ ] tabulate RMSE, MAPE, prediction uncert values (see sepia_scalar_gp_cv.py)
"""

import os
import sys
import argparse
import time
import pickle

import netCDF4 as nc

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.tri import Triangulation
from matplotlib import colors
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction
from sepia.SepiaLogLik import compute_log_lik
from sepia.SepiaDistCov import SepiaDistCov

from src.utils import import_config
from src.model import init_model, load_model

# def init_model(t_std, y_train_sim, exp, p, nx=nx, nt=nt,
#     data_dir='data/', scale='scalar', sd_threshold=1e-6,
#     recompute=False, plot=False):
#     """
#     Initialize SepiaData and SepiaModel instances

#     Parameters
#     ----------
#     t_std : (n simulations, t_dim)
#             Standardized simulation design matrix

#     y_train_sim : (n_simulations, y_dim)
#             Non-standardized simulation output matrix
    
#     config.exp : str
#                Name for simulation, used to generate file paths
    
#     p : int
#         Number of principal components to retain
    
#     Returns:
#     --------
#     SepiaData, SepiaModel
#     """
#     print('y_train_sim:', y_train_sim.dtype)
#     y_ind_sim = np.linspace(0, 1, y_train_sim.shape[1])
#     data = SepiaData(t_sim=t_std, y_sim=y_train_sim, y_ind_sim=y_ind_sim)

#     t_dim = t_std.shape[1]

#     # Compute a custom basis for comparison
#     if scale=='scalar':
#         mu_y = np.mean(y_train_sim)
#         sd_y = np.std(y_train_sim, ddof=1)
#     elif scale=='columnwise':
#         mu_y = np.mean(y_train_sim, axis=0)
#         sd_y = np.std(y_train_sim, ddof=1, axis=0)
#         sd_y[sd_y<sd_threshold] = sd_threshold

#     # No transformations for x, t. Already scaled in [0, 1]
#     data.transform_xt(t_notrans=np.arange(t_dim))

#     # Transform y with given mean, sd
#     data.standardize_y(y_mean=mu_y, y_sd=sd_y)

#     # Check scaling to be careful
#     y_std = (y_train_sim - mu_y)/sd_y
#     assert np.allclose(y_std, data.sim_data.y_std)

#     # Compute PCA basis
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     pca_fpattern = os.path.join(data_dir, 'pca_{}_{}.npy')
#     pca_fstat = np.array([os.path.exists(pca_fpattern.format(exp, arr)) for arr in ['U', 'S', 'Vh']])
#     pmax = 100
#     if recompute or not pca_fstat.all():
#         U,S,Vh = linalg.svd(y_std, full_matrices=False)

#         U = U[:, :pmax]
#         Vh = Vh[:pmax, :]
#         np.save(pca_fpattern.format(exp, 'U'), U)
#         np.save(pca_fpattern.format(exp, 'S'), S)
#         np.save(pca_fpattern.format(exp, 'Vh'), Vh)

#     U = np.load(pca_fpattern.format(exp, 'U'))
#     S = np.load(pca_fpattern.format(exp, 'S'))
#     Vh = np.load(pca_fpattern.format(exp, 'Vh'))
#     S2 = S**2
#     prop_var = S2/np.sum(S2)
#     cumul_var = np.cumsum(S2)
#     print('SVD proportion of variance:', prop_var[:10])
#     K = np.diag(S[:p]) @ Vh[:p] / np.sqrt(y_train_sim.shape[0])
#     print('K.shape', K.shape)
#     print('K.dtype', K.dtype)

#     data.create_K_basis(K=K)
#     model = SepiaModel(data)
#     out = (data, model)

#     if plot:
#         fig = plt.figure(figsize=(8, 8))
#         ncols = 2
#         nrows = int(np.ceil(p/ncols))
#         gs = GridSpec(nrows, ncols, left=0.1, right=0.95,
#             bottom=0.1, top=0.95, hspace=0.3)
#         cgs = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[-1,-1])
#         axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(ncols)] for i in range(nrows)])
#         cax = fig.add_subplot(cgs[2])

#         with nc.Dataset('../../issm/data/mesh/mesh_04.nc', 'r') as dmesh:
#             nodes = dmesh['tri/nodes'][:].data.T
        
        
#         def wavg(x, Z):
#             dx = 1e3
#             xleft = np.arange(0, 100e3, dx)
#             xright = np.arange(dx, 100e3+dx, dx)
#             Zavg = np.zeros((nt, len(xleft)))
#             for xi in range(len(xleft)):
#                 is_band = np.logical_and(x>=xleft[xi], x<=xleft[xi]+dx)
#                 Zavg[:, xi] = np.mean(Z[is_band, :], axis=0)
#             return xleft, Zavg

#         for i in range(p):
#             ax = axs.flat[i]
#             xl,kavg = wavg(nodes[:, 0], np.reshape(K[i, :], (nx, nt)))
#             # kavg = (kavg - np.mean(kavg))/np.std(kavg)

#             [xx, tt] = np.meshgrid(xl, np.arange(nt))
#             pc = ax.pcolormesh(xx, tt, kavg, vmin=-1, vmax=1,
#                 cmap=cmocean.cm.curl)
#             ax.set_title('PC{:d} ({:.1f}%)'.format(i+1, 100*prop_var[i]))
#             ax.set_xticks(np.linspace(0, 100e3, 6))
#             ax.set_xticklabels(np.linspace(0, 100, 6).astype(int))
#             ax.set_yticks(np.linspace(0, 365, 5))
#             ax.set_yticklabels(np.linspace(0, 12, 5).astype(int))
        
#         for ax in axs[:, 1:].flat:
#             ax.set_yticklabels([])
#         for ax in axs.flat[:-3]:
#             ax.set_xticklabels([])
        
#         for ax in axs.flat[p:]:
#             ax.set_visible(False)
        
#         fig.text(0.025, 0.5, 'Month', rotation=90, va='center', ha='center')
#         cb = fig.colorbar(pc, cax=cax, orientation='horizontal')
#         cb.set_label('PC coefficients')

        
#         axs[-1, 0].set_xlabel('Distance from terminus (km)')
#         axs[-2, 1].set_xlabel('Distance from terminus (km)')
#         out = (data, model, fig)

#     return out

def plot_pca_rmse(y_train_sim, npcs, data_dir, exp, scale='columnwise'):
    """
    Plot PCA truncation error RMSE and cumulative proportion of variance

    Parameters
    ----------
    y_train_sim : (m, nx * nt) array
            Simulation data in original units
    
    npcs : array of int
           Principal components to compute RMSE for
    
    data_dir : str
               Path to directory to save PC data
    
    exp : str
          Name of experiment used to concat file names
    
    scale : ['columnwise' or 'scalar']
            To compute mean and SD per column or as a scalar
            Strongly recommend columnwise scaling
    
    Returns
    -------
    fig : figure handle
    """
    if scale=='scalar':
        mu_y = np.mean(y_train_sim)
        sd_y = np.std(y_train_sim, ddof=1)
    elif scale=='columnwise':
        mu_y = np.mean(y_train_sim, axis=0)
        sd_y = np.std(y_train_sim, ddof=1, axis=0)
        sd_y[sd_y<1e-6] = 1e-6
    
    pca_fpattern = os.path.join(data_dir, 'pca_{}_{}.npy')
    U = np.load(pca_fpattern.format(exp, 'U'))
    S = np.load(pca_fpattern.format(exp, 'S'))
    Vh = np.load(pca_fpattern.format(exp, 'Vh'))
    S2 = S**2
    prop_var = S2/np.sum(S2)
    cumul_var = np.cumsum(prop_var)

    pca_rmse = np.zeros(npcs.shape)
    pca_mape = np.zeros(npcs.shape)
    pca_cvar = cumul_var[npcs.astype(int)-1]
    for i in range(len(npcs)):
        print('Using %d PCs' % npcs[i])
        pi = npcs[i]
        Y_trunc = mu_y + sd_y*(U[:, :pi] @ np.diag(S[:pi]) @ Vh[:pi])
        print('Y_trunc.dtype', Y_trunc.dtype)
        trunc_err = Y_trunc - y_train_sim
        pca_rmse[i] = np.sqrt(np.mean(trunc_err**2))
        perc_err = np.abs(trunc_err/y_train_sim)
        perc_err[y_train_sim<1e-2] = np.nan
        pca_mape[i] = np.nanmean(perc_err)
    
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.plot(npcs, pca_rmse)
    ax1.grid(linestyle=':', linewidth=0.5)

    ax2.plot(npcs, pca_cvar)
    ax2.grid(linestyle=':', linewidth=0.5)

    ax1.set_xlabel('Number of PCs')
    ax2.set_xlabel('Number of PCs')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('Cumulative proportion of variance')

    ax1.set_ylim([0, 0.15])
    fig.subplots_adjust(left=0.1, bottom=0.15, top=0.95, right=0.95, wspace=0.3)
    return fig

def compute_test_predictions(model, samples, t_pred, n_folds=100, quantile=0.025):
    """
    Cross-validation error

    Compute mean predictions and prediction intervals on
    test samples. This is intentionally written as a
    CPU-inefficient 'for' loop to minimize memory usage.

    Parameters
    ----------
    model : SepiaModel

    samples : dict
                   Posterior samples to use in predictions
    
    t_pred : array
             Test settings for predictions
    
    quantile : float [0, 1]
               Prediction intervals are computed for the
               interval [quantile, 1-quantile]
    
    Returns
    -------
    mean, lower_quantile, upper_quantile : array
        (number of predictions, nx*nt) arrays
    """
    m = model.data.sim_data.y.shape[0]
    N = model.data.sim_data.y.shape[1]
    m_pred = t_pred.shape[0]
    pred_mean = np.zeros((m_pred, N), dtype=model.data.sim_data.y.dtype)
    pred_lower = np.zeros((m_pred, N), dtype=model.data.sim_data.y.dtype)
    pred_upper = np.zeros((m_pred, N), dtype=model.data.sim_data.y.dtype)
    for i in range(m_pred):
        xi = t_pred[i:i+1]
        print('Sample {}/{}:'.format(i+1, m_pred))
        pred = SepiaEmulatorPrediction(samples=samples,
            model=model, t_pred=xi)
        pred.w = pred.w.astype(model.data.sim_data.y.dtype)
        preds = pred.get_y()
        pred_mean[i, :] = np.mean(preds, axis=0)
        pred_lower[i, :] = np.quantile(preds, quantile, axis=0)
        pred_upper[i, :] = np.quantile(preds, 1-quantile, axis=0)
    return pred_mean, pred_lower, pred_upper

def plot_rmse(config, y_test_pred, test_error, test_lq, test_uq):
    figs = []

    with open(os.path.join(config.sim_dir, config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements'].astype(int)-1

    mtri = Triangulation(nodexy[:, 0]/1e3, nodexy[:, 1]/1e3, connect)

    # Pick ensemble members, nodes, and time steps
    m = config.m
    m_test = y_test_pred.shape[0]
    nx = nodexy.shape[0]
    nt = int(y_test_pred.shape[1]/nx)
    dim_separated_test_error = np.zeros((m_test, nx, nt), dtype=np.float32)
    for i in range(m_test):
        # Ysim = y_test_pred[i, :] + test_error[i, :]
        dim_separated_test_error[i, :, :] = test_error[i, :].reshape((nx, nt))
    # dim_separated_test_error[~test_mask, :] = np.nan
    rmse_m = np.sqrt(np.nanmean(dim_separated_test_error**2, axis=(1,2)))
    rmse_x = np.sqrt(np.nanmean(dim_separated_test_error**2, axis=(0,2)))
    rmse_t = np.sqrt(np.nanmean(dim_separated_test_error**2, axis=(0, 1)))

    nodes = [-1, -1, -1]
    xpos = [15e3, 30e3, 50e3]
    ypos = [12.5e3, 12.5e3, 12.5e3]
    nodes[0] = np.argmin( (nodexy[:, 0]-xpos[0])**2 + (nodexy[:, 1]-ypos[0])**2)
    nodes[1] = np.argmin( (nodexy[:, 0]-xpos[1])**2 + (nodexy[:, 1]-ypos[1])**2)
    nodes[2] = np.argmin( (nodexy[:, 0]-xpos[2])**2 + (nodexy[:, 1]-ypos[2])**2)

    # Pick low (5%), median, and high (95%) ensemble members
    m_low = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, 0.05)))
    m_med = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, 0.5)))
    m_high = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, 0.95)))
    ms = [m_high, m_med, m_low]

    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(2, 2, height_ratios=(150, 100), width_ratios=(100, 5),
        left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.05, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    tpc = ax2.tripcolor(mtri, rmse_x, vmin=0, vmax=0.15, cmap=cmocean.cm.matter)
    ax2.set_aspect('equal')
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0, 25])
    ax2.set_yticks([0, 12.5, 25])
    ax2.set_ylabel('Distance across (km)')
    ax2.set_xlabel('Distance from terminus (km)')
    cb = fig.colorbar(tpc, cax=cax)
    cb.set_label('RMSE')

    t_month = np.arange(365) * 12/365
    ax1.plot(t_month,rmse_t)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('RMSE')
    ax1.set_ylim([0, 0.20])
    ax1.set_xlim([0, 12])
    ax1.set_xticks(np.arange(0, 12, 5).astype(int))
    ax1.grid(linestyle=':', linewidth=0.5)

    ax1.text(0.025, 0.95, 'a', transform=ax1.transAxes,
        fontweight='bold', ha='left', va='top')
    ax2.text(0.025, 0.95, 'b', transform=ax2.transAxes,
        fontweight='bold', ha='left', va='top')
    figs.append(fig)


    # Pick logical time steps (winter, spring, summer)
    t_steps = [120, 160, 220]

    # Width-averaged
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(len(ms)+1, 4, wspace=0.2, hspace=0.2,
        left=0.06, bottom=0.08, right=0.98, top=0.925,
        height_ratios=[8] + len(ms)*[100])
    axs = np.array([[fig.add_subplot(gs[i+1,j]) for j in range(4)]
                        for i in range(len(ms))])
    caxs = [fig.add_subplot(gs[0, i]) for i in range(4)]

    def width_average(x, dx=2):
        """Width-average (nx, nt) array x"""
        xedge = np.arange(0, 100+dx, dx)
        xmid = 0.5*(xedge[1:] + xedge[:-1])
        xavg = np.zeros((len(xmid), x.shape[1]))
        for i in range(len(xavg)):
            xi = xmid[i]
            mask = np.abs(nodexy[:,0]/1e3 - xi)<dx/2
            xavg[i] = np.nanmean(x[mask,:],axis=0)
        return xavg
    
    for i in range(len(ms)):
        mi = ms[i]
        y_sim_mi = y_test_pred[mi] + test_error[mi]
        avg_ysim = width_average(y_sim_mi.reshape((nx, nt)))
        avg_ypred = width_average(y_test_pred[mi].reshape((nx, nt)))
        avg_err = width_average(test_error[mi].reshape((nx, nt)))
        avg_sd = width_average((test_uq[mi] - test_lq[mi]).reshape((nx, nt)))
        dx = 2
        xedge = np.arange(0, 100+dx, dx)
        t = np.arange(0, 366)
        [tt,xx] = np.meshgrid(t,xedge)

        ax0 = axs[i, 0]
        ypc = ax0.pcolormesh(xx, tt, avg_ysim, cmap=cmocean.cm.delta,
            vmin=0, vmax=2, shading='flat')

        ax00 = axs[i, 1]
        ax00.pcolormesh(xx, tt, avg_ypred, cmap=cmocean.cm.delta,
            vmin=0, vmax=2, shading='flat')

        ax1 = axs[i, 2]
        epc = ax1.pcolormesh(xx, tt, avg_err, cmap=cmocean.cm.balance, 
            vmin=-0.5, vmax=0.5, shading='flat')

        ax2 = axs[i, 3]
        spc = ax2.pcolormesh(xx, tt, avg_sd, cmap=cmocean.cm.amp, 
            vmin=0, vmax=0.5, shading='flat')
        
        for ax in axs[i, :]:
            ax.text(0.95, 0.95, 'm=%d'%mi, transform=ax.transAxes,
                ha='right', va='top')
        
        
    for ax in axs.flat:
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        months = np.array([4, 6, 8, 10])
        ticks = 365/12*months
        ax.set_yticks(ticks)
        ax.set_yticklabels(months + 1)
        ax.set_ylim([4*365/12, 10*365/12])
        for i in range(len(nodes)):
            ax.axvline(nodexy[nodes[i], 0]/1e3, linestyle=':', color='k', linewidth=0.75)
    
    for ax in axs[:, 1:].flat:
        ax.set_yticklabels([])  
    
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    fig.text(0.0, 0.5, 'Month', rotation=90, va='center')
    fig.text(0.5, 0.02, 'Distance from terminus (km)', ha='center')
    
    cbar_ysim = fig.colorbar(ypc, cax=caxs[0], orientation='horizontal')
    cbar_ysim.set_label('Sim flotation fraction')

    cbar_ygp = fig.colorbar(ypc, cax=caxs[1], orientation='horizontal')
    cbar_ygp.set_label('GP flotation fraction')

    cbar_error = fig.colorbar(epc, cax=caxs[2], orientation='horizontal')
    cbar_error.set_label(r'$\Delta$ flotation fraction')

    cbar_sd = fig.colorbar(spc, cax=caxs[3], orientation='horizontal')
    cbar_sd.set_label('95% prediction interval')

    for cax in caxs:
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
    
    figs.append(fig)

    # Timeseries error
    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(3, 3, wspace=0.15, hspace=0.1, left=0.08, right=0.96,
        bottom=0.1, top=0.95)
    lws = [1.5, 1]
    axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(3)]
                        for i in range(3)])
    tt = np.arange(365)
    for j,node in enumerate(nodes[:3]):
        for i,mi in enumerate(ms):
            yi_sim = y_test_pred[mi].reshape((nx, nt))[node, :] + test_error[mi].reshape((nx, nt))[node, :]
            yi_pred = y_test_pred[mi].reshape((nx, nt))[node, :]
            yi_lq = test_lq[mi].reshape((nx, nt))[node, :]
            yi_uq = test_uq[mi].reshape((nx, nt))[node, :]
            ax = axs[i,j]
            ax.fill_between(tt, yi_lq, yi_uq,
                color='#B58898', alpha=0.67, edgecolor='none')
            ax.plot(tt, yi_sim, color='#222222', label='Sim', linewidth=1.5)
            ax.plot(tt, yi_pred, color='#823853', label='GP', linewidth=1)
            ax.grid(linestyle='dotted', linewidth=0.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels(months + 1)
            ax.set_xlim([4*365/12, 10*365/12])
            ax.text(0.95, 0.95, 'm=%d'%mi, transform=ax.transAxes,
                va='top', ha='right')
            for k in range(len(t_steps)):
                ax.axvline(t_steps[k], linestyle=':', color='k', linewidth=0.75)

    for i,ax in enumerate(axs[0,:]):
        ax.text(0.5, 1.05, 'node={:d}, x={:.1f} km'.format(nodes[i], (xpos[i]/1e3)), transform=ax.transAxes,
            ha='center')
    
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    axs[-1, 1].set_xlabel('Month')
    
    for ax in axs[:, 1:].flat:
        ax.set_yticklabels([])
    
    for i in range(3):
        ylims = np.array([ax.get_ylim() for ax in axs[i, :]])
        ylim_max = np.max(ylims)
        for ax in axs[i, :]:
            ax.set_ylim([0, ylim_max])
    
    axs[1, 0].set_ylabel('Flotation fraction')
    leg_width = 1.
    leg_height = 0.4
    axs[-1, 0].legend(bbox_to_anchor=(0.5-leg_width/2, -leg_height , leg_width, leg_height), 
        loc='lower center', frameon=False, ncols=2, mode='expand')
    
    figs.append(fig)

    return figs

def plot_scatter(config, y_test_sim, y_test_pred):
    # with nc.Dataset(config.mesh, 'r') as dmesh:
    #     nodexy = dmesh['tri/nodes'][:].data.T

    with open(os.path.join(config.sim_dir, config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements'].astype(int)-1

    # Make a scatter plot
    fig = plt.figure(figsize=(8, 2.5))
    wspace = 10
    gs = GridSpec(1, 6, width_ratios=(100, wspace, 100, wspace, 100, 8),
        left=0.075, right=0.9, bottom=0.225, top=0.95, wspace=0.1)
    axs = np.array([fig.add_subplot(gs[2*i]) for i in range(3)])
    cax = fig.add_subplot(gs[-1])
    rng = np.random.default_rng()
    rng_inds = rng.choice(np.arange(np.prod(y_test_sim.shape)), size=int(1e6), replace=False)
    y_sim_scatter = y_test_sim.flat[rng_inds]
    y_pred_scatter = y_test_pred.flat[rng_inds]

    # surf = 390 + 6*( (np.sqrt(nodexy[:, 0] + 5e3) - np.sqrt(5e3)))
    # bed = 350
    surf = np.load('../issm/data/geom/synthetic_surface.npy')
    bed = np.load('../issm/data/geom/synthetic_bed.npy')
    thick = surf - bed
    p_i_spatial = 9.8*910*(surf - bed)
    p_i = np.tile(np.vstack(p_i_spatial), (1, 365))
    p_i = np.tile(p_i, (config.m, 1, 1))

    p_i_scatter = p_i.flat[rng_inds]
    N_sim_scatter = p_i_scatter*(1 - y_sim_scatter)
    N_pred_scatter = p_i_scatter*(1 - y_pred_scatter)

    phi_sim_scatter = 350*1e3*9.8 + p_i_scatter*y_sim_scatter
    phi_pred_scatter = 350*1e3*9.8 + p_i_scatter*y_pred_scatter

    # Define bounds
    ff_min = -0.1
    ff_max = 1.6
    ff_ticks = [0, 0.5, 1, 1.5]

    N_min = -0.5
    N_max = 5
    N_ticks = [0, 2, 4] 

    phi_min = 350*9.8*1e3/1e6
    phi_max = 2e7/1e6
    phi_ticks = [5, 10, 15, 20]

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
    Fit GP, compute and save test prediction error, make basic figures

    Parameters
    ----------
    config : module
             Configuration file loaded as a module

    test_config : module
                  Configuration file for test set
    
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

    p = config.p
    scale = 'columnwise'
    
    cputime = {}
    t_orig = time.perf_counter()

    t0_pca = time.perf_counter()
    data, model = init_model(t_std, y_train_sim, config.exp, p, 
        data_dir=config.data_dir, recompute=False)
    t1_pca = time.perf_counter()
    cputime['PCA'] = t1_pca - t0_pca
    
    data_dir = os.path.join('.', 'data/reference')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(config.figures):
        os.makedirs(config.figures)
    
    # pca_basis_figure = os.path.join(config.figures, 'pca_basis.png')
    # if not os.path.exists(pca_basis_figure) or recompute:
    #     pca_basis_fig.savefig(pca_basis_figure, dpi=400)

    # Fit model with MCMC sampling
    # model_file = os.path.join(data_dir, config.exp)
    # print('main::model_file:', model_file)
    # if recompute or not os.path.exists(model_file + '.pkl'):
    #     # model.tune_step_sizes(100, 10, update_vals=True)
    #     t0_train = time.perf_counter()
    #     model.do_mcmc(512)
    #     model.save_model_info(file_name=model_file)
    #     samples = model.get_samples(numsamples=256, nburn=256)
    #     t1_train = time.perf_counter()
    #     cputime['train'] = t1_train - t0_train
    #     # trace, box, stats = mcmc_diagnostics(model, samples)
    #     # trace.savefig(os.path.join(config.figures, 'mcmc_diagnostics_trace.png'), dpi=600)
    #     # box.savefig(os.path.join(config.figures, 'mcmc_diagnostics_rho_boxplot.png'), dpi=600)
    
    # model.restore_model_info(model_file)
    sepia_data, model = load_model(config, config.m, config.p)
    print('main::model', model)


    # logLik1 = model.logLik()
    # beta = samples['betaU']
    # lamU = samples['lamUz']
    # lamW = samples['lamWs']

    # beta = np.median(beta, axis=0)
    # lamU = np.median(lamU, axis=0)
    # lamW = np.median(lamW, axis=0)

    # # model.set_param('betaU', val=beta.reshape((9, config.p)))
    # model.params.betaU.val = beta.reshape(model.params.betaU.val.shape)
    # model.params.lamUz.val = lamU.reshape(model.params.lamUz.val.shape)
    # model.params.lamWs.val = lamW.reshape(model.params.lamWs.val.shape)
    # logLik2 = model.logLik()

    # d = t_std.shape[1]
    # k = 2*(config.p*(d+1))
    # aic = 2*k - 2*logLik1
    # bic = k*np.log(config.m) - 2*logLik1

    # print('aic:', aic)
    # print('bic:', bic)

    # # dc = SepiaDistCov(model.data.x, cat_ind=[])
    # # cov = dc.compute_cov_mat(beta, lamU, lams=lamW)
    # # print(beta.shape)
    # # print('cov.shape:', cov.shape)

    # print(logLik1)
    # print(logLik2)
    # return

    # Compute CV predictions and error

    # Binary for full space-time resolved fields
    y_test_pred_file = os.path.join(data_dir, 'test_mean.npy')
    test_lq_file = os.path.join(data_dir, 'test_lower.npy')
    test_uq_file = os.path.join(data_dir, 'test_upper.npy')

    # .csv for smaller aggregate files
    # (RMSE, MAPE, and prediction uncertainty)
    test_rmse_file = os.path.join(data_dir, 'test_rmse.csv')
    test_mape_file = os.path.join(data_dir, 'test_mape.csv')
    test_pu_file = os.path.join(data_dir, 'test_pu.csv')
    if recompute or not os.path.exists(y_test_pred_file):
        samples = model.get_samples(numsamples=64, nburn=256)
        for key in samples.keys():
            samples[key] = samples[key].astype(dtype)
        t0_test = time.perf_counter()
        y_test_pred, test_lq, test_uq = compute_test_predictions(model,
            samples, t_test_std, quantile=0.025)
        t1_test = time.perf_counter()
        dt_test_mean = (t1_test - t0_test)/t_test_std.shape[0]
        cputime['predict'] = dt_test_mean
        np.save(y_test_pred_file, y_test_pred)
        np.save(test_lq_file, test_lq)
        np.save(test_uq_file, test_uq)

        test_error = y_test_pred - y_test_sim
        test_rmse = np.sqrt(np.nanmean(test_error**2, axis=1))
        inner_mape = test_error/y_test_sim
        y_test_lq = np.quantile(y_test_sim, 0.1)
        inner_mape[np.abs(y_test_sim<y_test_lq)] = np.nan
        test_mape = np.nanmean(np.abs(inner_mape), axis=1)

        test_pu = np.zeros(1)    # TODO compute this somewhere

        np.savetxt(test_rmse_file, test_rmse, delimiter=',', fmt='%.6e')
        np.savetxt(test_mape_file, test_mape, delimiter=',', fmt='%.6e')
        np.savetxt(test_pu_file, test_pu, delimiter=',', fmt='%.6e')
        
    else:
        y_test_pred = np.load(y_test_pred_file).astype(dtype)[:config.m, :]
        test_lq = np.load(test_lq_file).astype(dtype)[:config.m :]
        test_uq = np.load(test_uq_file).astype(dtype)[:config.m, :]

        test_rmse = np.loadtxt(test_rmse_file, delimiter=',')
        test_mape = np.loadtxt(test_mape_file, delimiter=',')
        test_pu = np.loadtxt(test_pu_file, delimiter=',')

    rmsefig, rmse_wavg, rmse_ts = plot_rmse(config, 
        y_test_pred=y_test_pred, test_error=y_test_pred-y_test_sim, test_lq=test_lq, test_uq=test_uq)
    rmsefig.savefig(os.path.join(
        config.figures, 'test_rmse_timeseries_spatial.png'), dpi=400)
    rmse_wavg.savefig(os.path.join(
        config.figures, 'test_error_width_avg.png'), dpi=400)
    rmse_ts.savefig(os.path.join(
        config.figures, 'test_error_timeseries.png'), dpi=400)

    scatter_fig = plot_scatter(config, y_test_sim, y_test_pred)
    scatter_fig.savefig(os.path.join(
        config.figures, 'test_error_scatter.png'), dpi=400)

    print('Timing (seconds):', cputime)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file')
    parser.add_argument('test_file')
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    config = import_config(args.conf_file)
    test_config = import_config(args.test_file)
    main(config, test_config, recompute=args.recompute)
