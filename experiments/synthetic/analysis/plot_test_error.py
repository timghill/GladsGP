"""
Evaluate GP for GlaDS ensembles: timeseries, width-averaged test errors.

usage: plot_test_error.py [-h] [--recompute] conf_file test_file
"""

import os
import sys
import argparse
import time
import pickle

# Specific colourmap tools
sys.path.append(os.path.expanduser('~/SFU-code//'))
from palettes.code import palettes, tools

from src.utils import import_config
from src.model import load_model

import numpy as np

import matplotlib
matplotlib.rc('font', size=8)

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.tri import Triangulation
from matplotlib import colors as mpc
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction


def compute_test_predictions(model, samples, t_pred, quantile=0.025):
    """
    Test-set prediction error

    Compute mean predictions and prediction intervals on
    test samples. This is intentionally written as a
    CPU-inefficient 'for' loop to manage memory usage.

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
    m_pred = t_pred.shape[0]
    n = model.data.sim_data.y.shape[1]
    mu_y = np.mean(model.data.sim_data.y, axis=0)
    sd_y = np.std(model.data.sim_data.y, ddof=1, axis=0)
    sd_y[sd_y<1e-6] = 1e-6

    pred_mean = np.zeros((m_pred, n), dtype=model.data.sim_data.y.dtype)
    pred_lower = np.zeros((m_pred, n), dtype=model.data.sim_data.y.dtype)
    pred_upper = np.zeros((m_pred, n), dtype=model.data.sim_data.y.dtype)
    for i in range(m_pred):
        xi = t_pred[i:i+1]
        print('Sample {}/{}:'.format(i+1, m_pred))
        pred = SepiaEmulatorPrediction(samples=samples,
            model=model, t_pred=xi)
        pred.w = pred.w.astype(model.data.sim_data.y.dtype)
        emulator_preds = pred.get_y()
        error_preds = np.zeros(emulator_preds.shape, dtype=np.float32)
        for j in range(error_preds.shape[0]):
            error_preds[j] = sd_y*np.random.normal(scale=1/np.sqrt(samples['lamWOs'][j]), size=n)

        pred_mean[i, :] = np.mean(emulator_preds, axis=0)
        pred_lower[i, :] = np.quantile(emulator_preds + error_preds, quantile, axis=0)
        pred_upper[i, :] = np.quantile(emulator_preds + error_preds, 1-quantile, axis=0)
    return pred_mean, pred_lower, pred_upper

def plot_rmse(config, sim_y, test_y, test_error, test_lq, test_uq):
    """
    Make all test error figures

    Parameters
    ----------
    config : module
             Training ensemble configuration
    
    sim_y : (m, n_x*n_t) array
            Simulation output matrix
    
    test_* : (m, n_x*n_t) array
           Test predictions, errors and quantiles
    
    Returns
    -------
    figs : list of figures
    """
    figs = []

    with open(os.path.join(config.sim_dir,config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements']-1
    mtri = Triangulation(nodexy[:, 0]/1e3, nodexy[:, 1]/1e3, connect)

    # Pick ensemble members, nodes, and time steps
    m_test = test_y.shape[0]
    nx = len(mesh['x'])
    nt = int(test_y.shape[1]/nx)
    dim_separated_cv_error = np.zeros((m_test, nx, nt), dtype=np.float32)
    for i in range(m_test):
        # Ysim = test_y[i, :] + test_error[i, :]
        dim_separated_cv_error[i, :, :] = test_error[i, :].reshape((nx, nt))
    # dim_separated_cv_error[~cv_mask, :] = np.nan
    rmse_m = np.sqrt(np.nanmean(dim_separated_cv_error**2, axis=(1,2)))
    rmse_x = np.sqrt(np.nanmean(dim_separated_cv_error**2, axis=(0,2)))
    rmse_t = np.sqrt(np.nanmean(dim_separated_cv_error**2, axis=(0, 1)))

    nodes = [-1, -1, -1]
    xpos = [15e3, 30e3, 50e3]
    ypos = [12.5e3, 12.5e3, 12.5e3]
    nodes[0] = np.argmin( (nodexy[:, 0]-xpos[0])**2 + (nodexy[:, 1]-ypos[0])**2)
    nodes[1] = np.argmin( (nodexy[:, 0]-xpos[1])**2 + (nodexy[:, 1]-ypos[1])**2)
    nodes[2] = np.argmin( (nodexy[:, 0]-xpos[2])**2 + (nodexy[:, 1]-ypos[2])**2)
    colors = cmocean.cm.algae([0.25, 0.5, 0.75])

    # Pick low (5%), median, and high (95%) ensemble members
    qntls = [0.95, 0.5, 0.05]
    m_low = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, qntls[2])))
    m_med = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, qntls[1])))
    m_high = np.nanargmin(np.abs(rmse_m - np.nanquantile(rmse_m, qntls[0])))
    ms = [m_high, m_med, m_low]

    # Pick logical time steps (winter, spring, summer)
    t_steps = []
    alphabet = ['a', 'b', 'c', 'd']

    ## 1. Width-averaged

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
        
    fig = plt.figure(figsize=(6, 3.75))
    gs = GridSpec(len(ms)+1, 4, wspace=0.15, hspace=0.2,
        left=0.06, bottom=0.1, right=0.98, top=0.9,
        height_ratios=[8] + len(ms)*[100])
    axs = np.array([[fig.add_subplot(gs[i+1,j]) for j in range(4)]
                        for i in range(len(ms))])
    caxs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    cm2 = mpc.LinearSegmentedColormap.from_list('', cmocean.cm.gray(np.linspace(0.05, 1, 128)))
    cmap = tools.join_cmaps(cmocean.cm.dense, cm2, average=0, N1=128, N2=128)
    
    for i in range(len(ms)):
        mi = ms[i]
        y_sim_mi = sim_y[mi]
        avg_ysim = width_average(y_sim_mi.reshape((nx, nt)))
        avg_ypred = width_average(test_y[mi].reshape((nx, nt)))
        avg_err = width_average(test_error[mi].reshape((nx, nt)))
        avg_sd = width_average((test_uq[mi] - test_lq[mi]).reshape((nx, nt)))
        dx = 2
        xedge = np.arange(0, 100+dx, dx)
        t = np.arange(0, 366)
        [tt,xx] = np.meshgrid(t,xedge)

        ax0 = axs[i, 0]
        ypc = ax0.pcolormesh(xx, tt, avg_ysim, cmap=cmap,
            vmin=0, vmax=2, shading='flat', rasterized=True)

        ax00 = axs[i, 1]
        ax00.pcolormesh(xx, tt, avg_ypred, cmap=cmap,
            vmin=0, vmax=2, shading='flat', rasterized=True)

        ax1 = axs[i, 2]
        epc = ax1.pcolormesh(xx, tt, avg_err, cmap=cmocean.cm.balance, 
            vmin=-0.5, vmax=0.5, shading='flat', rasterized=True)

        ax2 = axs[i, 3]
        spc = ax2.pcolormesh(xx, tt, avg_sd, cmap=cmocean.cm.amp, 
            vmin=0, vmax=0.5, shading='flat', rasterized=True)
        
        for j,ax in enumerate(axs[i, :]):
            col = 'w' if j<=1 else 'k'
            ax.text(0.975, 0.8, '({}{})'.format(alphabet[j],str(i+1)),
                transform=ax.transAxes, fontweight='bold',
                ha='right', va='bottom', color=col)
        axs[i,-1].text(0.975, 0.05, '$m_{{{}}}$ = {}'.format(qntls[i], ms[i]),
            transform=axs[i,-1].transAxes, ha='right', va='bottom')
        
        
    for ax in axs.flat:
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        months = np.array([4, 5, 6, 7, 8, 9, 10])
        ticks = 365/12*months
        ticklabels = ['May','','July','','Sept','','Nov']
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.set_ylim([4*365/12, 10*365/12])
        for i in range(len(nodes)):
            ax.axvline(nodexy[nodes[i], 0]/1e3, linestyle=':', color='k', linewidth=0.75)
        for j in range(len(t_steps)):
            ax.axhline(t_steps[j], linestyle=':', color='k', linewidth=0.75)
    
    for ax in axs[:, 1:].flat:
        ax.set_yticklabels([])  
    
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    # fig.text(0.0, 0.5, 'Month', rotation=90, va='center')
    fig.text(0.5, 0.02, 'Distance from terminus (km)', ha='center')
    
    cbar_ysim = fig.colorbar(ypc, cax=caxs[0], orientation='horizontal')
    cbar_ysim.set_label(r'GlaDS $f_{\rm{w}}$')
    cbar_ysim.set_ticks([0, 0.5, 1, 1.5, 2])
    cbar_ysim.set_ticklabels(['0', '0.5', '1', '1.5', '2'])

    cbar_ygp = fig.colorbar(ypc, cax=caxs[1], orientation='horizontal')
    cbar_ygp.set_label(r'GP emulator $f_{\rm{w}}$')
    cbar_ygp.set_ticks([0, 0.5, 1, 1.5, 2])
    cbar_ygp.set_ticklabels(['0', '0.5', '1', '1.5', '2'])

    cbar_error = fig.colorbar(epc, cax=caxs[2], orientation='horizontal')
    # cbar_error.set_label(r'$\Delta f_{\rm{w}}$')
    cbar_error.set_label('GP emulator error')
    cbar_error.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar_error.set_ticklabels(['-0.5', '-0.25', '0', '0.25', '0.5'])

    cbar_sd = fig.colorbar(spc, cax=caxs[3], orientation='horizontal')
    cbar_sd.set_label('95% prediction interval')
    cbar_sd.set_ticks([0, 0.25, 0.5])
    cbar_sd.set_ticklabels(['0', '0.25', '0.5'])

    for cax in caxs:
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
    
    figs.append(fig)

    ## 2. Timeseries error
    fig = plt.figure(figsize=(6, 3))
    gs = GridSpec(3, 3, wspace=0.1, hspace=0.1, left=0.07, right=0.975,
        bottom=0.12, top=0.9)
    lws = [1.5, 1]
    axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(3)]
                        for i in range(3)])
    tt = np.arange(365)
    for j,node in enumerate(nodes[:3]):
        for i,mi in enumerate(ms):
            yi_sim = sim_y[mi].reshape((nx, nt))[node, :]
            yi_pred = test_y[mi].reshape((nx, nt))[node, :]
            yi_lq = test_lq[mi].reshape((nx, nt))[node, :]
            yi_uq = test_uq[mi].reshape((nx, nt))[node, :]
            ax = axs[i,j]
            ax.fill_between(tt, yi_lq, yi_uq,
                color=colors[j], alpha=0.5, edgecolor='none')
            ax.plot(tt, yi_sim, color='k', label='GlaDS', linewidth=1.5)
            ax.plot(tt, yi_pred, color=colors[j], label='Emulator', linewidth=1)
            ax.grid(linestyle='dotted', linewidth=0.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, rotation=45)
            ax.set_xlim([4*365/12, 10*365/12])
            for k in range(len(t_steps)):
                ax.axvline(t_steps[k], linestyle=':', color='k', linewidth=0.75)
            
            ax.text(0.025, 0.8, '({}{})'.format(alphabet[j],str(i+1)), transform=ax.transAxes,
                fontweight='bold', ha='left', va='bottom')
            
            ax.spines[['right', 'top']].set_visible(False)
    
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    for ax in axs[:, 1:].flat:
        ax.set_yticklabels([])
    for i in range(3):
        ylims = np.array([ax.get_ylim() for ax in axs[i, :]])
        ylim_max = np.max(ylims)
        for ax in axs[i, :]:
            ax.set_ylim([0, ylim_max])
            # ax.set_ylim([0.5, 1.25])
        

        axs[i,-1].text(0.975, 0.8, '$m_{{{}}}$ = {}'.format(qntls[i], ms[i]), transform=ax.transAxes,
            va='bottom', ha='right')
    
    axs[1, 0].set_ylabel('Flotation fraction')
    axs[0,1].legend(bbox_to_anchor=(0,1,1,0.2), loc='lower left',
        ncols=2, frameon=False)
    
    figs.append(fig)

    return figs


def plot_scatter(config, y_sim, test_y):
    """
    Scatter plot of simulated vs emulated flot frac, N, hydraulic potential

    Parameters
    ----------
    config : module
             Training ensemble configuration
    
    y_sim : (m, n_x*n_t) array
            Simulation output matrix
    
    test_y : (m, n_x*n_t) array
           Test predictions
    
    Returns
    -------
    figure
    """
    with open(os.path.join(config.sim_dir, config.mesh), 'rb') as meshin:
        mesh = pickle.load(meshin)
    nodexy = np.array([mesh['x'], mesh['y']]).T
    # Make a scatter plot
    # fig = plt.figure(figsize=(6, 1.875))
    fig = plt.figure(figsize=(2.25, 5))
    wspace = 10
    gs = GridSpec(5, 2, height_ratios=(100, wspace, 100, wspace, 100),
        width_ratios=(100, 5), wspace=-0.15, hspace=0.2,
        left=0.15, right=0.775, bottom=0.1, top=0.975)
    axs = np.array([fig.add_subplot(gs[2*i,0]) for i in range(3)])
    # cax = fig.add_subplot(gs[-1,0])
    cax = fig.add_subplot(gs[2,1])
    rng = np.random.default_rng()
    rng_inds = rng.choice(np.arange(np.prod(y_sim.shape)), size=int(1e6), replace=False)
    y_sim_scatter = y_sim.flat[rng_inds]
    y_pred_scatter = test_y.flat[rng_inds]

    surf = 390 + 6*( (np.sqrt(nodexy[:, 0] + 5e3) - np.sqrt(5e3)))
    bed = 350
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

    countnorm = mpc.LogNorm(vmin=1e0, vmax=1e4, clip=True)

    axs[0].hexbin(y_sim_scatter, y_pred_scatter, norm=countnorm,
        cmap=cmocean.cm.rain, gridsize=100, edgecolors='none',
        extent=(ff_min, ff_max, ff_min, ff_max), rasterized=True)
    axs[0].set_xlim([ff_min, ff_max])
    axs[0].set_ylim([ff_min, ff_max])
    R2 = np.corrcoef(y_sim_scatter.flatten(), y_pred_scatter.flatten())[0,1]**2
    axs[0].text(0.95, 0.025, '$r^2={:.3f}$'.format(R2),
        ha='right', va='bottom', transform=axs[0].transAxes)
    axs[0].set_aspect('equal')
    axs[0].set_xticks(ff_ticks)
    axs[0].set_yticks(ff_ticks)
    axs[0].set_xlabel(r'$f_{\rm{w}}$ GlaDS')
    axs[0].set_ylabel(r'$f_{\rm{w}}$ emulator')
    axs[0].text(0.025, 0.95, '(a)', transform=axs[0].transAxes,
        fontweight='bold', ha='left', va='top')

    axs[1].hexbin(N_sim_scatter/1e6, N_pred_scatter/1e6, norm=countnorm,
        cmap=cmocean.cm.rain, gridsize=100, edgecolors='none',
        extent=(N_min, N_max, N_min, N_max), rasterized=True)
    axs[1].set_xlim([N_min, N_max])
    axs[1].set_ylim([N_min, N_max])
    R2 = np.corrcoef(N_sim_scatter, N_pred_scatter)[0,1]**2
    axs[1].text(0.95, 0.025, '$r^2={:.3f}$'.format(R2),
        ha='right', va='bottom', transform=axs[1].transAxes)
    axs[1].set_aspect('equal')
    axs[1].set_xticks(N_ticks)
    axs[1].set_yticks(N_ticks)
    axs[1].set_xlabel('$N$ (MPa) GlaDS')
    axs[1].set_ylabel('$N$ (MPa) emulator')
    axs[1].text(0.025, 0.95, '(b)', transform=axs[1].transAxes,
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
    axs[2].set_xlabel(r'$\phi$ (MPa) GlaDS')
    axs[2].set_ylabel(r'$\phi$ (MPa) emulator')
    axs[2].text(0.025, 0.95, '(c)', transform=axs[2].transAxes,
        fontweight='bold', ha='left', va='top')

    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label('Count (n=$10^{}$)'.format(int(np.log10(len(rng_inds)))))

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

    cputime = {}
    t_orig = time.perf_counter()
    data,model = load_model(config, config.m, config.p)
    print('main::model', model)
    
    data_dir = 'data/reference/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(config.figures):
        os.makedirs(config.figures)

    # Compute CV predictions and error

    # Binary for full space-time resolved fields
    cv_y_file = os.path.join(data_dir, 'cv_mean.npy')
    cv_lq_file = os.path.join(data_dir, 'cv_lower.npy')
    cv_uq_file = os.path.join(data_dir, 'cv_upper.npy')
    if recompute or not os.path.exists(cv_y_file):
        samples = model.get_samples(numsamples=128, nburn=256)
        for key in samples.keys():
            samples[key] = samples[key].astype(dtype)
        t0_cv = time.perf_counter()
        test_y, test_lq, test_uq = compute_test_predictions(model, 
            samples, t_test_std, quantile=0.025)
        t1_cv = time.perf_counter()
        cputime['preds'] = t1_cv - t0_cv
        np.save(cv_y_file, test_y)
        np.save(cv_lq_file, test_lq)
        np.save(cv_uq_file, test_uq)
        
    else:
        test_y = np.load(cv_y_file).astype(dtype)[:config.m, :]
        test_lq = np.load(cv_lq_file).astype(dtype)[:config.m :]
        test_uq = np.load(cv_uq_file).astype(dtype)[:config.m, :]

    rmse_wavg, rmse_ts = plot_rmse(config, sim_y=y_test_sim,
        test_y=test_y, test_error=test_y-y_test_sim, test_lq=test_lq, test_uq=test_uq)
    rmse_wavg.savefig(os.path.join(
        config.figures, 'test_error_width_avg.png'), dpi=400)
    rmse_wavg.savefig(os.path.join(
        config.figures, 'test_error_width_avg.pdf'), dpi=400)

    rmse_ts.savefig(os.path.join(
        config.figures, 'test_error_timeseries.png'), dpi=400)
    rmse_ts.savefig(os.path.join(
        config.figures, 'test_error_timeseries.pdf'), dpi=400)

    scatter_fig = plot_scatter(config, y_test_sim, test_y)
    scatter_fig.savefig(os.path.join(
        config.figures, 'test_error_scatter.png'), dpi=400)
    scatter_fig.savefig(os.path.join(
        config.figures, 'test_error_scatter.pdf'), dpi=400)

    print('Timing (seconds):', cputime)
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
