"""
Fit GP to GlaDS ensemble simulations and evaluate with LOOCV,
for different scalar outputs of interest

Figures
    * MCMC diagnostics
        * Trace, ???
    * Error boxplots
        * RMSE, MAPE
        * Mean absolute standardized error (by CV std.)
    * Errors in parameter space
        * RMSE, MAPE, standardized
"""

import argparse

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean
# from scipy import stats

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction
from sepia.SepiaPrior import SepiaPrior

from src import utils

def init_model(t_std, y_sim, exp_name, data_dir='data/'):
    """
    Initialize SepiaData and SepiaModel instances

    Parameters
    ----------
    t_std : (n simulations,)
            Standardized simulation design matrix
    
    exp_name : str
               Name for simulation, used to generate file paths
    
    Returns:
    --------
    SepiaData, SepiaModel
    """
    data = SepiaData(t_sim=t_std, y_sim=y_sim)
    data.transform_xt()
    y_mean = np.mean(y_sim, axis=0)
    y_sd = np.std(y_sim, axis=0)
    data.standardize_y()
    model = SepiaModel(data)
    return data, model

# def mcmc_diagnostics(model, samples):
#     """
#     Do MCMC diagnostics. Add diagnostics as we go.
#     """
#     trace_fig = SepiaPlot.mcmc_trace(samples)
#     trace_fig.subplots_adjust(right=0.7)
#     box_fig = SepiaPlot.rho_box_plots(model)
#     stats = SepiaPlot.param_stats(samples)
#     return (trace_fig, box_fig, stats)

def threshold_boxplots(train_config):
    # Load data and initialize model

    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [  np.arange(5e3, 35e3, 5e3)/1e3,
                    np.arange(5e3, 35e3, 5e3)/1e3,
                    [0.5, 1.],
    ]
    xlabels = [ 'Fluxgate (km)',
                'Fluxgate (km)',
                'Channel radius (m)',
    ]
    ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None)
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1)
    t_log = np.loadtxt(train_config.X_log, delimiter=',', skiprows=1)
    t_std = t_std[:train_config.m, :]
    t_phys = t_phys[:train_config.m, :]

    qq = 0.05
    t_lq = np.quantile(t_std, qq, axis=0)
    t_uq = np.quantile(t_std, 1-qq, axis=0)
    test_mask = np.logical_and(t_std>=t_lq, t_std<=t_uq).all(axis=1)
    m_mask = len(np.where(test_mask)[0])
    print('m_mask:', m_mask)

    boxprops = dict(
    medianprops = {'color':'#000000'},
    boxprops = {'edgecolor':'none', 'facecolor':'#888888'},
    patch_artist=True)
    alphabet = ['a', 'b']


    boxplot_abserror = []
    boxplot_relerror = []
    for k in range(len(scalar_defs)):

        data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
        fig_dir = os.path.join(train_config.figures, 'scalars')
        
        n_defs = len(thresholds[k])
        abserror = np.zeros((n_defs, m_mask))
        # relerror = np.zeros((n_defs, m_mask))
        relerror = n_defs*[0]

        for j in range(len(thresholds[k])):

            Y_fname = os.path.join(train_config.sim_dir, 
                '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
            print(Y_fname)
            y_sim = np.load(Y_fname).T[:train_config.m]

            # Arbitrarily take one of the fluxgate positions
            y_sim = (np.vstack(y_sim[:, j]))
            print('\ty_sim.shape:', y_sim.shape)
            # print(y_sim)

            # Compute CV predictions and error
            test_y_file = os.path.join(data_dir, 
                'test_{qoi}_{threshold}.npy'.format(qoi=scalar_defs[k], threshold=j))
            print('cv file:', test_y_file)
            test_qntl_file = os.path.join(data_dir, 
                'test_{qoi}_{threshold}.npy'.format(qoi=scalar_defs[k], threshold=j))
            test_y = np.load(test_y_file)
            test_std_pred = np.load(test_qntl_file)
            
            test_error = test_y - y_sim
            print('test_error.shape', test_error.shape)

            abserror[j, :] = (test_error[test_mask]).flatten()

            re = test_error[test_mask]/y_sim[test_mask]
            eps = np.quantile(np.abs(y_sim), 0.1)
            eps = max(eps, 0.1)
            print('eps:', eps)
            re[np.abs(y_sim[test_mask])<eps] = np.nan
            relerror[j] = 100*re[~np.isnan(re)].flatten()
        
        boxplot_abserror.append(abserror)
        boxplot_relerror.append(relerror)

        fig, axs = plt.subplots(figsize=(8, 4), ncols=2)
        axs[0].boxplot(abserror.T, labels=thresholds[k], **boxprops)
        axs[1].boxplot(relerror, labels=thresholds[k], **boxprops)
        axs[0].set_title('Error')
        axs[1].set_title('Percent error (%)')
        axs[0].grid(linestyle=':', linewidth=0.5)
        axs[1].grid(linestyle=':', linewidth=0.5)
        axs[0].text(0.025, 0.95, alphabet[0], transform=axs[0].transAxes,
            fontweight='bold', va='top', ha='left')
        axs[1].text(0.025, 0.95, alphabet[1], transform=axs[1].transAxes,
            fontweight='bold', va='top', ha='left')
        axs[0].set_ylabel(ylabels[k])
        fig.text(0.5, 0.025, xlabels[k], va='bottom', ha='center')
        fig.subplots_adjust(bottom=0.15, left=0.125, right=0.95, top=0.9)
        fig.savefig(os.path.join(fig_dir, '{qoi}_boxplot.png'.format(qoi=scalar_defs[k])), dpi=400)

        fig, ax = plt.subplots()
        ax.scatter(test_y, y_sim)
    return


def response_surface_profiles(train_config):
    # Load data and initialize model
    train_config.m = 128

    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [  np.arange(5e3, 35e3, 5e3)/1e3,
                    np.arange(5e3, 35e3, 5e3)/1e3,
                    [0.5, 1.],
    ]
    labels = ['{:.0f} km', '{:.0f} km', '{:.1f} m']
    xlabels = [ 'Fluxgate position (km)',
                'Fluxgate position (km)',
                'Channel radius (m)',
    ]
    ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']
    ylims = [[0, 1], [-1.5, 0.25], [0, 1500]]

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None)
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1)
    t_log = np.loadtxt(train_config.X_log, delimiter=',', skiprows=1)
    t_std = t_std[:train_config.m, :]
    t_phys = t_phys[:train_config.m, :]

    # Prediction points
    ntarget = 11
    nintegrate = 30
    n_dim = t_std.shape[1]
    sampler = stats.qmc.LatinHypercube(n_dim-1, 
        optimization='random-cd', scramble=False, seed=20240418)
    Xintegrate = sampler.random(n=nintegrate)
    Xpred = np.zeros((ntarget*nintegrate, n_dim))
    Xpred[:, 1:] = np.tile(Xintegrate, (ntarget, 1))
    xtarget = np.linspace(0, 1, ntarget)
    Xpred[:, 0]  = np.kron(xtarget, np.ones(nintegrate))

    fig_dir = os.path.join(train_config.figures, 'scalars')
    # fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(8, 6))
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(n_dim, len(scalar_defs),
        hspace=0.25, wspace=0.25, 
        bottom=0.1, top=0.975, left=0.05, right=0.975)
    axs = np.array(
        [[fig.add_subplot(gs[i,j]) for j in range(len(scalar_defs))]
            for i in range(n_dim)])
    

    for k in range(len(scalar_defs)):

        data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
        
        n_defs = len(thresholds[k])

        colors = cmocean.cm.delta(np.linspace(0.15, 0.85, len(thresholds[k])))

        for j in range(len(thresholds[k])):

            Y_fname = os.path.join(train_config.sim_dir, 
                '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
            print(Y_fname)
            y_sim = np.load(Y_fname).T[:train_config.m]

            # Arbitrarily take one of the fluxgate positions
            y_sim = (np.vstack(y_sim[:, j]))
            print('\ty_sim.shape:', y_sim.shape)
            # print(y_sim)

            data, model = init_model(t_std, y_sim, train_config.exp, 
                data_dir=train_config.data_dir)
            
            data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
            fig_dir = os.path.join(train_config.figures, 'scalars')

            # Fit model with MCMC sampling
            model_file = os.path.join(data_dir, 
                '{exp}_{qoi}_n{m}_t{threshold}'.format(exp=train_config.exp, qoi=scalar_defs[k], m=train_config.m, threshold=j))
            print('main::model_file:', model_file)
            
            model.restore_model_info(model_file)

            samples = model.get_samples(nburn=500, numsamples=30)


            for d_index in range(n_dim):
                dnums = np.arange(n_dim)
                dnums = dnums + d_index
                dnums = np.mod(dnums, n_dim)
                # dnums = np.mod(dnums-d_index, n_dim)
                xpred = np.zeros(Xpred.shape)
                xpred[:, d_index] = Xpred[:, 0]
                xpred[:, dnums[1:]] = Xpred[:, 1:]

                GPpred = SepiaEmulatorPrediction(model=model, t_pred=xpred, samples=samples)
                Ypred = GPpred.get_y()
                Ymean = np.mean(Ypred, axis=0).flatten()

                Yint = np.zeros(xtarget.shape)
                for i in range(ntarget):
                    start = i*nintegrate
                    end = (i+1)*nintegrate
                    Yint[i] = np.mean(Ymean[start:end])
                
                ax = axs[d_index, k]
                ax.plot(xtarget, Yint, color=colors[j], label=labels[k].format(thresholds[k][j]))
                ax.grid(linestyle=':')
                ax.set_xlim([0, 1])
                ax.set_ylim(ylims[k])
                ax.spines[['right', 'top']].set_visible(False)
                ax.set_xlabel(t_names[d_index], labelpad=-6)
                ax.patch.set_alpha(0.)
            
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    axs[-1,0].legend(bbox_to_anchor=(0, -0.75, 2.5, 0.5), loc='upper center', 
            ncols=3, frameon=False, title='Fluxgate position')

    axs[-1,2].legend(bbox_to_anchor=(0, -0.75, 1, 0.5), loc='upper center', 
            ncols=1, frameon=False, title='Channel radius')
    # for ax in axs[:, 1:].flat:
    #     ax.set_yticklabels([])
    
    # for i in range(n_dim):
    #     axs[i,0].set_xlabel(t_names[i])
    
    # for ax in axs[::2, 0].flat:
    #     ax.set_yticklabels([])
    
    for k in range(len(scalar_defs)):
        axs[0,k].set_title(ylabels[k], fontsize=11)
        
        # axs.flat[0].get_shared_x_axes().join(axs.flat[0], *axs.flat[1:])

        
        # axs[0, 0].legend(bbox_to_anchor=(0, 1, 4, 0.5), loc='lower left', 
        #     ncols=len(thresholds[k]), frameon=False)
        # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, hspace=0.25)
        # fig.text(0.025, 0.5, ylabels[k], rotation=90, va='center', ha='center')
    
    # for ax in axs[:
    fig.savefig(os.path.join(fig_dir, 'mean_response_profiles.png'), dpi=400)

    return


def response_surface_pairs_average(train_config, recompute=False):
    # Load data and initialize model

    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [  np.arange(5e3, 35e3, 5e3)/1e3,
                    np.arange(5e3, 35e3, 5e3)/1e3,
                    [0.5, 1.],
    ]
    threshold_index = [-1, -1, 0]
    # average = [True, True, False]
    pairs = [   [ (0, 1), (4, 1), (2, 3) ],
                [ (0, 5), (0, 2), (3, 5) ],
                [ (0, 1), (2, 4), (3, 5) ],
    ]
    labels = ['{:.0f} km', '{:.0f} km', '{:.1f} m']
    xlabels = [ 'Fluxgate position (km)',
                'Fluxgate position (km)',
                'Channel radius (m)',
    ]
    ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']
    ylims = [[0, 1], [-1.5, 0.5], [0, 1500]]

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None)
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1)
    t_log = np.loadtxt(train_config.X_log, delimiter=',', skiprows=1)
    t_std = t_std[:train_config.m, :]
    t_phys = t_phys[:train_config.m, :]

    azim = np.array([[-90-15, -90+30, -90+60],
                    [90-30, 90-30, 90+60],
                    [30,-90+60,90+30]])
    elev = np.array([[22.5, 22.5, 22.5],
                    [22.5, 22.5, 22.5],
                    [15, 22.5, 22.5]])

    # view_opts = ({'azim':-90+60, 'elev':22.5}, {'azim':-90-60, 'elev':22.5}, {'azim':-90-45, 'elev':22.5})

    # Prediction points
    ntarget = 20
    nintegrate = 20
    n_dim = t_std.shape[1]
    sampler = stats.qmc.LatinHypercube(n_dim-2, 
        optimization='random-cd', scramble=False, seed=20240418)
    Xintegrate = sampler.random(n=nintegrate)
    Xpred = np.zeros((ntarget**2 * nintegrate, n_dim))
    Xpred[:, 2:] = np.tile(Xintegrate, (ntarget**2, 1))
    dx = 1/ntarget
    xleft = np.arange(0, 1, dx)
    xright = np.arange(dx, 1+dx, dx)
    xmid = 0.5*(xleft + xright)
    xedge = np.arange(0, 1+dx, dx)

    [xx, yy] = np.meshgrid(xmid, xmid)
    xvec = xx.flatten()
    yvec = yy.flatten()
    Xpred[:, 0]  = np.kron(xvec, np.ones(nintegrate))
    Xpred[:, 1]  = np.kron(yvec, np.ones(nintegrate))

    xxpad,yypad = np.meshgrid(xedge, xedge)

    fig = plt.figure(figsize=(8, 6.5))
    pad = -30
    gs = GridSpec(3, 6, width_ratios=(100, pad, 100, pad, 100, 10),
        wspace=0.67, hspace=0.1, left=0.05, right=0.9, bottom=0.1, top=0.975)
    axs = np.array([[fig.add_subplot(gs[i,2*j], projection='3d') for j in range(3)]
        for i in range(3)])
    caxs = np.array([fig.add_subplot(gs[i,-1]) for i in range(3)])
    fig_dir = os.path.join(train_config.figures, 'scalars')

    for k in range(len(scalar_defs)):

        data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
        fig_dir = os.path.join(train_config.figures, 'scalars')

        Y_fname = os.path.join(train_config.sim_dir, 
            '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
        print(Y_fname)
        y_sim = np.load(Y_fname).T[:train_config.m]

        # Arbitrarily take one of the fluxgate positions
        if threshold_index[k]==-1:
            y_sim = np.vstack(np.mean(y_sim, axis=1))
        else:
            y_sim = (np.vstack(y_sim[:, threshold_index[k]]))
        print('\ty_sim.shape:', y_sim.shape)
        # print(y_sim)

        data, model = init_model(t_std, y_sim, train_config.exp, 
            data_dir=train_config.data_dir)

        # Fit model with MCMC sampling
        model_file = os.path.join(data_dir, 
            '{exp}_{qoi}_n{m}_t{threshold}'.format(exp=train_config.exp, 
            qoi=scalar_defs[k], m=train_config.m, threshold=threshold_index[k]))
        print('main::model_file:', model_file)
        
        if not os.path.exists(model_file+'.pkl'):
            fit_model(model, model_file)

        model = load_model(model, model_file)

        samples = model.get_samples(nburn=500, numsamples=10)


        for d_index in range(3):
            print('d_index:', d_index)
            dnums = np.arange(n_dim)
            pair = pairs[k][d_index]
            dnums = np.delete(dnums, pair)
            xpred = np.zeros(Xpred.shape)
            xpred[:, dnums] = Xpred[:, 2:]
            xpred[:, pair] = Xpred[:, :2]
            Yint = np.zeros(xx.shape)
            response_fname = os.path.join(data_dir, 
                'response_surface_avg_{}_{}_{}.npy'.format(
                    scalar_defs[k], pair[0], pair[1]))

            if recompute or not os.path.exists(response_fname):

                for i in range(ntarget):
                    for j in range(ntarget):
                        
                        start = (i*ntarget+j)*nintegrate
                        end = start + nintegrate

                        GPpred = SepiaEmulatorPrediction(model=model, t_pred=xpred[start:end, :],
                            samples=samples)
                        Ypred = GPpred.get_y()
                        Yint[i, j] = np.mean(Ypred)
                
                np.save(response_fname, Yint.astype(np.float32))
            
            Yint = np.load(response_fname)

            ax = axs[k, d_index]
            ax.patch.set_alpha(0.)
            pc = ax.plot_surface(xx, yy, Yint, 
                vmin=ylims[k][0], vmax=ylims[k][1], cmap=cmocean.cm.curl,
                edgecolor='none')
            aspect = (np.ptp(xx), np.ptp(yy), np.ptp(Yint))
            # ax.set_box_aspect((1, 1, 500))
            # ax.set_aspect('auto')
            # ax
            cb = fig.colorbar(pc, cax=caxs[k])
            cb.set_label(ylabels[k])
            # ax.imshow(Yint)
            ax.set_xlabel(t_names[pair[0]])
            ax.set_ylabel(t_names[pair[1]])
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_zlim(ylims[k])
            ax.view_init(azim=azim[k][d_index], elev=elev[k][d_index])
            # ax.set_title(ylabels[k])
            ax.set_aspect('auto')
    fig.savefig(os.path.join(fig_dir, 'mean_response_pairs_averaged.png'), dpi=400)
    return


def compute_num_sims_average(train_config, test_train_config):
    # Load data and initialize model
    nsims = [16, 32, 64, 128, 256]

    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [ np.array([5, 10, 15, 20, 35, 30, -1]),
                    np.array([5, 10, 15, 20, 35, 30, -1]),
                    np.array([0.5, 1.])
    ]
    def_thresholds = [-1, -1, 0.5]

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None)
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1)
    t_log = np.loadtxt(train_config.X_log, delimiter=',', skiprows=1)
    exp_name = train_config.exp

    test_std = np.loadtxt(test_train_config.X_standard, delimiter=',', skiprows=1,
        comments=None)

    n_dim = t_std.shape[1]
    sampler = stats.qmc.LatinHypercube(n_dim, 
        optimization='random-cd', scramble=False, seed=42186)
    t_integrate = sampler.random(n=256)

    for k in range(len(scalar_defs)):
        print('For response', scalar_defs[k])
        data_dir = os.path.join(exp_name, 'data/{qoi}'.format(qoi=scalar_defs[k]))
        fig_dir = os.path.join(train_config.figures, 'scalars')
        datafile = os.path.join(data_dir, 'n_sim_pred.csv')
        betafile = os.path.join(data_dir, 'n_sim_beta.csv')
        betalqfile = os.path.join(data_dir, 'n_sim_beta_lower.csv')
        betauqfile = os.path.join(data_dir, 'n_sim_beta_u.csv')
        simfile = os.path.join(data_dir, 'n_sim_ysim.csv')

        boxi = []
        boxj = []
        positions = []
        counter_nsims = []
        counter_thresh = []

        betaU = np.zeros((len(nsims), len(thresholds[k]), 8))
        betaU_lower = np.zeros((len(nsims), len(thresholds[k]), 8))
        betaU_upper = np.zeros((len(nsims), len(thresholds[k]), 8))

        integrated_sd = np.zeros((len(nsims), len(thresholds[k])))

        for m_index in range(len(nsims)):
            m = nsims[m_index]

            ti_std = t_std[:m, :]
            ti_phys = t_phys[:m, :]

            for j in range(len(thresholds[k])):
                counter_nsims.append(m)
                counter_thresh.append(thresholds[k][j])

                Y_fname = os.path.join(train_config.sim_dir, 
                    '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
                if thresholds[k][j]==-1:
                    y_all = np.load(Y_fname).T[:m]
                    y_sim = np.vstack(np.mean(y_all, axis=1))
                else:
                    y_sim = np.load(Y_fname).T[:m]
                    y_sim = y_sim[:, j:j+1]

                Y_test_fname = os.path.join(test_train_config.sim_dir, 
                    '{exp}_{qoi}.npy'.format(exp=test_train_config.exp, qoi=scalar_defs[k]))
                y_test = np.load(Y_test_fname).T

                if thresholds[k][j]==-1:
                    y_test_all = y_test[:, :]
                    y_test = np.vstack(np.mean(y_test_all, axis=1))
                else:
                    y_test = y_test[:, j:j+1]

                data, model = init_model(ti_std, y_sim, exp_name, 
                    data_dir=train_config.data_dir)
                model_file = os.path.join(data_dir, 
                    '{exp}_{qoi}_n{m}_t{threshold}'.format(exp=train_config.exp, 
                        qoi=scalar_defs[k], m=m, threshold=j))
                if not os.path.exists(model_file + '.pkl'):
                    model = fit_model(model, model_file)

                model = load_model(model, model_file)
                samples = model.get_samples(64, nburn=250-128)

                # Get GP length scales. Discard first, it goes with the dummy x variable
                mean_betaU = np.median(samples['betaU'], axis=0)[1:]
                coverage = 0.95
                lowq = np.quantile(samples['betaU'], (1 - coverage)/2, axis=0)[1:]
                upq = np.quantile(samples['betaU'], (1 + coverage)/2, axis=0)[1:]
                betaU[m_index, j, :] = mean_betaU
                betaU_lower[m_index, j, :] = lowq
                betaU_upper[m_index, j, :] = upq


                test_pred = SepiaEmulatorPrediction(
                    t_pred=test_std, model=model, samples=samples)
                y_pred = test_pred.get_y()
                y_pred_mean = np.mean(y_pred, axis=0)

                int_pred = SepiaEmulatorPrediction(
                    t_pred=t_integrate, model=model, samples=samples)
                int_y = int_pred.get_y()
                y_lower = np.quantile(int_y, (1 - coverage)/2, axis=0)
                y_upper = np.quantile(int_y, (1 + coverage)/2, axis=0)
                int_sd = np.mean(y_upper - y_lower)
                integrated_sd[m_index, j] = int_sd
                test_error = y_pred_mean - y_test
                boxi.append(test_error.flatten())
                boxj.append(y_sim.flatten())


        boxarr = np.zeros((test_std.shape[0]+3, len(boxi)))
        boxarr[3:, :] = np.array(boxi).T
        boxarr[0, :] = counter_nsims
        boxarr[1, :] = counter_thresh
        boxarr[2, :] = integrated_sd.flatten()
        header = 'Num sims, Threshold, Uncert, Error,'
        np.savetxt(datafile, boxarr.T, delimiter=',', fmt='%.3e', header=header)

        simarr = np.zeros((nsims[-1]+2, len(boxj)))
        for m_index in range(len(nsims)):
            istart = m_index*len(thresholds[k])
            iend = (m_index+1)*len(thresholds[k])
            unpad_data = np.array(boxj[istart:iend]).T
            padded_data = np.nan*np.zeros((nsims[-1], unpad_data.shape[1]))
            padded_data[:nsims[m_index], :] = unpad_data
            simarr[2:, istart:iend] = padded_data
            simarr[0, istart:iend] = nsims[m_index]
            simarr[1, istart:iend] = counter_thresh[istart:iend]
            header = 'Num sims, Threshold, Sim,'
        np.savetxt(simfile, simarr.T, delimiter=',', fmt='%.3e', header=header)

        [xx,yy] = np.meshgrid(nsims, thresholds[k])
        x = xx.flatten() # nsims
        y = yy.flatten() # thresholds
        allbeta = np.zeros((len(x)*3, 8+3))
        quantiles = [(1 - coverage)/2, 0.5, (1 + coverage)/2]
        for i,beta in enumerate((betaU_lower, betaU, betaU_upper)):
            betarr = np.zeros((len(x), 8+2))
            # betarr[:, 0] = counter_nsims
            # betarr[:, 1] = counter_thresh
            betarr[:, 2:] = np.reshape(beta, (len(x), 8))
            istart = i*len(x)
            iend = (i+1)*len(x)
            allbeta[istart:iend, 1:] = betarr
            allbeta[istart:iend, 0] = quantiles[i]
        beta_header = r'Quantile, Num sims, Threshold, \beta'
        np.savetxt(betafile, allbeta,
            delimiter=',', fmt='%.3e', header=beta_header)

def plot_num_sims_average(train_config, test_train_config):
    coverage = 0.95
    nsims = [16, 32, 64, 128, 256]

    labels = ['{:.0f} km', '{:.0f} km', '{:.1f} m']
    xlabels = [ 'Fluxgate (km)',
                'Fluxgate (km)',
                'Channel radius (m)',
    ]
    ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']

    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [ np.array([5, 10, 15, 20, 35, 30, -1]),
                    np.array([5, 10, 15, 20, 35, 30, -1]),
                    np.array([0.5, 1.])
    ]
    def_thresholds = [-1, -1, 0.5]
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)


    fig = plt.figure(figsize=(10, 7.5))
    gs = GridSpec(3, 6, width_ratios=(30, 15, 150, 15, 75, 6),
        left=0.1, right=0.925, bottom=0.1, top=0.95,
        wspace=0.05, hspace=0.1,
        )
    ax0s = np.array([
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),])

    ax1s = np.array([
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 2]),])
    
    ax2s = np.array([
        fig.add_subplot(gs[0, 4]),
        fig.add_subplot(gs[1, 4]),
        fig.add_subplot(gs[2, 4]),])
    caxs = np.array([
        fig.add_subplot(gs[0, 5]),
        fig.add_subplot(gs[1, 5]),
        fig.add_subplot(gs[2, 5]),])
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    fig_small = plt.figure(figsize=(8, 6))
    # cat_colours = cmocean.cm.haline([0.25, 0.4, 0.65])
    cat_colours = ['#67a9cf', '#1c9099', '#016c59']
    gs_small = GridSpec(3, 5, width_ratios=(50, 15, 150, 15, 100),
        left=0.12, right=0.975, bottom=0.12, top=0.95,
        wspace=0.1, hspace=0.15,
        )
    ax0s_small = np.array([
        fig_small.add_subplot(gs_small[0, 0]),
        fig_small.add_subplot(gs_small[1, 0]),
        fig_small.add_subplot(gs_small[2, 0]),])

    ax1s_small = np.array([
        fig_small.add_subplot(gs_small[0, 2]),
        fig_small.add_subplot(gs_small[1, 2]),
        fig_small.add_subplot(gs_small[2, 2]),])
    
    ax2s_small = np.array([
        fig_small.add_subplot(gs_small[0, 4]),
        fig_small.add_subplot(gs_small[1, 4]),
        fig_small.add_subplot(gs_small[2, 4]),])

    for k in range(len(scalar_defs)):
        print('For response', scalar_defs[k])
        data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
        fig_dir = os.path.join(train_config.figures, 'scalars')
        datafile = os.path.join(data_dir, 'n_sim_pred.csv')
        betafile = os.path.join(data_dir, 'n_sim_beta.csv')
        simfile = os.path.join(data_dir, 'n_sim_ysim.csv')
        if thresholds[k][-1]==-1:
            cticks = np.zeros(len(thresholds[k]))
            cticks[:-1] = np.linspace(0.15, 0.85, len(thresholds[k])-1)
            colors = cmocean.cm.delta(cticks)
            colors[-1] = [0.4, 0.4, 0.4, 1.]
            cticks = cticks[:-1]
        else:
            cticks = np.linspace(0.15, 0.85, len(thresholds[k]))
            colors = cmocean.cm.delta(cticks)

        # Compute boxwidth so plot looks less empty for channel length
        nthresh = len(thresholds[k])
        line = 0.25 - (0.25-0.07)*(nthresh - 2)/(7-2)
        boxwidth = min(0.25, max(0.07, line))
        
        p0 = 1.625*boxwidth
        positions = np.array(
            [[p0*(j + 0.5 -len(thresholds[k])/2) + m_index for j in range(len(thresholds[k]))] 
                for m_index in range(len(nsims))]).flatten()
        boxarr = np.loadtxt(datafile, delimiter=',', skiprows=1)
        counter_nsims = boxarr[:, 0]
        counter_thresh = boxarr[:, 1]
        integrated_sd = boxarr[:, 2].reshape((len(nsims), len(thresholds[k])))
        boxdata = boxarr[:, 2:]
        boxi = boxdata.T

        simarr = np.loadtxt(simfile, delimiter=',', skiprows=1)
        boxj = simarr[counter_nsims==256, 3:].T

        beta_arr = np.loadtxt(betafile, skiprows=1, delimiter=',')
        beta_quantiles = beta_arr[:, 0]
        betaU = beta_arr[beta_quantiles==0.5, 3:].reshape((len(nsims), len(thresholds[k]), 8))
        betaU_lower = beta_arr[np.isclose(beta_quantiles, (1-coverage)/2), 3:].reshape((len(nsims), len(thresholds[k]), 8))
        betaU_upper = beta_arr[np.isclose(beta_quantiles, (1+coverage)/2), 3:].reshape((len(nsims), len(thresholds[k]), 8))
        
        ax2 = ax2s[k]
        for j in range(len(thresholds[k])):
            ax2.plot(np.arange(len(nsims)), integrated_sd[:, j], color=colors[j])
        
        ax2_small = ax2s_small[k]
        ax2_small.plot(np.arange(len(nsims)), integrated_sd[:, -1], color=cat_colours[k])

        boxprops = dict(
            medianprops = {'color':'#000000'},
            boxprops = {'edgecolor':'none', 'facecolor':'#888888'},
            flierprops = {'marker':'+', 'markersize':3, 'markerfacecolor':'k', 'markeredgecolor':'k'},
            patch_artist=True
        )
        mean_pos = np.mean(np.reshape(positions, (len(nsims), len(thresholds[k]))), axis=1)
        ax1 = ax1s[k]
        bp = ax1.boxplot(boxi, positions=positions, widths=boxwidth, **boxprops)
        for i in range(len(bp['boxes'])):
            cindex = i % len(thresholds[k])
            bp['boxes'][i].set_color(colors[cindex])
        ax1.grid(linestyle=':')
        ax1.set_xticks(np.arange(len(nsims)))
        ax1.set_xticklabels(nsims)
        ax1.set_xlim([positions[0]-0.25, positions[-1]+0.25])
        ax1.text(0, 1, alphabet[3*k + 1], transform=ax1.transAxes,
            fontweight='bold', ha='right', va='bottom')
        
        ax0 = ax0s[k]
        pos = positions[-len(thresholds[k]):]
        bp = ax0.boxplot(boxj, positions=pos, widths=boxwidth, **boxprops)
        for i in range(len(bp['boxes'])):
            cindex = i % len(thresholds[k])
            bp['boxes'][i].set_color(colors[cindex])
        ax0.grid(linestyle=':')
        ax0.set_xticks([len(nsims)-1])
        ax0.set_xticklabels([nsims[-1]])
        ax0.set_ylabel(ylabels[k])
        ax0.set_xlim([pos[0]-0.2, pos[-1]+0.2])
        ax0.text(0, 1, alphabet[3*k], transform=ax0.transAxes,
            fontweight='bold', ha='right', va='bottom')
        
        ax = ax2s[k]
        ylim = ax.get_ylim()
        yrange = ylim[1] - 0
        ymag = 10**np.round(np.log10(yrange))
        inc = 0.25
        yrange = inc*ymag*np.round(yrange/ymag/inc + inc)

        ax.set_ylim([0, yrange])
        ax.grid(linestyle=':')
        ax.set_xticks(np.arange(len(nsims)))
        ax.set_xticklabels(nsims)
        ax.text(0, 1, alphabet[3*k+2], transform=ax.transAxes,
            fontweight='bold', ha='right', va='bottom')

        norm = Normalize(0, 1)
        cmappable = ScalarMappable(norm=norm, cmap=cmocean.cm.delta)
        cbar = fig.colorbar(cmappable, cax=caxs[k])
        cbar.set_ticks(cticks)
        # if thresholds[k][-1]==-1:
        cbar.set_ticklabels(thresholds[k][thresholds[k]>0])
        # cbar.set_ticklabels(thresholds[k])
        cbar.set_label(xlabels[k])

        ax0_small = ax0s_small[k]
        bp = ax0_small.boxplot(boxj[:, thresholds[k]==def_thresholds[k]], widths=0.25, **boxprops)
        for i in range(len(bp['boxes'])):
            bp['boxes'][i].set_color(cat_colours[k])
        ax0_small.grid(linestyle=':')
        ax0_small.set_xticks([1])
        ax0_small.set_xticklabels([nsims[-1]])
        ax0_small.set_xlim([0.75, 1.25])
        ax0_small.set_ylabel(ylabels[k])
        ax0_small.text(0, 1, alphabet[3*k], transform=ax0_small.transAxes,
            fontweight='bold', ha='right', va='bottom')

        ax1_small = ax1s_small[k]
        bp = ax1_small.boxplot(boxdata[counter_thresh==def_thresholds[k]].T,  widths=0.5, **boxprops)
        for i in range(len(bp['boxes'])):
            bp['boxes'][i].set_color(cat_colours[k])
        ax1_small.grid(linestyle=':')
        ax1_small.set_xticks(np.arange(1, len(nsims)+1))
        ax1_small.set_xticklabels(nsims)
        ax1_small.set_xlim([0.25, len(nsims)+0.75])
        ax1_small.text(0, 1, alphabet[3*k + 1], transform=ax1_small.transAxes,
            fontweight='bold', ha='right', va='bottom')

        ylim = ax2_small.get_ylim()
        yrange = ylim[1] - 0
        ymag = 10**(np.round(np.log10(yrange))-1)
        inc = 0.5
        yrange = inc*ymag*np.round(yrange/ymag/inc + inc)
        ax2_small.set_ylim([0, yrange])
        ax2_small.grid(linestyle=':')
        ax2_small.set_xticks(np.arange(len(nsims)))
        ax2_small.set_xticklabels(nsims)
        ax2_small.text(0, 1, alphabet[3*k+2], transform=ax2_small.transAxes,
            fontweight='bold', ha='right', va='bottom')

        GPfig, GPaxs = plt.subplots(2, 4, figsize=(8, 4), sharex=True, sharey=False)
        print(betaU.shape)
        for j in range(len(thresholds[k])):
            # lengthscale = 1/np.sqrt(2*betaU)
            lengthscale = betaU
            for ix in range(betaU.shape[2]):
                GPaxs.flat[ix].plot(np.arange(len(nsims)), lengthscale[:, j, ix], color=colors[j])
                GPaxs.flat[ix].fill_between(np.arange(len(nsims)), 
                    betaU_lower[:, j, ix], betaU_upper[:, j, ix], color=colors[j], 
                    alpha=0.3, edgecolor='none')
                GPaxs.flat[ix].grid(True)
                GPaxs.flat[ix].text(0.5, 1.02, t_names[ix],
                    transform=GPaxs.flat[ix].transAxes, ha='center', va='bottom')
                
                GPaxs.flat[ix].set_xticks(np.arange(len(nsims)))
                GPaxs.flat[ix].set_xticklabels(nsims)
                GPaxs.flat[ix].set_xticks(np.arange(len(nsims)))
        GPfig.text(0.5, 0.98, ylabels[k], va='top', ha='center')
        GPfig.text(0.5, 0.02, 'Number of simulations', va='bottom', ha='center')
        GPfig.text(0.02, 0.5, r'$\beta$', ha='left', va='center', rotation=90)
        GPfig.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.875, hspace=0.3, wspace=0.35)
        GPfig.savefig(os.path.join(fig_dir, 'qoi_convergence_lengthscales_avg_{}.png'.format(k)), dpi=400)
        
    for ax in ax0s[:-1]:
        ax.set_xticklabels([])
    for ax in ax1s[:-1]:
        ax.set_xticklabels([])
    for ax in ax2s[:-1]:
        ax.set_xticklabels([])
    
    ax1s[-1].set_xlabel('Number of simulations')
    ax0s[0].set_title('Ensemble')
    ax1s[0].set_title('Prediction error')
    ax2s[0].set_title('Prediction uncertainty')
    fig.savefig(os.path.join(fig_dir, 
        'qoi_convergence_boxplot_avg.png'.format(qoi=scalar_defs[k])), dpi=400)

    
    for ax in ax0s_small[:-1]:
        ax.set_xticklabels([])
    for ax in ax1s_small[:-1]:
        ax.set_xticklabels([])
    for ax in ax2s_small[:-1]:
        ax.set_xticklabels([])
        

    ax1s_small[-1].set_xlabel('Number of simulations')
    ax0s_small[0].set_title('Ensemble')
    ax1s_small[0].set_title('Prediction error')
    ax2s_small[0].set_title('Prediction uncertainty')
    fig_small.savefig(
        os.path.join(fig_dir, 'qoi_convergence_boxplot_avgonly.png'), dpi=400)

    return

# def compare_qoi(train_config):
#     # Load data and initialize model

#     scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
#     thresholds = [1, 5, 0]
#     ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']

#     t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
#         comments=None)
#     t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
#         dtype=str, comments=None)
#     t_names= [tn.strip('#') for tn in t_names]
#     t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1)
#     t_log = np.loadtxt(train_config.X_log, delimiter=',', skiprows=1)
#     t_std = t_std[:train_config.m, :]
#     t_phys = t_phys[:train_config.m, :]

#     qq = 0.05
#     t_lq = np.quantile(t_std, qq, axis=0)
#     t_uq = np.quantile(t_std, 1-qq, axis=0)
#     test_mask = np.logical_and(t_std>=t_lq, t_std<=t_uq).all(axis=1)
#     m_mask = len(np.where(test_mask)[0])
#     print('m_mask:', m_mask)

#     boxprops = dict(
#     medianprops = {'color':'#000000'},
#     boxprops = {'edgecolor':'none', 'facecolor':'#888888'},
#     widths=0.67,
#     patch_artist=True,)
#     alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

#     fig = plt.figure(figsize=(8, 6))
#     gs = GridSpec(2, 3, wspace=0.4, hspace=0.2,
#         left=0.1, bottom=0.1, right=0.95, top=0.95)
#     axs = np.array(
#         [[fig.add_subplot(gs[i,j]) for j in range(3)] for i in range(2)])
#     for k in range(len(scalar_defs)):

#         data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
#         fig_dir = os.path.join(train_config.figures, 'scalars')

#         Y_fname = os.path.join(train_config.sim_dir, 
#             '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
#         y_sim = np.load(Y_fname).T[:train_config.m]

#         y_sim = (np.vstack(y_sim[:, thresholds[k]]))
#         print('\ty_sim.shape:', y_sim.shape)
#         # print(y_sim)

#         # Compute CV predictions and error
#         test_y_file = os.path.join(data_dir, 
#             'test_{qoi}_n{m}_t{threshold}.npy'.format(qoi=scalar_defs[k], m=train_config.m, threshold=thresholds[k]))
#         print('cv file:', test_y_file)
#         test_y = np.load(test_y_file)
        
#         test_error = test_y - y_sim
#         print('test_error.shape', test_error.shape)

#         y_sim = y_sim[test_mask]
#         test_error = test_error[test_mask]

#         abserror = test_error.flatten()

#         re = test_error/y_sim
#         eps = np.quantile(np.abs(y_sim), 0.1)
#         eps = max(eps, 0.1)
#         print('eps:', eps)
#         relerror = re[np.abs(y_sim)>eps]
        
#         ax0 = axs[0, k]
#         ax1 = axs[1, k]

#         ax0.boxplot(abserror, **boxprops)
#         ylim = ax0.get_ylim()
#         ymax = np.max(np.abs(ylim))
#         ax0.set_ylim([-ymax, ymax])
#         ax1.boxplot(100*relerror, **boxprops)
#         ax1.set_ylim([-30, 30])

#         ax0.grid(linestyle=':')
#         ax1.grid(linestyle=':')

#         ax0.spines[['top', 'right']].set_visible(False)
#         ax1.spines[['top', 'right']].set_visible(False)

#         ax0.set_xlim([0.5, 1.5])
#         ax1.set_xlim([0.5, 1.5])

    
#     axs[0,0].set_ylabel('Error')
#     axs[1,0].set_ylabel('Percent error (%)')

#     for i,ax in enumerate(axs.flat):
#         ax.text(0.05, 0.95, alphabet[i], transform=ax.transAxes,
#             fontweight='bold', ha='left', va='top')
    
#     for i in range(3):
#         axs[0,i].set_xticklabels([])
#         ax = axs[1,i]
#         ax.set_xticklabels([ylabels[i]])

#     fig_dir = os.path.join(train_config.figures, 'scalars')
#     fig.savefig(os.path.join(fig_dir,
#         'qoi_abs_rel_error_boxplots.png'), dpi=400)
#     return

def compute_sensitivity(train_config):
    train_config.m = 128
    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    cols = [5, 5, 0]

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None)[:train_config.m]
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    Y = np.zeros((train_config.m, 3))

    sample_dicts = []
    models = []
    
    for k in range(len(scalar_defs)):
        Y_fname = os.path.join(train_config.sim_dir, 
            '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
        y_sim = np.load(Y_fname).T[:train_config.m]

        y_sim = np.vstack(y_sim[:, cols[k]])
        Y[:, k] = np.squeeze(y_sim)
        print('main\n\ty_sim.shape:', y_sim.shape)
        
        data, model = init_model(t_std, y_sim, train_config.exp, 
            data_dir=train_config.data_dir)
        
        data_dir = os.path.join(train_config.exp, 'data/{qoi}'.format(qoi=scalar_defs[k]))
        fig_dir = os.path.join(train_config.figures, 'scalars')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Fit model with MCMC sampling
        model_file = os.path.join(data_dir, 
            '{exp}_{qoi}_n{m}_t{threshold}'.format(exp=train_config.exp, 
                qoi=scalar_defs[k], m=train_config.m, threshold=cols[k]))
        
        model = load_model(model, model_file)
        models.append(model)

        samples = model.get_samples(nburn=500, numsamples=64)
        sample_dicts.append(samples)
    
    print('Y:', Y)
    
    def func(x):
        y = np.zeros((x.shape[0], 3))
        for k in range(len(scalar_defs)):
            gp = models[k]
            samples = sample_dicts[k]
            preds = SepiaEmulatorPrediction(model=gp, 
                t_pred=x, samples=samples)
            ymean = np.mean(preds.get_y(), axis=0).astype(np.float64)
            y[:, k] = np.squeeze(ymean)
        return y
    
    sensitivity_dir = os.path.join(train_config.data_dir, 'sensitivity/')
    print('sensitivity_dir:', sensitivity_dir)
    first_order, total_index, bootstrap = utils.saltelli_sensitivity_indices(func, t_std.shape[1], 8, bootstrap=True)
    
    header= ','.join(t_names)
    fmt = dict(delimiter=',', fmt='%.3e', header=header, comments='')
    np.savetxt(os.path.join(sensitivity_dir, 'sobol_indices_scalar.csv'),
        first_order, **fmt)
    np.savetxt(os.path.join(sensitivity_dir, 'sobol_indices_scalar_boostraplow.csv'),
        bootstrap['first_order'].confidence_interval.low, **fmt)
    np.savetxt(os.path.join(sensitivity_dir, 'sobol_indices_scalar_boostraphigh.csv'),
        bootstrap['first_order'].confidence_interval.high, **fmt)
    np.savetxt(os.path.join(sensitivity_dir, 'sobol_indices_scalar_boostrapmedian.csv'),
        np.median(bootstrap['first_order'].bootstrap_distribution, axis=-1), **fmt)
    
    np.savetxt(os.path.join(sensitivity_dir, 'total_indices_scalar.csv'),
        total_index, **fmt)
    np.savetxt(os.path.join(sensitivity_dir, 'total_indices_scalar_boostraplow.csv'),
        bootstrap['total_index'].confidence_interval.low, **fmt)
    np.savetxt(os.path.join(sensitivity_dir, 'total_indices_scalar_boostraphigh.csv'),
        bootstrap['total_index'].confidence_interval.high, **fmt)
    np.savetxt(os.path.join(sensitivity_dir, 'total_indices_scalar_boostrapmedian.csv'),
        np.median(bootstrap['total_index'].bootstrap_distribution, axis=-1), **fmt)
   
def plot_sobol_indices(train_config):
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']
    n_dim = len(train_config.theta_bounds)
    sobol_indices = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/sobol_indices_scalar.csv'), delimiter=',', skiprows=1)
    sobol_low = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/sobol_indices_scalar_boostraplow.csv'), delimiter=',', skiprows=1)
    sobol_high = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/sobol_indices_scalar_boostraphigh.csv'), delimiter=',', skiprows=1)
    sobol_median = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/sobol_indices_scalar_boostrapmedian.csv'), delimiter=',', skiprows=1)

    total_indices = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/total_indices_scalar.csv'), delimiter=',', skiprows=1)
    total_low = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/total_indices_scalar_boostraplow.csv'), delimiter=',', skiprows=1)
    total_high = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/total_indices_scalar_boostraphigh.csv'), delimiter=',', skiprows=1)
    total_median = np.loadtxt(os.path.join(train_config.exp, 
        'data/sensitivity/total_indices_scalar_boostrapmedian.csv'), delimiter=',', skiprows=1)

    sobol_indices[sobol_indices<0] = 0
    sobol_low[sobol_low<0] = 0
    sobol_high[sobol_high<0] = 0

    total_indices[total_indices<0] = 0
    total_low[total_low<0] = 0
    total_high[total_high<0] = 0

    print('sobol_indices:', sobol_indices)

    n_plot = len(sobol_indices)
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, n_plot, left=0.1, right=0.95, top=0.95)
    axs = np.array([fig.add_subplot(gs[0,j]) for j in range(n_plot)])
    
    # for i in range(n_dim):
    for j in range(n_plot):
        ax = axs[j]
        ax.barh(np.arange(n_dim)-0.2, sobol_median[j], height=0.35, color='#aaaaaa',
            label='First-order', xerr=(sobol_low[j], sobol_high[j]), ecolor='k', capsize=3)
        ax.barh(np.arange(n_dim)+0.2, total_median[j], height=0.35, color='#555555',
            label='Total', xerr=(total_low[j], total_high[j]), ecolor='k', capsize=3)
        ax.invert_yaxis()
        ax.set_xlim([0, 1.])
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid()
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.])
        # ax.set_xticklabels(['0', '0.25', '0.5', '0.75'])
        ax.set_xticklabels(['0.0', '', '0.5', '', ''])
        ax.set_yticks(np.arange(n_dim))
        ax.set_title(ylabels[j])
    
    for ax in axs[1:]:
        ax.set_yticklabels([])
    axs[0].set_yticklabels(t_names)
    axs[0].legend(bbox_to_anchor=(0., -0.135, 3.5, 0.15), loc='lower left', ncols=2, frameon=False, mode='expand')
    fig.text(0.5, 0.025, 'Sensitivity', ha='center')
    fig_dir = os.path.join(train_config.exp, 'figures/')
    fig.savefig(os.path.join(fig_dir, 'sensitivity_scalar_indices.png'), dpi=400)


        
    

def fit(train_config, test_config, nsims, recompute=False):

    # Define scalar variables
    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [ np.array([5, 10, 15, 20, 25, 30, -1]),
                    np.array([10, 15, 20, 25, 30, -1]),
                    np.array([0.5, 1.])
    ]

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(np.float32)
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_phys = np.loadtxt(train_config.X_physical, delimiter=',', skiprows=1).astype(np.float32)
    exp_name = train_config.exp

    t_pred = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1).astype(np.float32)

    for k in range(len(scalar_defs)):
        print('For response', scalar_defs[k])
        for m_index in range(len(nsims)):
            print('Using {:d} simulations'.format(nsims[m_index]))
            m = nsims[m_index]

            ti_std = t_std[:m, :]
            ti_phys = t_phys[:m, :]

            for j in range(len(thresholds[k])):
                
                Y_fname = os.path.join(train_config.sim_dir, 
                    '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
                print(Y_fname)
                y_sim = np.load(Y_fname).T[:m].astype(np.float32)


                # Arbitrarily take one of the fluxgate positions
                print('y_sim.shape:', y_sim.shape)
                y_sim = (np.vstack(y_sim[:, j]))
                print('main\n\ty_sim.shape:', y_sim.shape)

                data, model = init_model(ti_std, y_sim, exp_name, 
                    data_dir=train_config.data_dir)
                
                data_dir = 'data/{qoi}'.format(qoi=scalar_defs[k])
                fig_dir = os.path.join('figures/', 'scalars')
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)

                # Fit model with MCMC sampling
                model_file = os.path.join(data_dir, 
                    '{exp}_{qoi}_n{m}_t{threshold}'.format(exp=train_config.exp, 
                        qoi=scalar_defs[k], m=m, threshold=j))
                print('main::model_file:', model_file)
                if recompute or not os.path.exists(model_file + '.pkl'):
                    # fit_model(model, model_file)
                    model.do_mcmc(512)
                    model.save_model_info(file_name=model_file)
                
                model.restore_model_info(model_file)

                samples = model.get_samples(nburn=256, numsamples=64)

                # Compute CV predictions and error
                test_y_file = os.path.join(data_dir, 
                    'test_{qoi}_n{m}_t{threshold}.npy'.format(qoi=scalar_defs[k], m=m, threshold=j))
                test_qntl_file = os.path.join(data_dir, 
                    'test_std_{qoi}_n{m}_t{threshold}.npy'.format(qoi=scalar_defs[k], m=m, threshold=j))
                if recompute or not os.path.exists(test_y_file):
                    print('Predicting on test set...')
                    test_preds = SepiaEmulatorPrediction(model=model,
                        t_pred=t_pred, samples=samples)
                    test_yvals = test_preds.get_y()
                    test_y = test_yvals.mean(axis=0)
                    test_lq = np.quantile(test_yvals, 0.025, axis=0)
                    test_uq = np.quantile(test_yvals, 0.975, axis=0)
                    test_qntls = np.array([test_lq, test_uq]).T
                    np.save(test_y_file, test_y)
                    np.save(test_qntl_file, test_qntls)
                else:
                    test_y = np.load(test_y_file)
                    test_qntls = np.load(test_qntl_file)
    return

def main():
    """
    Command-line interface to src.model tools for model fitting

    usage: fit_all_models.py [-h] --npc NPC [NPC ...] --nsim NSIM [NSIM ...] [--recompute] train_config_file
    
    positional arguments:
        train_config_file

    options:
        -h, --help            show help message and exit
        --npc NPC [NPC ...]
        --nsim NSIM [NSIM ...]
        --recompute, -r
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)
    test_config = utils.import_config(args.test_config)

    fit(train_config, test_config, nsims=args.nsim, recompute=args.recompute)

if __name__=='__main__':
    # import argparse
    # import importlib
    # from src import utils
    # parser = argparse.ArgumentParser()
    # parser.add_argument('train_train_config')
    # parser.add_argument('test_train_config')
    # parser.add_argument('--recompute', '-r', action='store_true')
    # args = parser.parse_args()
    # train_train_config = utils.import_config(args.train_train_config)
    # test_train_config = utils.import_config(args.test_train_config)
    # main(train_train_config, test_train_config, args.nsim, recompute=args.recompute)
    main()

    # test_train_config = utils.import_config('KAN_all_test/train_config.py')
    # compute_num_sims_average(train_config, test_train_config)
    # plot_num_sims_average(train_config, test_train_config)
    # plt.close()

    # train_config.m = 32
    # response_surface_profiles(train_config)
    # plt.close()
    
    
    # compare_qoi(train_config)
    # plt.close()
    # response_surface_pairs_average(train_config, recompute=args.recompute)
    # plt.close()

    # compute_sensitivity(train_config)
    # plot_sobol_indices(train_config)
    
