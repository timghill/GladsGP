"""
Fit GP models for different subsets of training data and different
choices for the number of principal components for scalar variables,
make scalar variable performance boxplots.

usage: fit_scalar_models.py [-h] --nsim NSIM [NSIM ...] [--recompute] train_config test_config
"""

import argparse

import os
import sys

import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction

from src import utils

def init_model(t_std, y_sim, exp_name, data_dir='data/'):
    """
    Initialize SepiaData and SepiaModel instances for scalar variable

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
    data.standardize_y()
    model = SepiaModel(data)
    return data, model

def plot_num_sims_average(train_config, test_config, nsims):
    """
    Boxplot of GP prediction error, uncertainty for training subsets

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration

    nsims : array-like
            Numbers of simulations to evaluate
    """
    coverage = 0.95

    labels = ['{:.0f} km', '{:.0f} km', '{:.1f} m']
    xlabels = [ 'Fluxgate (km)',
                'Fluxgate (km)',
                'Channel radius (m)',
    ]
    ylabels = [r'$f_Q$', r'$\log T_{\rm{s}}$ (a)', r'$L_{\rm{c}}$ (km)']

    scalar_defs = ['channel_frac',
        'log_transit_time', 'channel_length',
    ]
    thresholds = [ np.array([5, 10, 15, 20, 25, 30, -1]),
                    np.array([5, 10, 15, 20, 25, 30, -1]),
                    np.array([0.5, 1.])
    ]
    def_thresholds = [-1, -1, 0]
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)

    data_dir = 'data/scalars'
    fig_dir = os.path.join(train_config.figures, 'scalars')

    ## 1 - Plot full dataset (supplement)
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(3, 6, width_ratios=(30, 15, 150, 15, 75, 6),
        left=0.08, right=0.925, bottom=0.08, top=0.95,
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

    for k in range(len(scalar_defs)):
        print('For response:', scalar_defs[k])

        y_sim = np.load(os.path.join(test_config.sim_dir, '{}_{}.npy'.format(test_config.exp, scalar_defs[k])))
        nthresh = len(thresholds[k])
        y_error = np.zeros((nthresh*len(nsims), y_sim.shape[1]))
        pred_uncert = np.zeros((nthresh, len(nsims)))
        
        for j in range(len(nsims)):
            y_pred = np.load(os.path.join(data_dir, 'test_pred_{}_n{}.npy'.format(scalar_defs[k], nsims[j])))
            y_error[j*nthresh:(j+1)*nthresh,:] = y_pred - y_sim
            y_qntl = np.load(os.path.join(data_dir, 'test_quantiles_{}_n{}.npy'.format(scalar_defs[k], nsims[j])))
            pred_uncert[:, j] = np.mean(y_qntl[:,:,1] - y_qntl[:,:,0], axis=1)
            
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
        
        boxprops = dict(
            medianprops = {'color':'#000000'},
            boxprops = {'edgecolor':'none'},
            flierprops = {'marker':'+', 'markersize':2, 'markerfacecolor':'k', 'markeredgewidth':0.6},
            patch_artist=True,
            whiskerprops = {'linewidth':0.65}
        )

        ax0 = ax0s[k]
        ax1 = ax1s[k]
        ax2 = ax2s[k]
        box0s = ax0.boxplot(y_sim.T, positions=positions[-len(thresholds[k]):], 
            widths=boxwidth, **boxprops)

        box1s = ax1.boxplot(y_error.T, positions=positions, widths=boxwidth, **boxprops)

        for j in range(nthresh):
            box0s['boxes'][j].set_facecolor(colors[j])

            for box in box1s['boxes'][j::nthresh]:
                box.set_facecolor(colors[j])

            ax2.plot(np.arange(1, len(nsims)+1), pred_uncert[j],
                color=colors[j])
        
        # Styling
        for ax in (ax0, ax1, ax2):
            ax.grid(linestyle=':')
            ax.spines[['top', 'right']].set_visible(False)

        ax0.grid(linestyle=':')
        ax0.set_xticks([len(nsims)-1])
        ax0.set_xticklabels([test_config.m])
        ax0.set_ylabel(ylabels[k])
        ax0.set_xlim([positions[-nthresh]-0.2, positions[-1]+0.2])
        ax0.text(0, 1, alphabet[0] + str(k+1), transform=ax0.transAxes,
            fontweight='bold', ha='right', va='bottom')

        ax1.set_xticks(np.arange(len(nsims)))
        ax1.set_xticklabels(nsims)
        ax1.set_xlim([positions[0]-0.25, positions[-1]+0.25])
        ax1.text(0, 1, alphabet[1] + str(k+1), transform=ax1.transAxes,
            fontweight='bold', ha='right', va='bottom')
        
            
        ax2.grid(linestyle=':')
        ax2.set_xticks(np.arange(1, len(nsims)+1), nsims)
        ax2.set_xticklabels(nsims)
        ax2.text(0, 1, alphabet[2] + str(k+1), transform=ax.transAxes,
            fontweight='bold', ha='right', va='bottom')
        ylim2 = ax2.get_ylim()
        ax2.set_ylim([0, ylim2[1]])

        norm = Normalize(0, 1)
        cmappable = ScalarMappable(norm=norm, cmap=cmocean.cm.delta)
        cbar = fig.colorbar(cmappable, cax=caxs[k])
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(thresholds[k][thresholds[k]>0])
        cbar.set_label(xlabels[k])

        for ax in (ax0, ax1, ax2):
            if k<2:
                ax.set_xticklabels([])
            else:
                ax1.set_xlabel('Number of simulations')

    fig.savefig(os.path.join(train_config.figures, 'scalar_qoi_convergence.png'), dpi=400)
    fig.savefig(os.path.join(train_config.figures, 'scalar_qoi_convergence.pdf'))

    ## 2 - Plot just the averaged quantities
    fig_small = plt.figure(figsize=(6, 4))
    cat_colours = ['#E4003A', '#EB5B00', '#FFB200']
    gs_small = GridSpec(3, 5, width_ratios=(30, 15, 150, 15, 100),
        left=0.1, right=0.975, bottom=0.1, top=0.95,
        wspace=0.2, hspace=0.15,
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
        y_sim = np.load(os.path.join(test_config.sim_dir, '{}_{}.npy'.format(test_config.exp, scalar_defs[k])))
        y_error = np.zeros((len(nsims), y_sim.shape[1]))
        pred_uncert = np.zeros((len(nsims)))
        
        for j in range(len(nsims)):
            y_pred = np.load(os.path.join(data_dir, 'test_pred_{}_n{}.npy'.format(scalar_defs[k], nsims[j])))
            y_pred = y_pred[def_thresholds[k]]
            y_error[j,:] = y_pred - y_sim[def_thresholds[k]]
            y_qntl = np.load(os.path.join(data_dir, 'test_quantiles_{}_n{}.npy'.format(scalar_defs[k], nsims[j])))
            y_qntl = y_qntl[def_thresholds[k]]
            pred_uncert[j] = np.mean(y_qntl[:,1] - y_qntl[:,0])
        
        ax0 = ax0s_small[k]
        ax1 = ax1s_small[k]
        ax2 = ax2s_small[k]
        axs = (ax0, ax1, ax2)
        box0 = ax0.boxplot(y_sim[def_thresholds[k]], widths=0.25, **boxprops)
        box1 = ax1.boxplot(y_error.T, **boxprops)
        ax2.plot(np.arange(1, len(nsims)+1), pred_uncert, color=cat_colours[k])

        for box in box0['boxes']:
            box.set_facecolor(cat_colours[k])
        for box in box1['boxes']:
            box.set_facecolor(cat_colours[k])

        ax0.set_xticks([1], [test_config.m])
        ax1.set_xticks(np.arange(1, len(nsims)+1), nsims)
        ax2.set_xticks(np.arange(1, len(nsims)+1), nsims)

        # ax1.set_ylabel('Error')
        ax0.text(-1.0, 0.5, ylabels[k], rotation=90, transform=ax0.transAxes, ha='right', va='center')
        ax1.text(-0.15, 0.5, 'Error', rotation=90, transform=ax1.transAxes, ha='right', va='center')
        ax2.text(-0.3, 0.5, 'Uncertainty', rotation=90, transform=ax2.transAxes, ha='right', va='center')

        ax0.set_xlim([0.65, 1.35])
        ax1.set_xlim([0.25, len(nsims)+0.75])


        ax2.set_ylim([0, 1.25*ax2.get_ylim()[1]])


        for i,ax in enumerate(axs):
            ax.grid(linestyle=':')
            ax.spines[['right', 'top']].set_visible(False)
            ax.text(0.035, 0.95, alphabet[i] + str(k+1), transform=ax.transAxes,
                fontweight='bold', ha='left', va='top')
            if k<2:
                ax.set_xticklabels([])
        
        if k==2:
            ax1.set_xlabel('Number of simulations')


        print('y_pred.shape', y_pred.shape)

    fig_small.savefig(os.path.join(train_config.figures, 'scalar_convergence.png'), dpi=400)
    fig_small.savefig(os.path.join(train_config.figures, 'scalar_convergence.pdf'))

def plot_GP_lengthscales(train_config, test_config, nsim):
    def_thresholds = [6, 6, 0]
    thresholds = [ np.array([5, 10, 15, 20, 25, 30, -1]),
                    np.array([5, 10, 15, 20, 25, 30, -1]),
                    np.array([0.5, 1.])
    ]
    scalar_defs = ['channel_frac',
        'log_transit_time', 'channel_length']
    # ylabels = [r'$f_Q$', r'$\log T_{\rm{s}}$ (a)', r'$L_{\rm{c}}$ (km)']
    ylabels = [r'$f_Q$', r'$\log T_{\rm{s}}$ (a)', r'$L_{\rm{c}}$ (km)']
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    labels = ['{:.0f} km', '{:.0f} km', '{:.1f} m']

    data_dir = 'data/scalars'
    fig_dir = os.path.join(train_config.figures, 'scalars')
    fig = plt.figure(figsize=(6, 8.5))
    gs = GridSpec(8, 3, bottom=0.125, left=0.085, right=0.975, top=0.975,
        hspace=0., wspace=0.25)
    axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(3)] for i in range(8)])

    for j in range(3):
        if thresholds[j][-1]==-1:
            cticks = np.zeros(len(thresholds[j]))
            cticks[:-1] = np.linspace(0.15, 0.85, len(thresholds[j])-1)
            colors = cmocean.cm.delta(cticks)
            colors[-1] = [0.4, 0.4, 0.4, 1.]
            cticks = cticks[:-1]
        else:
            cticks = np.linspace(0.15, 0.85, len(thresholds[j]))
            colors = cmocean.cm.delta(cticks)
        nt = len(thresholds[j])
        beta_median = np.zeros((8, nt, len(nsim)))
        beta_range = np.zeros((8, 2, nt, len(nsim)))
        for tindex in range(len(thresholds[j])):
            for k,m in enumerate(nsim):
                modelfile = os.path.join(data_dir, '{}_{}_n{}_t{}.pkl'.format(
                    train_config.exp, scalar_defs[j], m, tindex)
                )
                modelj = np.load(modelfile, allow_pickle=True)
                beta_samples = modelj['samples']['betaU'][:, 1:, 0]
                beta_median[:, tindex, k] = np.median(beta_samples, axis=0)
                beta_range[:, 0, tindex, k] = np.quantile(beta_samples, 0.025, axis=0)
                beta_range[:, 1, tindex, k] = np.quantile(beta_samples, 0.975, axis=0)
        print(beta_median.shape)
        for i in range(8):
            ax = axs[i, j]
            for tindex in range(len(thresholds[j])):
                label = labels[j].format(thresholds[j][tindex])
                if thresholds[j][tindex]==-1:
                    label = 'Mean'
                ax.plot(np.arange(len(nsim)), beta_median[i, tindex], color=colors[tindex],
                    label=label)
                ax.fill_between(np.arange(len(nsim)), beta_range[i, 0, tindex], beta_range[i, 1, tindex],
                    alpha=0.3, color=colors[tindex])

            ax.set_xticks(np.arange(len(nsim)), nsim)
            ax.grid(linestyle=':')
            ax.spines[['right', 'top']].set_visible(False)
        
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([])
    
    for ax in axs[-1,:]:
        ax.set_xlabel('Number of simulations')
    
    for i,ax in enumerate(axs[:, 0]):
        ax.set_ylabel(r'$\beta$ ({})'.format(t_names[i]))
    # print(beta_median)
    # plt.show()
    axs[-1,0].legend(bbox_to_anchor=(0, -0.9, 2.5, 0.5), loc='upper center', 
            ncols=4, frameon=False, title='Fluxgate position')

    axs[-1,2].legend(bbox_to_anchor=(0, -0.9, 1, 0.5), loc='upper center', 
            ncols=1, frameon=False, title='Channel radius')
    for i,ax in enumerate(axs[0]):
        ax.set_title(ylabels[i], fontsize=8)
    fig.savefig('figures/scalar_lengthscale_convergence.png', dpi=400)
    fig.savefig('figures/scalar_lengthscale_convergence.pdf')
    

def fit(train_config, test_config, nsims, recompute=False):
    """
    Fit GP emulators of scalar variables.

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration

    nsims : array-like
            Numbers of simulations to evaluate
    
    recompute : bool, optional
                Force to redo MCMC sample and overwrite on disk
    """
    # Define scalar variables
    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    thresholds = [ np.array([5, 10, 15, 20, 25, 30, -1]),
                    np.array([5, 10, 15, 20, 25, 30, -1]),
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

    data_dir = 'data/scalars'
    fig_dir = os.path.join('figures/', 'scalars')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for k in range(len(scalar_defs)):
        print('For response', scalar_defs[k])
        Y_fname = os.path.join(train_config.sim_dir, 
            '{exp}_{qoi}.npy'.format(exp=train_config.exp, qoi=scalar_defs[k]))
        y_sim = np.load(Y_fname).T
        for m_index in range(len(nsims)):
            print('Using {:d} simulations'.format(nsims[m_index]))
            m = nsims[m_index]

            all_preds = np.zeros((len(thresholds[k]), t_pred.shape[0]))
            all_quantiles = np.zeros((len(thresholds[k]), t_pred.shape[0], 2))
            test_y_file = os.path.join(data_dir, 
                'test_pred_{qoi}_n{m}.npy'.format(qoi=scalar_defs[k], m=m))
            test_qntl_file = os.path.join(data_dir, 
                'test_quantiles_{qoi}_n{m}.npy'.format(qoi=scalar_defs[k], m=m))

            ti_std = t_std[:m, :]
            ti_phys = t_phys[:m, :]

            for j in range(len(thresholds[k])):

                # Arbitrarily take one of the fluxgate positions
                y_simj = (np.vstack(y_sim[:m, j]))

                data, model = init_model(ti_std, y_simj, exp_name, 
                    data_dir=train_config.data_dir)

                # Fit model with MCMC sampling
                model_file = os.path.join(data_dir, 
                    '{exp}_{qoi}_n{m}_t{threshold}'.format(exp=train_config.exp, 
                        qoi=scalar_defs[k], m=m, threshold=j))
                if recompute or not os.path.exists(model_file + '.pkl'):
                    # fit_model(model, model_file)
                    model.tune_step_sizes(100, 10)
                    model.do_mcmc(512)
                    model.save_model_info(file_name=model_file)
                
                model.restore_model_info(model_file)

                samples = model.get_samples(nburn=256, numsamples=64)

                # Compute CV predictions and error                
                print('Predicting on test set...')
                test_preds = SepiaEmulatorPrediction(model=model,
                    t_pred=t_pred, samples=samples)
                test_yvals = test_preds.get_y()
                test_y = test_yvals.mean(axis=0)
                test_lq = np.quantile(test_yvals, 0.025, axis=0)
                test_uq = np.quantile(test_yvals, 0.975, axis=0)
                all_preds[j,:] = test_y.squeeze()
                all_quantiles[j,:,0] = test_lq.squeeze()
                all_quantiles[j,:,1] = test_uq.squeeze()
            np.save(test_y_file, all_preds)
            np.save(test_qntl_file, all_quantiles)
    return

def summarize_performance(train_config, test_config):
    """
    Print summary statistics of emulator performance for scalar variables.

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration
    """
    m_train = train_config.m
    m_test = test_config.m
    scalar_vars = ['channel_frac', 'log_transit_time', 'channel_length']
    def_thresholds = [6, 6, 0]
    for k in range(len(scalar_vars)):
        print(scalar_vars[k])
        y_sim = np.load(os.path.join(test_config.sim_dir,
            train_config.exp+'_' + scalar_vars[k] + '.npy'))
        y_sim = y_sim[def_thresholds[k]]
        # print(y_sim.shape)

        y_pred = np.load(os.path.join('data/scalars',
            'test_pred_{}_n{}.npy'.format(
                scalar_vars[k], m_train)))
        y_pred = y_pred[def_thresholds[k]]
        # print(y_pred.shape)

        test_err = y_pred - y_sim

        test_rmse = np.sqrt(np.mean(test_err**2))
        mape_arg = test_err/y_sim
        # Neglect +-10% of data closest to zero to avoid MAPE blowing up
        mape_arg[np.abs(y_sim)<np.quantile(np.abs(y_sim), 0.2)] = np.nan
        test_mape = np.nanmean(np.abs(mape_arg))
        print('Range of values:', np.min(y_sim), np.max(y_sim))
        print('RMSE:', test_rmse)
        print('5% error:', np.quantile(test_err, 0.05))
        print('25% error:', np.quantile(test_err, 0.25))
        print('50% error:', np.quantile(test_err, 0.50))
        print('75% error:', np.quantile(test_err, 0.75))
        print('95% error:', np.quantile(test_err, 0.95))
        print('MAPE:', 100*test_mape)
        print('5% percent error:', 100*np.nanquantile(mape_arg, 0.05))
        print('25% percent error:', 100*np.nanquantile(mape_arg, 0.25))
        print('50% percent error:', 100*np.nanquantile(mape_arg, 0.50))
        print('75% percent error:', 100*np.nanquantile(mape_arg, 0.75))
        print('95% percent error:', 100*np.nanquantile(mape_arg, 0.95))
        print('\n')



def main():
    """
    usage: fit_scalar_models.py [-h] --nsim NSIM [NSIM ...] [--recompute] train_config test_config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)
    test_config = utils.import_config(args.test_config)

    if args.recompute:
        fit(train_config, test_config, nsims=args.nsim, recompute=args.recompute)

    plot_num_sims_average(train_config, test_config, args.nsim)
    plot_GP_lengthscales(train_config, test_config, args.nsim)
    summarize_performance(train_config, test_config)

if __name__=='__main__':
    main()