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
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cmocean
# from scipy import stats

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction

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
    data.standardize_y()
    model = SepiaModel(data)
    return data, model

def plot_num_sims_average(train_config, test_config, nsims):
    coverage = 0.95

    labels = ['{:.0f} km', '{:.0f} km', '{:.1f} m']
    xlabels = [ 'Fluxgate (km)',
                'Fluxgate (km)',
                'Channel radius (m)',
    ]
    ylabels = ['Channel fraction', 'log Transit time (a)', 'Channel length (km)']

    scalar_defs = ['channel_frac',
        'log_transit_time', 'channel_length',
    ]
    # thresholds = [ np.array([5, 10, 15, 20, 35, 30, -1]),
    #                 np.array([5, 10, 15, 20, 35, 30, -1]),
    #                 np.array([0.5, 1.])
    # ]
    def_thresholds = [-1, -1, 0]
    t_names = np.loadtxt(train_config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)

    data_dir = 'data/scalars'
    fig_dir = os.path.join(train_config.figures, 'scalars')

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
    

def fit(train_config, test_config, nsims, recompute=False):
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

    if args.recompute:
        fit(train_config, test_config, nsims=args.nsim, recompute=args.recompute)

    plot_num_sims_average(train_config, test_config, args.nsim)

if __name__=='__main__':
    main()