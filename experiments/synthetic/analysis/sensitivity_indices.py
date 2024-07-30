"""
TODO
Compute Sobol' indices for functional and scalar outputs

2 figures
    1 - FF field, scalar variables sensitivity
    2 - principal components sensitivity
"""

import os
import argparse
import pickle

import numpy as np
from scipy import stats

import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction

from src import model as md
from src import utils

def compute_field_indices(config, dtype=np.float32, recompute=True):
    """
    Compute sensitivity indices using homemade code that mimics the
    scipy.stats.sobol_indices function.

    See Table 2 from:
    Andrea Saltelli, Paola Annoni, Ivano Azzini, Francesca Campolongo, 
    Marco Ratto, Stefano Tarantola
    Variance based sensitivity analysis of model output. Design and estimator
    for the total sensitivity index,
    Computer Physics Communications, Volume 181, Issue 2, 2010
    https://doi.org/10.1016/j.cpc.2009.09.018.
    """
    # Load data and initialize model
    t_std = np.loadtxt(config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_std = t_std[:config.m, :]
    y_sim = np.load(config.Y_physical).T[:config.m, :].astype(dtype)
    exp_name = config.exp
    p = config.p
    data, model = md.init_model(t_std, y_sim, exp_name, p, 
        data_dir=config.data_dir)
    
    data_dir = 'data'
    sensitivity_dir = os.path.join(data_dir, 'sensitivity/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sensitivity_dir):
        os.makedirs(sensitivity_dir)
    if not os.path.exists(config.figures):
        os.makedirs(config.figures)
    
    # Load model with MCMC sampling
    model_file = os.path.join(data_dir, 
        'models/{}_n{:03d}_p{:02d}'.format(config.exp, config.m, config.p))
    model.restore_model_info(model_file)
    samples = model.get_samples(numsamples=30)
    n_dim = t_std.shape[1]
    def func(x):
        """
        Callable for sobol indices calculation.

        Parameters
        ----------
        x:          (n, d) array
                    n: number of samples
                    d: input dimensionality
        output :    (n, p)
                    p: output dimensionality
        """
        GPpreds = SepiaEmulatorPrediction(model=model, 
            t_pred=x, samples=samples)
        ws = GPpreds.w.astype(np.float32)
        wmean = np.mean(ws, axis=0)
        return wmean.astype(np.float64)
    pca_fpattern = 'data/models/pca_{}_n{:03d}_{{}}.npy'.format(config.exp, config.m)
    S = np.load(pca_fpattern.format('S'))
    S2 = S**2
    pcvar = S2/np.sum(S2)
    pcvar = pcvar[:p]
    print('Starting sensitivity calculation...')
    indices = utils.PCA_saltelli_sensitivity_indices(func, n_dim, 8, pcvar, bootstrap=True)
    print('Done computing sensitivity indices')
    first_order = indices[0]
    total_index = indices[1]
    general_first_order = indices[2]
    general_total_index = indices[3]
    bootstrap = indices[4]

    info = dict(
        first_order = indices[0],
        total_index = indices[1],
        general_first_order = indices[2],
        general_total_index = indices[3])
    info['boostrap'] = dict(
        first_order = np.array((bootstrap['first_order'].confidence_interval.low,
                        bootstrap['first_order'].confidence_interval.high)),
        total_index = np.array((bootstrap['total_index'].confidence_interval.low,
                        bootstrap['total_index'].confidence_interval.high)),
        general_first_order = np.array((bootstrap['general_first_order'].confidence_interval.low,
                        bootstrap['general_first_order'].confidence_interval.high)),
        general_total_index = np.array((bootstrap['general_total_index'].confidence_interval.low,
                        bootstrap['general_total_index'].confidence_interval.high)),
    )

    with open('sobol_indices.pkl', 'wb') as sobin:
        pickle.dump(info, sobin)
    return info

def compute_scalar_indices(config, dtype=np.float32, recompute=True):
    """
    Compute sensitivity indices using homemade code that mimics the
    scipy.stats.sobol_indices function.

    See Table 2 from:
    Andrea Saltelli, Paola Annoni, Ivano Azzini, Francesca Campolongo, 
    Marco Ratto, Stefano Tarantola
    Variance based sensitivity analysis of model output. Design and estimator
    for the total sensitivity index,
    Computer Physics Communications, Volume 181, Issue 2, 2010
    https://doi.org/10.1016/j.cpc.2009.09.018.
    """
    # Load data and initialize model
    t_std = np.loadtxt(config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_std = t_std[:config.m, :]
    exp_name = config.exp
    p = config.p

    scalar_defs = ['channel_frac', 'log_transit_time', 'channel_length']
    cols = [5, 5, 0]
    
    data_dir = 'data'
    sensitivity_dir = os.path.join(data_dir, 'sensitivity/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sensitivity_dir):
        os.makedirs(sensitivity_dir)
    if not os.path.exists(config.figures):
        os.makedirs(config.figures)
    
    sample_dicts = []
    models = []
    for k in range(len(scalar_defs)):
        var = scalar_defs[k]
        # Load model with MCMC sampling
        model_file = os.path.join(data_dir, 
            'scalars/{}_{}_n{}_t{}'.format(
            config.exp, var, config.m, cols[k])
        )
        y_sim = np.load('../issm/train/{}_{}.npy'.format(
            config.exp, var)).T[:config.m,cols[k]]
        print('y_sim.shape:', y_sim.shape)
        data = SepiaData(t_sim=t_std, y_sim=y_sim)
        data.transform_xt()
        data.standardize_y()
        model = SepiaModel(data)
        model.restore_model_info(model_file)
        samples = model.get_samples(numsamples=16)
        sample_dicts.append(samples)
        models.append(model)
        n_dim = t_std.shape[1]
    
    # Now compute the indices
    def func(x):
        """
        Callable for sobol indices calculation.

        Parameters
        ----------
        x:          (n, d) array
                    n: number of samples
                    d: input dimensionality
        output :    (n, p)
                    p: output dimensionality
        """
        # GPpreds = SepiaEmulatorPrediction(model=model, 
        #     t_pred=x, samples=samples)
        # ws = GPpreds.w.astype(np.float32)
        # wmean = np.mean(ws, axis=0)
        # return wmean.astype(np.float64)
        y = np.zeros((x.shape[0], 3))
        for k in range(3):
            preds = SepiaEmulatorPrediction(model=models[k],
                t_pred=x, samples=sample_dicts[k])
            y_mean = np.mean(preds.get_y().astype(np.float32), axis=0).squeeze()
            y[:, k] = y_mean
        return y
        
    indices = utils.saltelli_sensitivity_indices(func, n_dim, 8, bootstrap=True)
    print('Done computing sensitivity indices')
    first_order = indices[0]
    total_index = indices[1]
    bootstrap = indices[2]

    info = dict(
        first_order = indices[0],
        total_index = indices[1]
    )
    info['boostrap'] = dict(
        first_order = np.array((bootstrap['first_order'].confidence_interval.low,
                        bootstrap['first_order'].confidence_interval.high)),
        total_index = np.array((bootstrap['total_index'].confidence_interval.low,
                        bootstrap['total_index'].confidence_interval.high))
    )

    indices_file = os.path.join(sensitivity_dir, 'scalar_indices.pkl')
    with open(indices_file, 'wb') as sobin:
        pickle.dump(info, sobin)
    return info

def plot_main_indices(config):
    indices = np.load('sobol_indices.pkl', allow_pickle=True)
    scalar_indices = np.load('data/sensitivity/scalar_indices.pkl',
        allow_pickle=True)
    alphabet = ['a', 'b', 'c', 'd']
    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(1, 4, bottom=0.15, left=0.05, right=0.95, top=.95,
        wspace=0.3)
    axs = np.array([fig.add_subplot(gs[i]) for i in range(4)])
    dy = 0.2
    y1 = np.arange(8) - dy
    y2 = np.arange(8) + dy
    axs[0].barh(y1, indices['general_first_order'], height=0.35,
        color='#aaaaaa', label='First-order',
        xerr=(indices['boostrap']['general_first_order']),
        ecolor='k', capsize=3)
    axs[0].barh(y2, indices['general_total_index'], height=0.35,
        color='#555555', label='Total',
        xerr=(indices['boostrap']['general_total_index']),
        ecolor='k', capsize=3)
    axs[0].legend(bbox_to_anchor=(0, -0.2, 1, 0.3), frameon=False)

    for k in range(3):
        ax = axs[k+1]
        ax.barh(y1, scalar_indices['first_order'][k], height=0.32,
        color='#aaaaaa', label='First-order',
        xerr=(scalar_indices['boostrap']['first_order'][:, k]),
        ecolor='k', capsize=3)
        ax.barh(y2, scalar_indices['total_index'][k], height=0.32,
        color='#555555', label='First-order',
        xerr=(scalar_indices['boostrap']['total_index'][:, k]),
        ecolor='k', capsize=3)


    for i,ax in enumerate(axs):
        ax.set_ylim([-1, 7.5])
        ax.grid(linestyle=':')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])

        ax.spines[['right', 'top']].set_visible(False)
        ax.text(0.025, 1., alphabet[i], transform=ax.transAxes,
            fontweight='bold', ha='left', va='top')

        ax.set_yticks(np.arange(8), config.theta_names)
    
    fig.text(0.5, 0.05, 'Sensitivity', ha='center', va='bottom')
    
    for ax in axs[1:]:
        ax.set_yticklabels([])
    fig.savefig(os.path.join(config.figures, 'sensitivity_indices.png'), dpi=400)
    fig.savefig(os.path.join(config.figures, 'sensitivity_indices.pdf'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)

    if args.recompute:
        compute_scalar_indices(train_config)
        compute_field_indices(train_config)

    
    plot_main_indices(train_config)
    
if __name__=='__main__':
    main()
