"""
Assess MCMC sampling chain convergence, posterior sampling

usage: mcmc_diagnostics_simple.py [-h] train_config
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

from sepia import SepiaParam

from src import utils
from src import model as mtools

def mcmc_trace(train_config):
    """
    MCMC diagnostic plots
        1. Trace plots
        2. Repeat chain from random intialization point,
           explore differences in distribution with
           a. qq-plot
           b. histogram comparisons
        3. Posterior sample. Repeat sampling posterior, explore
           differences in distribution with same 2 methods
    """
    # data,model = mtools.load_model(train_config, train_config.m, train_config.p)

    # samples = model.get_samples()
    model = np.load('data/models/synthetic_n{}_p{:02d}.pkl'.format(
        train_config.m, train_config.p), allow_pickle=True)
    samples = model['samples']
    betas = np.array(samples['betaU']).squeeze()
    betas = betas[:, 1:train_config.p+1, :]
    lamUz = np.array(samples['lamUz']).squeeze()
    lamWOs = np.array(samples['lamWOs']).squeeze()
    n_samples = len(lamUz)
    xx = np.arange(n_samples)
    print(betas.shape)

    fig,axs = plt.subplots(figsize=(6, 9), nrows=train_config.p+2,
        sharex=True)

    axs[0].plot(xx, lamUz)
    axs[0].set_ylabel(r'$\lambda$')
    for i in range(train_config.p):
        axs[1+i].plot(xx, betas[:, i, :])
        axs[i+1].set_ylabel(r'$\beta_{}$'.format(i+1))
    
    axs[-1].plot(xx, lamWOs)
    axs[-1].set_ylabel(r'$\lambda_{\rm{sim}}$')

    for i,ax in enumerate(axs):
        ax.set_xlim([0, n_samples])
        ax.grid(linestyle=':')

    axs[-1].set_xlabel('Iteration')
    fig.subplots_adjust(bottom=0.05, top=0.975, left=0.1, right=0.95)
    fig.savefig('figures/mcmc_trace.png', dpi=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)
    mcmc_trace(train_config)

if __name__=='__main__':
    main()