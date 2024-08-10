"""
Assess MCMC sampling chain convergence, posterior sampling

usage: mcmc_diagnostics.py [-h] train_config
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

from sepia import SepiaParam

from src import utils
from src import model as mtools

def mcmc_diagnostics(train_config):
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
    data,model = mtools.load_model(train_config, train_config.m, train_config.p)
    model.clear_samples()
    chains = []
    n_repeats = 2
    n_samples = 512
    n_posterior = 128
    n_burn = 128
    d = 8
    sim_data = data.sim_data

    w = np.dot(np.linalg.pinv(sim_data.K).T, sim_data.y_std.T).T
    y_sim_std_hat = np.dot(w, sim_data.K)
    pc_resid = sim_data.y_std - y_sim_std_hat
    pc_var = np.var(pc_resid)
    pc_prec = 1/pc_var
    print('pc variance:', pc_var)
    print('pc precision:', pc_prec)
    
    model.print_value_info()
    model.print_mcmc_info()


    for repeat in range(n_repeats):
        model.clear_samples()
        model.do_mcmc(n_samples + n_burn, no_init=False)
        samples = model.get_samples(n_samples, nburn=n_burn)
        chains.append(samples)
    
    betas = np.array([chain['betaU'] for chain in chains])
    betas = betas.reshape((betas.shape[0], betas.shape[1], d+1, train_config.p))[:, :, 1:, :]
    
    lambdas_GP = np.array([chain['lamUz'] for chain in chains])
    lambdas_PC = np.array([chain['lamWOs'] for chain in chains])

    # 1. Trace plot
    fig,axs = plt.subplots(figsize=(6, 10), nrows=d+2, sharex=True)
    axs[0].plot(np.arange(n_samples), lambdas_GP[0,:,:])
    for j in range(train_config.p):
        col = (j+1)/(train_config.p+2)*np.ones(3)
        axs[0].plot(np.arange(n_samples), lambdas_GP[0, :, j], color=col)
    axs[0].set_ylabel(r'$\lambda^{Uz}$')
    for i in range(d):
        for j in range(train_config.p):
            col = (j+1)/(train_config.p+2)*np.ones(3)
            axs[i+1].plot(np.arange(n_samples), betas[0, :, i, j], color=col)
        axs[i+1].set_ylabel(r'$\beta_{}$'.format(i+1))
    
    axs[-1].plot(np.arange(n_samples), lambdas_PC[0,:,0])
    axs[-1].set_ylabel(r'$\lambda^{WOs}$')
    fig.subplots_adjust(bottom=0.05, top=0.98, left=0.15, right=0.95)
    for ax in axs.flat:
        ax.grid()
        ax.set_xlim([0, n_samples])
    
    axs[-1].set_xlabel('MCMC samples')
    fig.savefig('figures/mcmc_trace.png', dpi=600)
    
    # 2a quantile plot
    fig, ax = plt.subplots()
    qntls = np.linspace(0, 1, int(n_samples/2))
    for p in range(train_config.p):
        for i in range(d):
            q1 = np.quantile(betas[0, :, i, p], qntls)
            q2 = np.quantile(betas[1, :, i, p], qntls)
            ax.scatter(q1, q2, s=5)
            ax.set_aspect('equal')
            ax.grid(linestyle=':')
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    fig.savefig('figures/mcmc_sample_qq.png', dpi=600)

    #2b histogram plot
    fig, axs = plt.subplots(nrows=d, ncols=train_config.p)
    for p in range(train_config.p):
        for i in range(d):
            # for repeats in range(2):
            counts0, bins0,_ = axs[i,p].hist(betas[0, :, i, p], 
                alpha=0.5, bins=16)
            counts1, bins1,_ = axs[i,p].hist(betas[1, :, i, p],
                alpha=0.5, bins=bins0)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    fig.savefig('figures/mcmc_sample_hist.png', dpi=600)

    ## Posterior sample

    # 3a quantile plot
    fig, ax = plt.subplots()
    qntls = np.linspace(0, 1, int(n_posterior/2))
    post_factor = int(n_samples/n_posterior)
    betas_post = model.get_samples(n_posterior, nburn=n_burn)['betaU']
    print('betas_post.shape:', betas_post.shape)
    betas_post = betas_post.reshape((n_posterior, d+1, train_config.p))[:, 1:, :]
    print('betas_post.shape:', betas_post.shape)
    for p in range(train_config.p):
        for i in range(d):
            q1 = np.quantile(betas[1, :, i, p], qntls)
            q2 = np.quantile(betas_post[:, i, p], qntls)
            ax.scatter(q1, q2, s=5)
            ax.set_aspect('equal')
            ax.grid(linestyle=':')
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    fig.savefig('figures/mcmc_posterior_qq.png', dpi=600)

    # 3b histogram plot
    fig, axs = plt.subplots(nrows=d, ncols=train_config.p)
    for p in range(train_config.p):
        for i in range(d):
            # for repeats in range(2):
            counts0, bins0,_ = axs[i,p].hist(betas[1, :, i, p], 
                alpha=0.5, bins=16)
            counts1, bins1,_ = axs[i,p].hist(betas_post[:, i, p],
                alpha=0.5, bins=bins0[::post_factor])
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    fig.savefig('figures/mcmc_posterior_hist.png', dpi=600)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)
    mcmc_diagnostics(train_config)

if __name__=='__main__':
    main()