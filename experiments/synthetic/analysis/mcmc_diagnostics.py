"""
Assess MCMC sampling chain convergence, posterior sampling

usage: mcmc_diagnostics.py [-h] [--recompute -r] train_config
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from sepia import SepiaParam

import cmocean

from src import utils
from src import model as mtools

def mcmc_diagnostics(train_config, recompute=True):
    """
    MCMC diagnostic plots
        1. Trace plots
        2. Acceptance rate
    """

    # 1. TRACE PLOTS
    model = np.load('data/models/synthetic_n{}_p{:02d}.pkl'.format(
        train_config.m, train_config.p), allow_pickle=True)
    samples = model['samples']
    betas = np.array(samples['betaU']).squeeze()
    print('original:', betas.shape)
    betas = betas[:, 1:9, :]
    lamUz = np.array(samples['lamUz']).squeeze()
    lamWOs = np.array(samples['lamWOs']).squeeze()
    n_samples = len(lamUz)
    xx = np.arange(n_samples)
    print(betas.shape)

    fig,axs = plt.subplots(figsize=(6, 9), nrows=train_config.p+2,
        sharex=True)

    axs[0].plot(xx, lamUz)
    axs[0].set_ylabel(r'$\lambda$')
    for i in range(8):
        axs[1+i].plot(xx, betas[:, i, :])
        axs[i+1].set_ylabel(r'$\beta_{}$'.format(i+1))
    
    axs[-1].plot(xx, lamWOs)
    axs[-1].set_ylabel(r'$\lambda_{\rm{sim}}$')

    for i,ax in enumerate(axs):
        ax.set_xlim([0, n_samples])
        # ax.set_xticks([0, 128, 256, 256+128, 512])
        ax.grid(linestyle=':')

    axs[-1].set_xlabel('Iteration')
    fig.subplots_adjust(bottom=0.05, top=0.975, left=0.1, right=0.95)
    fig.savefig('figures/appendix/B02.png', dpi=400)
    fig.savefig('figures/appendix/B02.pdf')

    # 2. CONVERGENCE DIAGNOSTICS
    chains = []
    n_repeats = 4
    n_samples = 2500
    n_burn = 2500
    d = 8
    if recompute:
        data,model = mtools.load_model(train_config, train_config.m, train_config.p)
        model.clear_samples()

        model.print_mcmc_info()
        # for i,p in enumerate(model.params.mcmcList):
        #     if p.name=='betaU':
        #         # print('betaU:', p, p.name)
        #         step = p.mcmc.stepParam.copy()
        #         print('min:', np.min(step))
        #         # Cap step size between [1e-2, 1]
        #         step[step<1e-2] = 1e-2
        #         # step[step>1] = 1

        #         # Tune k_C step size
        #         # step[2, 2] = 3
        #         # step[2, 4] = 0.1
        #         p.mcmc.stepParam = step
        
        beta_start = model.params.betaU.val.copy()

        rng = np.random.default_rng()
        for repeat in range(n_repeats):
            print('repeat {}/{}'.format(repeat+1, n_repeats))
            model.clear_samples()

            beta_new = beta_start + 0.1*rng.standard_normal(beta_start.shape)
            # Keep starting values away from 0
            beta_new[beta_new<1e-1] = 1e-1
            model.params.betaU.set_val(beta_new)

            model.do_mcmc(n_samples + n_burn, no_init=False)
            samples = model.get_samples(n_samples, nburn=n_burn)
            chains.append(samples)
        
        betas = np.array([chain['betaU'] for chain in chains])
        betas = betas.reshape((betas.shape[0], betas.shape[1], d+1, train_config.p))[:, :, 1:, :]
        print('betas:', betas.shape)

        np.save('data/mcmc_chains.npy', betas)
    
    betas = np.load('data/mcmc_chains.npy')
    phi = betas.reshape((2*n_repeats, -1, *betas.shape[2:]))
    Rhat, ESS = utils.mcmc_rhat_ess(betas, burn=0)
    accept_rate = utils.mcmc_accept_rate(betas, burn=0)

    print()
    print('Using m={} chains'.format(n_repeats))
    print('Using n={} samples per full chain'.format(n_samples))
    print('Warm-up period of {} samples'.format(n_burn))

    print()
    print('Acceptance:')
    print(accept_rate)
    print('Mean:', accept_rate.mean())
    print('Median:', np.median(accept_rate))
    print('Min/max:', accept_rate.min(), accept_rate.max())
    print('Median for each parameter:', np.median(accept_rate, axis=1))
    print('Median for each PC:', np.median(accept_rate, axis=0))

    print()
    print('Rhat:')
    print(Rhat)
    print('Mean:', Rhat.mean())
    print('Median:', np.median(Rhat))
    print('Min/max:', Rhat.min(), Rhat.max())
    print('Median for each parameter:', np.median(Rhat, axis=1))
    print('Median for each PC:', np.median(Rhat, axis=0))

    print()
    print('ESS:')
    print(ESS)
    print('Mean:', ESS.mean())
    print('Median:', np.median(ESS))
    print('Min/max:', ESS.min(), ESS.max())
    print('Median for each parameter:', np.median(ESS, axis=1))
    print('Median for each PC:', np.median(ESS, axis=0))

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3, height_ratios=(5, 100), width_ratios=(10, 100, 10),
        left=0.1, bottom=0.1, right=0.95, top=0.9,
        hspace=0.05)
    ax = fig.add_subplot(gs[1,:])
    cax = fig.add_subplot(gs[0,1])
    thetas = np.arange(8)
    pcs = np.arange(train_config.p)
    [pp, tt] = np.meshgrid(pcs, thetas)

    pcolor = ax.pcolormesh(pp, tt, accept_rate, vmin=0, vmax=1, cmap=cmocean.cm.delta)
    ax.set_yticks(np.arange(8), train_config.theta_names)
    xticks = ['PC{}'.format(ii) for ii in range(1, train_config.p+1)]
    ax.set_xticks(np.arange(train_config.p), xticks)
    ax.invert_yaxis()

    cbar = fig.colorbar(pcolor, cax=cax, orientation='horizontal')
    cbar.set_label('Acceptance rate')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    # cax.axvline(0.5, color='k')

    for para in range(8):
        for pc in range(train_config.p):
            ri = accept_rate[para,pc]
            tcol = 'k' if (np.abs(ri-0.5)<=0.25) else 'w'
            ax.text(pc, para, '{:.2f}'.format(ri), color=tcol, ha='center', va='center')
    
    fig.savefig('figures/scratch/accept.png', dpi=400)

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3, height_ratios=(5, 100), width_ratios=(10, 100, 10),
        left=0.1, bottom=0.1, right=0.95, top=0.9,
        hspace=0.05)
    ax = fig.add_subplot(gs[1,:])
    cax = fig.add_subplot(gs[0,1])
    thetas = np.arange(8)
    pcs = np.arange(train_config.p)
    [pp, tt] = np.meshgrid(pcs, thetas)

    clist1 = cmocean.cm.balance(np.linspace(0.25, 0.5, 128))
    clist2 = cmocean.cm.balance(np.linspace(0.5, 1., 128))
    clist = np.array([clist1, clist2]).reshape((256, -1))
    cmap = LinearSegmentedColormap.from_list('custom', clist)
    pcolor = ax.pcolormesh(pp, tt, Rhat, vmin=1., vmax=1.1, cmap=cmap)
    ax.set_yticks(np.arange(8), train_config.theta_names)
    xticks = ['PC{}'.format(ii) for ii in range(1, train_config.p+1)]
    ax.set_xticks(np.arange(train_config.p), xticks)
    ax.invert_yaxis()

    cbar = fig.colorbar(pcolor, cax=cax, orientation='horizontal')
    cbar.set_label(r'$\hat R$')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    # cax.axvline(1.05, color='k')
    # cbar.set_ticks([1, 1.05, 1.1, 1.15, 1.2])
    cbar.set_ticks([1, 1.025, 1.05, 1.075, 1.1])

    for para in range(8):
        for pc in range(train_config.p):
            Ri = Rhat[para,pc]
            tcol = 'k' if Ri<=1.08 else 'w'
            ax.text(pc, para, '{:.2f}'.format(Ri), color=tcol, ha='center', va='center')
    
    fig.savefig('figures/scratch/Rhat.png', dpi=400)



    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3, height_ratios=(5, 100), width_ratios=(10, 100, 10),
        left=0.1, bottom=0.1, right=0.95, top=0.9,
        hspace=0.05)
    ax = fig.add_subplot(gs[1,:])
    cax = fig.add_subplot(gs[0,1])
    thetas = np.arange(8)
    pcs = np.arange(train_config.p)
    [pp, tt] = np.meshgrid(pcs, thetas)

    ratio = 8
    n1 = int(256/ratio)
    clist = np.flipud(np.concatenate([cmocean.cm.balance(np.linspace(0.25, 0.5, 256-n1)),
        cmocean.cm.balance(np.linspace(0.50, 1.0, n1))]))
    cmap = LinearSegmentedColormap.from_list('', clist)
    pcolor = ax.pcolormesh(pp, tt, ESS/n_repeats, vmin=0, vmax=ratio*10, cmap=cmap)
    ax.set_yticks(np.arange(8), train_config.theta_names)
    xticks = ['PC{}'.format(ii) for ii in range(1, train_config.p+1)]
    ax.set_xticks(np.arange(train_config.p), xticks)
    ax.invert_yaxis()

    cbar = fig.colorbar(pcolor, cax=cax, orientation='horizontal', extend='max')
    cbar.set_label('ESS per chain')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    # cax.axvline(10, color='k')
    # cbar.set_ticks([0, 5*n_repeats, 10*n_repeats, 15*n_repeats, 20*n_repeats])
    # cbar.set_ticks([1, 1.05, 1.1, 1.15, 1.2])

    for para in range(8):
        for pc in range(train_config.p):
            Ni = ESS[para,pc]/n_repeats
            tcol = 'k' if ((Ni)>=5) else 'w'
            ax.text(pc, para, '{:.2f}'.format(Ni), color=tcol, ha='center', va='center')
    
    fig.savefig('figures/scratch/ESS.png', dpi=400)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('--recompute', required=False, action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)
    mcmc_diagnostics(train_config, recompute=args.recompute)

if __name__=='__main__':
    main()