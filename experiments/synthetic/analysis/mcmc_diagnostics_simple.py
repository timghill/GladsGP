"""
Assess MCMC sampling chain convergence, posterior sampling

usage: mcmc_diagnostics_simple.py [-h] train_config
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from sepia import SepiaParam

import cmocean

from src import utils
from src import model as mtools

def mcmc_trace(train_config, recompute=True):
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
        ax.set_xticks([0, 128, 256, 256+128, 512])
        ax.grid(linestyle=':')

    axs[-1].set_xlabel('Iteration')
    fig.subplots_adjust(bottom=0.05, top=0.975, left=0.1, right=0.95)
    fig.savefig('figures/mcmc_trace.png', dpi=400)

    # 2. CONVERGENCE DIAGNOSTICS
    # for pc in range(train_config.p):
    accept_rate = np.zeros((8, train_config.p))
    Rhat = np.zeros((8, train_config.p))
    Neff = np.zeros((8, train_config.p))
    nburn = 256
    for para in range(8):
        for pc in range(train_config.p):
            b = betas[:, para, pc]
            b1 = b[:nburn]
            b2 = b[nburn:]

            acceptance = np.ones((n_samples-nburn-1))
            acceptance[b2[1:]==b2[:-1]] = 0
            accept_rate[para, pc] = acceptance.sum()/len(acceptance)

    # 3. GELMAN CONVERGENCE STATISTIC R^HAT
    chains = []
    n_repeats = 4
    n_samples = 256
    n_burn = 256
    d = 8
    if recompute:
        data,model = mtools.load_model(train_config, train_config.m, train_config.p)
        model.clear_samples()

        beta_start = model.params.betaU.val.copy()
        rng = np.random.default_rng()
        for repeat in range(n_repeats):
            print('repeat {}/{}'.format(repeat+1, n_repeats))
            model.clear_samples()

            beta_new = beta_start + 0.1*rng.standard_normal(beta_start.shape)
            beta_new[beta_new<1e-2] = 1e-2
            model.params.betaU.set_val(beta_new)

            model.do_mcmc(n_samples + n_burn, no_init=False)
            samples = model.get_samples(n_samples, nburn=n_burn)
            chains.append(samples)
        
        betas = np.array([chain['betaU'] for chain in chains])
        betas = betas.reshape((betas.shape[0], betas.shape[1], d+1, train_config.p))[:, :, 1:, :]
        print('betas:', betas.shape)
        
        half_length = int(n_samples/2)
        betas_split = np.zeros((betas.shape[0]*2, half_length, betas.shape[2], betas.shape[3]))
        betas_split[:betas.shape[0]] = betas[:, :half_length]
        betas_split[betas.shape[0]:] = betas[:, half_length:]
        print('betas_split:', betas_split.shape)

        for para in range(8):
            for pc in range(train_config.p):
                phi = betas_split[:, :, para, pc]

                # between-chain variance
                n = half_length
                m = 2*n_repeats
                phibar_j = np.mean(phi, axis=1)
                phibar = np.mean(phibar_j)
                B = n/(m-1)*np.sum((phibar_j-phibar)**2)
                
                sj2 = 1/(n-1)*np.sum((phi-phibar_j[:, None])**2, axis=1)
                W = np.mean(sj2)

                varhat = (n-1)*W/n + B/n
                Rhat[para, pc] = np.sqrt(varhat/W)

                rhot = np.zeros(n)
                for ti in range(1,n):
                    phiss = phi[:, ti:]
                    philag = phi[:, :-ti]
                    Vt = np.sum((phiss-philag)**2)/m/(n-ti)
                    rhot[ti] = 1 - Vt/2/varhat
                T = 1
                for ti in range(1, n-1):
                    if ti%2==1:
                        if (rhot[ti]+rhot[ti+1])<0:
                            T = ti
                            break
                else:
                    T = n-1

                rhotsum = np.sum(rhot[:T])
                n_eff = m*n/(1 + 2*rhotsum)
                Neff[para, pc] = n_eff
        
        np.savetxt('data/Rhat.txt', Rhat, delimiter=',', fmt='%.3e')
        np.savetxt('data/Neff.txt', Neff, delimiter=',', fmt='%.3e')
        
    Rhat = np.loadtxt('data/Rhat.txt', delimiter=',')
    Neff = np.loadtxt('data/Neff.txt', delimiter=',')

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
    print('Neff:')
    print(Neff)
    print('Mean:', Neff.mean())
    print('Median:', np.median(Neff))
    print('Min/max:', Neff.min(), Neff.max())
    print('Median for each parameter:', np.median(Neff, axis=1))
    print('Median for each PC:', np.median(Neff, axis=0))

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3, height_ratios=(5, 100), width_ratios=(10, 100, 10),
        left=0.1, bottom=0.1, right=0.95, top=0.9,
        hspace=0.05)
    ax = fig.add_subplot(gs[1,:])
    cax = fig.add_subplot(gs[0,1])
    thetas = np.arange(8)
    pcs = np.arange(train_config.p)
    [pp, tt] = np.meshgrid(pcs, thetas)

    pcolor = ax.pcolormesh(pp, tt, Rhat, vmin=1, vmax=1.2, cmap=cmocean.cm.balance)
    ax.set_yticks(np.arange(8), train_config.theta_names)
    xticks = ['PC{}'.format(ii) for ii in range(1, train_config.p+1)]
    ax.set_xticks(np.arange(train_config.p), xticks)
    ax.invert_yaxis()

    cbar = fig.colorbar(pcolor, cax=cax, orientation='horizontal', extend='max')
    cbar.set_label(r'$\hat R$')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.axvline(1.10, color='k')
    cbar.set_ticks([1, 1.05, 1.1, 1.15, 1.2])

    for para in range(8):
        for pc in range(train_config.p):
            Ri = Rhat[para,pc]
            tcol = 'k' if (np.abs(Ri-1.1)<=0.05) else 'w'
            ax.text(pc, para, '{:.2f}'.format(Ri), color=tcol, ha='center', va='center')
    
    fig.savefig('figures/Rhat.png', dpi=400)



    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3, height_ratios=(5, 100), width_ratios=(10, 100, 10),
        left=0.1, bottom=0.1, right=0.95, top=0.9,
        hspace=0.05)
    ax = fig.add_subplot(gs[1,:])
    cax = fig.add_subplot(gs[0,1])
    thetas = np.arange(8)
    pcs = np.arange(train_config.p)
    [pp, tt] = np.meshgrid(pcs, thetas)

    pcolor = ax.pcolormesh(pp, tt, Neff, vmin=0, vmax=20*n_repeats, cmap=cmocean.cm.balance_r)
    ax.set_yticks(np.arange(8), train_config.theta_names)
    xticks = ['PC{}'.format(ii) for ii in range(1, train_config.p+1)]
    ax.set_xticks(np.arange(train_config.p), xticks)
    ax.invert_yaxis()

    cbar = fig.colorbar(pcolor, cax=cax, orientation='horizontal', extend='max')
    cbar.set_label(r'$N_{\rm{eff}}$')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.axvline(10*n_repeats, color='k')
    # cbar.set_ticks([1, 1.05, 1.1, 1.15, 1.2])

    for para in range(8):
        for pc in range(train_config.p):
            Ni = Neff[para,pc]
            tcol = 'k' if (np.abs(Ni-10*n_repeats)<=5*n_repeats) else 'w'
            ax.text(pc, para, '{:.2f}'.format(Ni), color=tcol, ha='center', va='center')
    
    fig.savefig('figures/Neff.png', dpi=400)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('--recompute', required=False, action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)
    mcmc_trace(train_config, recompute=args.recompute)

if __name__=='__main__':
    main()