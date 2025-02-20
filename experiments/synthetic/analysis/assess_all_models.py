"""
Compute prediction error for different numbers of simulations and different
choices for the number of principal components given a common test set

usage: assess_all_models.py [-h] --npc NPC [NPC ...] --nsim NSIM [NSIM ...] [--recompute] [--test]
                            train_conf test_conf

"""

import os
import argparse

fs = 8
import matplotlib
matplotlib.rc('font', size=fs)

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction

from src import utils
from src.model import load_model

# Settings for consistent boxplots
flierprops = {'marker':'+', 'markersize':2, 'markeredgewidth':0.6}
ylims = [(0, 0.35), (0, 15), (0, 0.3)]

def plot_marginal_loss(path, n_sims, n_pcs, m_ref, p_ref):
    """
    Plot GP prediction RMSE, MAPE, prediction uncertainty
    for n_sims and n_pcs separately.

    Parameters
    ----------
    path : str
           Pattern for csv performance statistics files
    
    n_sims : array
             List of numbers of simulations
    
    n_pcs : array
             List of numbers of PCs
    
    m_ref : int
            Number of simulations for the reference emulator
    
    p_ref : int
            Number of PCs for the reference emulator
    
    Returns
    -------
    matplotlib.figure
    """
    fig, axs = plt.subplots(figsize=(6, 3.5), ncols=3, nrows=2)

    # 1 For number of PCs, read in CSV data and put into arrays
    ax1,ax2,ax3 = axs[0]
    RMSE = None
    MAPE = None
    CI = None
    # full_cis = np.zeros(len(n_pcs))
    labelsize = 6
    for i in range(len(n_pcs)):
        p = n_pcs[i]
        performance = np.loadtxt(path.format(m_ref, p), delimiter=',')
        if RMSE is None:
            n_train = performance.shape[0]
            RMSE = np.zeros((len(n_pcs), n_train))
            MAPE = np.zeros((len(n_pcs), n_train))
            CI = np.zeros((len(n_pcs), n_train))
            cov = np.zeros((len(n_pcs), n_train))
        RMSE[i,:] = performance[:,0]
        MAPE[i,:] = performance[:,1]
        lower = performance[:, 2]
        upper = performance[:, 3]
        CI[i,:] = upper - lower
        # full_cis[i] = performance[0, 4]

    metrics = (RMSE.T, 100*MAPE.T, CI.T)
    labels = ('RMSE', 'MAPE (%)', '95% prediction interval')
    alphabet = ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)')
    dys = (0.05, 5, 0.1)
    labelpads = [2, 2, 0]
    medianprops = {'color':'#000000'}
    boxprops = {'edgecolor':'none'}
    fc = [
        (0.272, 0.259, 0.539), 
        (0.420, 0.431, 0.812),
        (0.647, 0.318, 0.580),
    ]
    # Plot each metric
    for i in range(len(metrics)):
        ax = axs[0,i]
        ax.grid(linestyle=':', linewidth=0.5)
        boxprops['facecolor'] = fc[i]
        boxes = ax.boxplot(metrics[i], tick_labels=n_pcs, patch_artist=True,
            medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
            flierprops=flierprops, whiskerprops={'linewidth':0.65})
        ax.set_ylabel(labels[i], labelpad=labelpads[i])
        ax.set_ylim(ylims[i])
        ax.text(0.15, 1., alphabet[i], transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        xtlabels = np.array(n_pcs).astype(str)
        xtlabels[1::2] = ''
        ax.set_xticks(n_pcs, xtlabels)
        ax.set_xlabel('Number of PCs')
    
    # axs[0,2].plot(np.arange(1, len(n_pcs)+1), full_cis, 
    #     linestyle='', marker='.', color='#000000', markersize=4, zorder=10)

    # 2 Number of simulations
    ax1,ax2,ax3 = axs[1]
    RMSE = None
    MAPE = None
    CI = None
    # full_cis = np.zeros(len(n_sims))
    for i in range(len(n_sims)):
        m = n_sims[i]
        performance = np.loadtxt(path.format(m,p_ref), delimiter=',')
        if RMSE is None:
            n_train = performance.shape[0]
            RMSE = np.zeros((len(n_sims), n_train))
            MAPE = np.zeros((len(n_sims), n_train))
            CI = np.zeros((len(n_sims), n_train))
            cov = np.zeros((len(n_sims), n_train))
        RMSE[i,:] = performance[:,0]
        MAPE[i,:] = performance[:,1]
        lower = performance[:, 2]
        upper = performance[:, 3]
        CI[i,:] = upper - lower
        # full_cis[i] = performance[0, 4]

    metrics = (RMSE.T, 100*MAPE.T, CI.T)
    for i in range(len(metrics)):
        ax = axs[1,i]
        ax.grid(linestyle=':', linewidth=0.5)
        boxprops['facecolor'] = fc[i]
        boxes = ax.boxplot(metrics[i], tick_labels=n_sims, patch_artist=True,
            medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
            flierprops=flierprops, whiskerprops={'linewidth':0.65})
        ax.set_ylabel(labels[i], labelpad=labelpads[i])
        ax.set_ylim(ylims[i])
        ax.text(0.15, 1., alphabet[i+3], transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold')
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        xtlabels = np.array(n_sims).astype(str)
        xtlabels[0::2] = ''
        ax.set_xticks(np.arange(1,len(n_sims)+1), xtlabels)
        ax.set_xlabel('Number of simulations')
        
    # axs[1,2].plot(np.arange(1, len(n_sims)+1), full_cis, 
    #     linestyle='', marker='.', color='#000000', markersize=4, zorder=10)

    ax2 = axs[1,-1].twinx()
    ax2.spines['top'].set_visible(False)
    cputime = 0.42*n_sims
    ax2.plot(np.arange(1, len(n_sims)+1), cputime,
        color='#000000', marker='.', markersize=3, linewidth=0.65)
    ax2.set_ylabel('CPU-hours', rotation=-90, labelpad=8)
    ax2.tick_params(axis='both', labelsize=fs)
    fig.subplots_adjust(left=0.085, bottom=0.125, right=0.92, top=0.95, wspace=0.35, hspace=0.4)
    return fig


def plot_coverage(path, n_sims, n_pcs, m_ref, p_ref):
    """
    Plot GP coverage percentrage for n_sims and n_pcs separately.

    Parameters
    ----------
    path : str
           Pattern for csv performance statistics files
    
    n_sims : array
             List of numbers of simulations
    
    n_pcs : array
             List of numbers of PCs
    
    m_ref : int
            Number of simulations for the reference emulator
    
    p_ref : int
            Number of PCs for the reference emulator
    
    Returns
    -------
    matplotlib.figure
    """
    fig, axs = plt.subplots(figsize=(6, 2.5), ncols=2, nrows=1)

    coverage = np.zeros(len(n_pcs))
    labelsize = 6
    cov = None
    for i in range(len(n_pcs)):
        p = n_pcs[i]
        performance = np.loadtxt(path.format(m_ref, p), delimiter=',')
        if cov is None:
            n_train = performance.shape[0]
            cov = np.zeros((len(n_pcs), n_train))
        cov[i,:] = performance[:,-1]
        coverage[i] = np.mean(cov[i])

    medianprops = {'color':'#000000'}
    boxprops = {'edgecolor':'none'}
    boxprops['facecolor'] = 'gray'
    ax = axs[0]
    boxes = ax.boxplot(100*cov.T, tick_labels=n_pcs, patch_artist=True,
        medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
        flierprops=flierprops, whiskerprops={'linewidth':0.65})
    ax.plot(np.arange(1, len(n_pcs)+1), 100*coverage, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)
    ax.text(0.05, 0.9, 'a', transform=ax.transAxes,
        ha='left', va='bottom', fontweight='bold')
    xtlabels = np.array(n_pcs).astype(str)
    xtlabels[1::2] = ''
    ax.set_xticks(n_pcs, xtlabels)

    # 2 Number of simulations
    cov = None
    # full_cis = np.zeros(len(n_sims))
    coverage = np.zeros(len(n_sims))
    for i in range(len(n_sims)):
        m = n_sims[i]
        performance = np.loadtxt(path.format(m,p_ref), delimiter=',')
        if cov is None:
            n_train = performance.shape[0]
            cov = np.zeros((len(n_sims), n_train))
        cov[i,:] = performance[:,-1]
        coverage[i] = np.mean(cov[i])

    ax = axs[1]
    boxes = ax.boxplot(100*cov.T, tick_labels=n_sims, patch_artist=True,
        medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
        flierprops=flierprops, whiskerprops={'linewidth':0.65})
    ax.plot(np.arange(1, len(n_sims)+1), 100*coverage, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)
    ax.text(0.05, 0.9, 'b', transform=ax.transAxes,
        ha='left', va='bottom', fontweight='bold')
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both', labelsize=fs)
    xtlabels = np.array(n_sims).astype(str)
    xtlabels[0::2] = ''
    ax.set_xticks(np.arange(1,len(n_sims)+1), xtlabels)
    

    ax2 = axs[-1].twinx()
    ax2.spines['top'].set_visible(False)
    cputime = 0.42*n_sims
    ax2.plot(np.arange(1, len(n_sims)+1), cputime,
        color='#000000', marker='.', markersize=3, linewidth=0.65)
    ax2.set_ylabel('CPU-hours', rotation=-90, labelpad=8)
    ax2.tick_params(axis='both', labelsize=fs)

    for ax in axs:
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        ax.set_ylim([50, 108])
        yt = np.array([50, 60, 70, 80, 90, 95, 100])
        ax.set_yticks(yt)
        ax.grid(linestyle=':', linewidth=0.5)
    
    axs[0].set_ylabel('Coverage (%)')
    axs[0].set_xlabel('Number of PCs')
    axs[1].set_xlabel('Number of simulations')
    fig.subplots_adjust(left=0.085, bottom=0.15, right=0.92, top=0.975, wspace=0.2, hspace=0.3)
    return fig

def plot_joint_loss(path, n_sims, n_pcs):
    """
    Plot GP prediction RMSE, MAPE, for n_sims and n_pcs together

    Parameters
    ----------
    path : str
           Pattern for csv performance statistics files
    
    n_sims : array
             List of numbers of simulations
    
    n_pcs : array
             List of numbers of PCs
    
    Returns
    -------
    matplotlib.figure
    """
    # fig, axs = plt.subplots(figsize=(6, 3), ncols=2)
    fig, axs = plt.subplots(figsize=(3, 5), nrows=2)
    ax1,ax2 = axs
    RMSE = np.zeros((len(n_sims), len(n_pcs)))
    MAPE = np.zeros((len(n_sims), len(n_pcs)))
    RMSE_lower = np.zeros((len(n_sims), len(n_pcs)))
    RMSE_upper = np.zeros((len(n_sims), len(n_pcs)))
    MAPE_lower = np.zeros((len(n_sims), len(n_pcs)))
    MAPE_upper = np.zeros((len(n_sims), len(n_pcs)))
    AIC = np.zeros((len(n_sims), len(n_pcs)))
    BIC = np.zeros((len(n_sims), len(n_pcs)))
    for i in range(len(n_sims)):
        for j in range(len(n_pcs)):
            m = n_sims[i]
            p = n_pcs[j]
            performance = np.loadtxt(path.format(m,p), delimiter=',')

            rmse = performance[:,0]
            mape = performance[:,1]

            RMSE[i,j] = np.sqrt(np.nanmean(rmse**2))
            MAPE[i,j] = np.nanmean(mape)

            qq = 0.25
            RMSE_lower[i,j] = np.nanquantile(rmse, qq)
            RMSE_upper[i,j] = np.nanquantile(rmse, 1-qq)

            MAPE_lower[i,j] = np.nanquantile(mape, qq)
            MAPE_upper[i,j] = np.nanquantile(mape, 1-qq)

            mean_rss = RMSE[i,j]**2
            k = p*(8+1)
            AIC[i,j] = 2*k + m*np.log(mean_rss)
            BIC[i,j] = k*np.log(m) + m*np.log(mean_rss)


    metrics = (RMSE, 100*MAPE)
    lower = (RMSE_lower, 100*MAPE_lower)
    upper = (RMSE_upper, 100*MAPE_upper)
    labels = ('RMSE', 'MAPE (%)', 'Max error')
    dys = (0.05, 2.5)
    alphabet = ('(a)', '(b)')
    cbars = [0, 0, 0]
    colors = cmocean.cm.deep(np.linspace(0.15, 0.9, len(n_sims)))
    for i in range(len(metrics)):
        ax = axs[i]
        ax.grid(linestyle=':', linewidth=0.5)
        for j in range (len(n_sims)):
            ax.plot(n_pcs, metrics[i][j,:], color=colors[j], label=n_sims[j])

            ymax = np.max(upper[i])
            dy = dys[i]
            ub = dy*np.ceil(ymax/dy)
            ax.set_xlim([n_pcs[0]-0.5, n_pcs[-1]+0.5])
            ax.set_xticks(n_pcs)
            ylim = ax.get_ylim()
            ax.set_ylim([0, ub])

            span_rects = []
            mean_rects = []
            dx = 0.1
            dsim = 0.05
            sim_offset = (2-j)*dsim
            for l in range(len(n_pcs)):
                R = Rectangle((n_pcs[l] - dx/2 + sim_offset, lower[i][j,l]), 
                    width=dx, height=(upper[i][j,l] - lower[i][j,l]))
                span_rects.append(R)

                R2 = Rectangle((n_pcs[l] - 0.1, metrics[i][j,l]), 
                    width=0.2, height=ymax/100)
                mean_rects.append(R2)
                
            span_pcol = PatchCollection(span_rects, color=colors[j], edgecolor='none',
                alpha=0.5)
            ax.add_collection(span_pcol)

        ax.set_ylabel(labels[i])
        ax.text(0.05, 0.95, alphabet[i], transform=ax.transAxes,
            ha='left', va='top', fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
    
    # fig.text(0.5, 0.025, 'Number of principal components', ha='center')
    ax1.legend(loc='upper right', frameon=False, ncols=2)
    ax2.set_xlabel('Number of principal components')
    fig.subplots_adjust(left=0.175, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.15)
    return fig

def main(train_config, test_config, n_sims, n_pcs):
    """
    Compute and plot test error.

    Parameters
    ----------
    train_config : module
                   Training ensemble configuration
    
    test_config: module
                 Test ensemble configuration
                
    n_sims : array
             List of numbers of simulations
    
    n_pcs : array
             List of numbers of PCs
    """
    path = os.path.join(train_config.data_dir, 'architecture/performance_n{:03d}_p{:02d}.csv')
    print('path:', path)
    fig1 = plot_joint_loss(path, n_sims, n_pcs)
    if not os.path.exists(train_config.figures):
        os.makedirs(train_config.figures)
    fig1.savefig(os.path.join(train_config.figures, 'main/fig04.png'),
        dpi=400)
    fig1.savefig(os.path.join(train_config.figures, 'main/fig04.pdf'))
    
    fig3 = plot_marginal_loss(path, np.array(n_sims), np.array(n_pcs), train_config.m, train_config.p)
    fig3.savefig(os.path.join(train_config.figures, 'main/fig05.png'), dpi=400)
    fig3.savefig(os.path.join(train_config.figures, 'main/fig05.pdf'))

    sfig = plot_coverage(path, np.array(n_sims), np.array(n_pcs), train_config.m, train_config.p)
    sfig.savefig(os.path.join(train_config.figures, 'appendix/B04.png'), dpi=400)
    sfig.savefig(os.path.join(train_config.figures, 'appendix/B04.pdf'))
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_conf')
    parser.add_argument('test_conf')
    parser.add_argument('--npc', nargs='+', type=int, required=True)
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    args = parser.parse_args()
    train_config = utils.import_config(args.train_conf)
    test_config = utils.import_config(args.test_conf)
    main(train_config, test_config, n_sims=args.nsim, n_pcs=args.npc)
    
