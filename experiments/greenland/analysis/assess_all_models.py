"""
Compute prediction error for different numbers of simulations and different
choices for the number of principal components given a common test set
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

flierprops = {'marker':'+', 'markersize':2, 'markeredgewidth':0.6}

def plot_marginal_loss(path, n_sims, n_pcs, m_ref, p_ref):
    """
    Plot GP prediction RMSE, std residuals for n_sims and n_pcs
    """
    fig, axs = plt.subplots(figsize=(6, 4), ncols=4, nrows=2)

    # 1 For number of PCs
    ax1,ax2,ax3,ax4 = axs[0]
    RMSE = None
    MAPE = None
    CI = None
    # full_cis = np.zeros(len(n_pcs))
    coverage = np.zeros(len(n_pcs))
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
        cov[i,:] = performance[:,4]
        coverage[i] = np.mean(cov[i])

    metrics = (RMSE.T, 100*MAPE.T, CI.T, 100*cov.T)
    labels = ('RMSE', 'MAPE (%)', '95% prediction interval', 'Coverage (%)')
    alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    dys = (0.05, 2, 0.1, 5)
    labelpads = [2, -2, 0, -4]
    # cbars = [0, 0, 0]
    # colors = cmocean.cm.amp(np.linspace(0.25, 1, len(n_sims)))
    medianprops = {'color':'#000000'}
    boxprops = {'edgecolor':'none'}
    # fc = ['#356575', '#6295A20.024, 0.', '#80B9AD', '#B3E2A7']
    fc = [(0.272, 0.259, 0.539), (0.420, 0.431, 0.812),
        (0.647, 0.318, 0.580), (0.858, 0.478, 0.791)]
    for i in range(len(metrics)):
        ax = axs[0,i]
        ax.grid(linestyle=':', linewidth=0.5)
        boxprops['facecolor'] = fc[i]
        boxes = ax.boxplot(metrics[i], tick_labels=n_pcs, patch_artist=True,
            medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
            flierprops=flierprops, whiskerprops={'linewidth':0.65})
        ax.set_ylabel(labels[i], labelpad=labelpads[i])
        ymax = np.max(metrics[i])
        dy = dys[i]
        upper = dy*np.ceil(ymax/dy)
        ylim = ax.get_ylim()
        ax.set_ylim([0, upper])
        ax.text(0.15, 0.9, alphabet[i], transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        xtlabels = np.array(n_pcs).astype(str)
        xtlabels[1::2] = ''
        ax.set_xticks(n_pcs, xtlabels)
    
    # axs[0,2].plot(np.arange(1, len(n_pcs)+1), full_cis, 
    #     linestyle='', marker='.', color='#000000', markersize=4, zorder=10)

    axs[0,-1].plot(np.arange(1, len(n_pcs)+1), 100*coverage, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)
    # axs[0,-1].set_ylim([0, 100])
    
    fig.text(0.5, 0.52, 'Number of PCs', ha='center')

    # 2 Number of simulations
    ax1,ax2,ax3,ax4 = axs[1]
    RMSE = None
    MAPE = None
    CI = None
    # full_cis = np.zeros(len(n_sims))
    coverage = np.zeros(len(n_sims))
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
        cov[i,:] = performance[:,4]
        coverage[i] = np.mean(cov[i])

    metrics = (RMSE.T, 100*MAPE.T, CI.T, 100*cov.T)
    # labels = ('RMSE', 'MAPE (%)', '95% prediction interval width', 'Coverage (%)')
    # medianprops = {'color':'#000000'}
    # boxprops = {'edgecolor':'none'}
    # # fc = ['#8a0A0A', '#BF3131', '#DA6262']
    # # fc = cmocean.cm.balance([0.85, 0.75, 0.15, 0.25])
    # fc = ['#356575', '#6295A2', '#80B9AD', '#B3E2A7']
    for i in range(len(metrics)):
        ax = axs[1,i]
        ax.grid(linestyle=':', linewidth=0.5)
        boxprops['facecolor'] = fc[i]
        boxes = ax.boxplot(metrics[i], tick_labels=n_sims, patch_artist=True,
            medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
            flierprops=flierprops, whiskerprops={'linewidth':0.65})
        ax.set_ylabel(labels[i], labelpad=labelpads[i])
        ymax = np.max(metrics[i])
        dy = dys[i]
        upper = dy*np.ceil(ymax/dy)
        ylim = ax.get_ylim()
        ax.set_ylim([0, upper])
        ax.text(0.15, 0.9, alphabet[i+4], transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold')
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        xtlabels = np.array(n_sims).astype(str)
        xtlabels[0::2] = ''
        ax.set_xticks(np.arange(1,len(n_sims)+1), xtlabels)
        
    # axs[1,2].plot(np.arange(1, len(n_sims)+1), full_cis, 
        # linestyle='', marker='.', color='#000000', markersize=4, zorder=10)
    axs[1,-1].plot(np.arange(1, len(n_sims)+1), 100*coverage, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)

    ax2 = axs[1,-1].twinx()
    ax2.spines['top'].set_visible(False)
    cputime = 0.5*n_sims
    ax2.plot(np.arange(1, len(n_sims)+1), cputime,
        color='#000000', marker='.', markersize=3, linewidth=0.65)
    ax2.set_ylabel('CPU-hours', rotation=-90, labelpad=8)
    ax2.tick_params(axis='both', labelsize=fs)

    axs[0,-1].set_ylim([50, 108])
    axs[1,-1].set_ylim([50, 108])
    yt = np.array([50, 60, 70, 80, 90, 95, 100])
    axs[0,-1].set_yticks(yt)
    axs[1,-1].set_yticks(yt)
    # axs[0, -1].axhline(95, color='k', linewidth=5/8, zorder=1, linestyle='dashed')
    # axs[1, -1].axhline(95, color='k', linewidth=5/8, zorder=2, linestyle='dashed')
    
    fig.text(0.5, 0.025, 'Number of Simulations', ha='center')
    fig.subplots_adjust(left=0.085, bottom=0.1, right=0.92, top=0.975, wspace=0.5, hspace=0.3)
    return fig

def plot_joint_loss(path, n_sims, n_pcs, linestyle='solid'):
    """
    Plot GP prediction RMSE, std residuals for n_sims and n_pcs
    """
    fig, axs = plt.subplots(figsize=(6, 3), ncols=2)
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
    alphabet = ('a', 'b')
    cbars = [0, 0, 0]
    colors = cmocean.cm.deep(np.linspace(0.15, 0.9, len(n_sims)))
    for j in range(len(n_sims)):
        lines = []
        collections = []
        for i in range(len(metrics)):
            ax = axs[i]
            ax.grid(linestyle=':', linewidth=0.5)
            if linestyle=='solid':
                line, = ax.plot(n_pcs, metrics[i][j,:], color=colors[j], label=n_sims[j])
            lines.append(line)

            ymax = np.max(upper[i])
            dy = dys[i]
            ub = dy*np.ceil(ymax/dy)
            ax.set_xlim([n_pcs[0]-0.5, n_pcs[-1]+0.5])
            ylim = ax.get_ylim()
            ax.set_ylim([0, ub])
            ax.set_xticks(n_pcs)

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
            collections.append(span_pcol)

            if linestyle=='bar':
                mean_pcol = PatchCollection(mean_rects, color=colors[j], edgecolor='none',
                        alpha=1.)
                ax.add_collection(mean_pcol)

            ax.set_ylabel(labels[i])
            ax.text(0.05, 0.95, alphabet[i], transform=ax.transAxes,
                ha='left', va='top', fontweight='bold')\
        
            fig.text(0.5, 0.025, 'Number of PCs', ha='center')
            leg = ax1.legend(bbox_to_anchor=(0, 0.98, 1, 0.2),
                ncols=len(n_sims), loc='lower left',frameon=False)
        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.975, top=0.9, wspace=0.2)
        # fig.savefig('figures/nsim_npc_model_selection_{:02d}.png'.format(j+1), dpi=600)

        if j==0:
            for line in lines:
                line.set_alpha(0.)
            for pcol in collections:
                pcol.set_alpha(0.)
            leg.set_visible(False)
            # fig.savefig('figures/nsim_npc_model_selection_blank.png'.format(j+1), dpi=600)

            for line in lines:
                line.set_alpha(1.)
            for pcol in collections:
                pcol.set_alpha(0.5)
            leg.set_visible(True)
    return fig

def main(train_config, test_config, n_sims, n_pcs):

    path = os.path.join(train_config.data_dir, 'architecture/performance_n{:03d}_p{:02d}.csv')
    print('path:', path)
    fig1 = plot_joint_loss(path, n_sims, n_pcs)
    if not os.path.exists(train_config.figures):
        os.makedirs(train_config.figures)
    fig1.savefig(os.path.join(train_config.figures, 'nsim_npcs_error.png'),
        dpi=400)
    fig1.savefig(os.path.join(train_config.figures, 'nsim_npcs_error.pdf'))
    
    fig3 = plot_marginal_loss(path, np.array(n_sims), np.array(n_pcs), train_config.m, train_config.p)
    fig3.savefig(os.path.join(train_config.figures, 'nsim_boxplot.png'), dpi=400)
    fig3.savefig(os.path.join(train_config.figures, 'nsim_boxplot.pdf'))
        

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
    
