
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

def plot_all_indices(config):
    """
    Plot general and PC sensitivity indices
    """
    data_dir = 'data'
    sensitivity_dir = os.path.join(data_dir, 'sensitivity/')
    indices = np.load(
        os.path.join(sensitivity_dir, 'sobol_indices.pkl'),
        allow_pickle=True)
    alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    pca_fpattern = os.path.join(data_dir, 'models/pca_{}_n{:03d}_S.npy')
    # S = np.load(pca_fpattern.format(config.exp, config.m))
    # pcvar = S**2/np.sum(S**2)
    pc_cvar = np.loadtxt('data/architecture/pca_cvar_n{}.csv'.format(config.m))[:,1]
    pcvar = np.diff(pc_cvar, prepend=0)
    # n_plot = len(pcvar[pcvar>0.01])
    # print('n_plot:', n_plot)
    n_plot = 3
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(1, n_plot+1, bottom=0.1, left=0.025, right=0.975, top=.925,
        wspace=0.3)
    axs = np.array([fig.add_subplot(gs[i]) for i in range(n_plot+1)])
    dy = 0.2
    # y1 = np.arange(8) - dy
    # y2 = np.arange(8) + dy
    y1 = np.arange(8)
    axs[0].barh(y1, indices['general_first_order'], height=0.65,
        color='#888888', label='First-order',
        xerr=(indices['boostrap']['general_first_order']),
        ecolor='k', capsize=2, zorder=5,
        error_kw={'elinewidth':0.75, 'capthick':0.75})
    # axs[0].barh(y2, indices['general_total_index'], height=0.35,
    #     color='#555555', label='Total',
    #     xerr=(indices['boostrap']['general_total_index']),
    #     ecolor='k', capsize=2, zorder=5,
    #     error_kw={'elinewidth':0.75, 'capthick':0.75})
    # axs[0].legend(bbox_to_anchor=(-0.1, -0.375, 1, 0.3), frameon=False,
    #     loc='upper left', borderaxespad=0, borderpad=0)
    
    # axs[0].set_title(r'$f_{\rm{w}}$')

    for k in range(n_plot):
        ax = axs[k+1]
        ax.barh(y1, indices['first_order'][k], height=0.65,
        color='#888888', label='First-order',
        xerr=(indices['boostrap']['first_order'][:, k]),
        ecolor='k', capsize=2, zorder=5,
        error_kw={'elinewidth':0.75, 'capthick':0.75})

        # ax.barh(y2, indices['total_index'][k], height=0.32,
        # color='#555555', label='First-order',
        # xerr=(indices['boostrap']['total_index'][:, k]),
        # ecolor='k', capsize=2, zorder=5,
        # error_kw={'elinewidth':0.75, 'capthick':0.75})

        ax.set_title('PC{} ({:.1%})'.format(k+1, pcvar[k]))

    for i,ax in enumerate(axs):
        ax.set_ylim([-1, 7.5])
        ax.grid(linestyle=':', which='both')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.], minor=True)

        ax.spines[['right', 'top']].set_visible(False)
        ax.text(0.025, 1., alphabet[i], transform=ax.transAxes,
            fontweight='bold', ha='left', va='top')

        ax.set_yticks(np.arange(8))
        ax.set_yticklabels([])
    
    fig.text(0.5, 0.01, 'Sensitivity index', ha='center', va='bottom')
    
    for ax in axs[1:]:
        ax.set_yticklabels([])
    fig.savefig(os.path.join(config.figures, 'sensitivity_indices.png'), dpi=400)
    fig.savefig(os.path.join(config.figures, 'sensitivity_indices.pdf'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_config)

    plot_all_indices(train_config)
    
if __name__=='__main__':
    main()
