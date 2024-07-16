import os

import argparse
from src.utils import import_config

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import colors
import cmocean
from scipy import stats

def plot_subsets(design, bounds, para_names, figure=None):
    scale = 'log'
    X = design['physical']
    dim = X.shape[1]

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(dim-1, dim-1, left=0.1, right=0.975, bottom=0.1, top=0.975)

    m_max = np.log2(X.shape[0])
    ms = np.arange(3-0.5, m_max+0.5+1)

    m = np.floor(np.log2(np.arange(0, X.shape[0])))
    m[0] = 0
    print('m:', m)
    for i in range(dim):
        for j in range(i+1, dim):
            ax = fig.add_subplot(gs[j-1, i])
            ax.set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_visible(True)
            cmap = cmocean.cm.ice
            cbounds = np.linspace(ms.min(), ms.max(), len(ms))
            norm = colors.BoundaryNorm(cbounds, cmap.N)
            for mi in range(len(ms)-1):
                mexp = int(ms[-mi-1]-0.5)
                cval = m[:2**mexp] + 1
                sc = ax.scatter(X[:2**mexp, i], X[:2**mexp, j], 
                    s=25, c=cval, marker='.', cmap=cmap, norm=norm,
                    edgecolor='#888888', linewidth=0.12)
            
            ax.set_xscale(scale)
            ax.set_yscale(scale)

            if i==0:
                ax.set_ylabel(para_names[j])
            else:
                ax.set_yticklabels([])
            
            if j==dim-1:
                ax.set_xlabel(para_names[i])
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_xticklabels([])
    caxgs = GridSpecFromSubplotSpec(5, 1, gs[0, 1])
    cax = fig.add_subplot(caxgs[0, 0])
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cbar.set_label('log$_2$ samples')
    cbar.set_ticks(ms[:-1] + 0.5)
    if figure:
        fig.savefig(figure, dpi=800)
    
    # Now individually: plot just first input pair to highlight iterative space-filling nature
    fig2, ax2 = plt.subplots()
    ax2.grid(linestyle=':', which='both')
    ax2.set_xlim(10**bounds[0])
    ax2.set_ylim(10**bounds[3])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.set_xlabel(para_names[0])
    ax2.set_ylabel(para_names[3])
    # k_vals = [8, 6, 5, 4]
    k_vals = [5, 6, 7, 8]
    for k in k_vals:
        ax2.scatter(X[:2**k, 0], X[:2**k, 3], cmap=cmap, norm=norm,
            s=75, c=(np.ones(2**k)*k), edgecolor='#888888', linewidth=0.3, zorder=k_vals[-1]-k)
        fig2.savefig('sobol_subsets_{:02d}.png'.format(k), dpi=400)
        
    return fig

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file')
    args = parser.parse_args()
    config = import_config(args.conf_file)

    txtargs = dict(delimiter=',', skiprows=1)
    X_phys = np.loadtxt(config.X_physical, **txtargs)
    X_std = np.loadtxt(config.X_standard, **txtargs)
    X_log = np.loadtxt(config.X_log, **txtargs)
    design = dict(standard=X_std, physical=X_phys, log=X_log)
    para_names = config.theta_names
    bounds = config.theta_bounds
    plot_subsets(design, bounds, para_names, 
        os.path.join(config.figures, 'sobol_subsets.png'))