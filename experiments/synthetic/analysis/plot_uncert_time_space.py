import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

def plot_uncert():
    cv_lq = np.load('data/reference/cv_lower.npy')
    cv_uq = np.load('data/reference/cv_upper.npy')
    y_sim = np.load('../issm/train/synthetic_ff.npy').astype(np.float32).T
    nt = 365
    m = y_sim.shape[0]
    nx = int(y_sim.shape[1]/nt)
    y_sim = y_sim.reshape((m, nx, nt))
    y_std_space = np.std(y_sim, axis=(0,2))
    y_std_time = np.std(y_sim, axis=(0,1))
    global_std = 1/np.sqrt(17.5)
    width = cv_uq - cv_lq
    nt = 365
    nx = int(cv_lq.shape[1]/nt)
    width = width.reshape((width.shape[0], nx, nt))
    width_time = width.mean(axis=(0,1))
    width_space = width.mean(axis=(0,2))

    mesh = np.load('../issm/data/geom/synthetic_mesh.pkl', allow_pickle=True)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

    fig = plt.figure(figsize=(8, 4))
    gs = GridSpec(4,2, width_ratios=(100, 100), height_ratios=(150, 100, 10, 100),
        left=0.1, bottom=0.1, right=0.95, top=0.95,
        hspace=0.05, wspace=0.25)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[-1,0])

    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[-1,1])

    cax1 = fig.add_subplot(gs[2, 0])
    cax2 = fig.add_subplot(gs[2, 1])

    tt = 12/365*np.arange(nt)
    ax1.plot(tt, width_time)
    ax3.plot(tt, y_std_time)
    for ax in (ax1,ax3):
        ax.grid(linestyle=':')
        ax.set_xlim([0, 12])
        ax.set_xlabel('Month')
    ax1.set_ylabel('95% uncertainty interval')
    ax3.set_ylabel('Ensemble std dev')
    ax1.set_ylim([0, 0.5])
    ax3.set_ylim([0, 0.65])
    

    tripc = ax2.tripcolor(mtri, width_space, vmin=0, vmax=0.25, 
        cmap=cmocean.cm.matter, rasterized=True)
    ax2.tricontour(mtri, width_space, levels=(0.05, 0.1, 0.15, 0.2, 0.25), 
        colors='k', linewidths=0.5)

    stdpc = ax4.tripcolor(mtri, y_std_space, vmin=0, vmax=0.4,
        cmap=cmocean.cm.matter, rasterized=True)
    ax4.tricontour(mtri, y_std_space, levels=(0.08, 0.16, 0.24, 0.32, 0.4), 
        colors='k', linewidths=0.5)

    for ax in (ax2, ax4):
        ax.set_aspect('equal')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 25])
        ax.set_yticks([0, 12.5, 25])

    cbar1 = fig.colorbar(tripc, cax=cax1, orientation='horizontal')
    cbar1.set_label('95% uncertainty interval')

    cbar2 = fig.colorbar(stdpc, cax=cax2, orientation='horizontal')
    cbar2.set_label('Ensemble std dev')

    for cax in (cax1, cax2):
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')

    fig.savefig('figures/uncertainty_width.png', dpi=600)
    fig.savefig('figures/uncertainty_width.pdf', dpi=600)
    

if __name__=='__main__':
    plot_uncert()