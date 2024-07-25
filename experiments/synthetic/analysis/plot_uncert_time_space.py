import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

def plot_uncert():
    cv_lq = np.load('data/reference/cv_lower.npy')
    cv_uq = np.load('data/reference/cv_upper.npy')
    width = cv_uq - cv_lq
    nt = 365
    nx = int(cv_lq.shape[1]/nt)
    width = width.reshape((width.shape[0], nx, nt))
    width_time = width.mean(axis=(0,1))
    width_space = width.mean(axis=(0,2))

    mesh = np.load('../issm/data/geom/synthetic_mesh.pkl', allow_pickle=True)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(2,2, width_ratios=(100, 4), height_ratios=(150, 100),
        left=0.1, bottom=0.1, right=0.9, top=0.95,
        hspace=0.3, wspace=0.05)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    cax = fig.add_subplot(gs[1,1])

    tt = 12/365*np.arange(nt)
    ax1.plot(tt, width_time)
    ax1.grid(linestyle=':')
    ax1.set_xlim([0, 12])
    ax1.set_xlabel('Month')
    ax1.set_ylabel('95% uncertainty interval')
    ax1.set_ylim([0, 0.5])

    tripc = ax2.tripcolor(mtri, width_space, vmin=0, vmax=0.25, cmap=cmocean.cm.amp,
        rasterized=True)
    ax2.tricontour(mtri, width_space, levels=(0.05, 0.1, 0.15, 0.2, 0.25), 
        colors='k', linewidths=0.5)
    ax2.set_aspect('equal')
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0, 25])
    ax2.set_yticks([0, 12.5, 25])
    cbar = fig.colorbar(tripc, cax=cax)
    cbar.set_label('95% uncertainty interval')

    fig.savefig('figures/uncertainty_width.png', dpi=600)
    fig.savefig('figures/uncertainty_width.pdf', dpi=600)
    

if __name__=='__main__':
    plot_uncert()