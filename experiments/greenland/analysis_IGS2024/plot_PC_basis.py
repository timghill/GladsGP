import os
import argparse
import pickle

import numpy as np

import matplotlib
matplotlib.rc('font', size=14)

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

from src.utils import import_config

def plot_basis(config):
    nplot = 5
    meshin = os.path.join(config.sim_dir, config.mesh)
    mesh = np.load(meshin, allow_pickle=True)
    print(mesh)

    S = np.load('data/models/pca_greenland_n{:03d}_S.npy'.format(config.m)).astype(np.float32)
    U = np.load('data/models/pca_greenland_n{:03d}_U.npy'.format(config.m), 
        mmap_mode='r').astype(np.float32)
    V = np.load('data/models/pca_greenland_n{:03d}_Vh.npy'.format(config.m), 
        mmap_mode='r').astype(np.float32)
    pcvar = S**2/np.sum(S**2)
    # K = np.diag(S[:p]) @ Vh[:p] / np.sqrt(y_sim.shape[0])
    K = np.diag(S[:nplot]) @ V[:nplot] / np.sqrt(U.shape[0])
    print(K.shape)
    print(mesh['x'].shape)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    nx = len(mesh['x'])
    nt = int(K.shape[1]/nx)
    for i in range(nplot):
        ki = K[i].reshape((nx, nt))
        ki_mean = np.mean(ki, axis=1)
        fig,ax = plt.subplots(figsize=(6, 3.5))
        tpc = ax.tripcolor(mtri, ki[:,205], cmap=cmocean.cm.delta, vmin=-1, vmax=1)
        ax.set_aspect('equal')
        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('PC{} ({:.1%})'.format(i+1, pcvar[i]), fontsize=16)
        cax = ax.inset_axes((0.05, 0.25, 0.3, 0.05))
        # cax.set_xlabel('PC coefficient')
        # cax.xaxis.set_label_position('top')
        cb = fig.colorbar(tpc, cax=cax, orientation='horizontal')
        cb.set_label('PC coefficient')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95)
        fig.savefig('figures/IGS_2024/PC_basis_{:02d}.png'.format(i), dpi=400)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file')
    args = parser.parse_args()
    config = import_config(args.conf_file)
    plot_basis(config)

if __name__=='__main__':
    main()