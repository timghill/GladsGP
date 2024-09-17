"""
Assess channel convergence to steady state
"""

import collections

import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import colors
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import cmocean

from src import utils

# Import experiment configuration
config = utils.import_config('../train_config.py')

# Read channels and mesh
S = np.load('../issm/train/synthetic_S.npy', mmap_mode='r')
print('S.shape:', S.shape)
meshdict = np.load('../issm/data/geom/synthetic_mesh.pkl', allow_pickle=True)

class Model:
    def __init__(self, mesh):
        self.mesh = mesh

class Mesh:
    def __init__(self, meshdict):
        for key in meshdict.keys():
            setattr(self, key, meshdict[key])
        # self.numberofvertices = meshdict['numberofvertices']
        # self.numberofelements = meshdict['numberofelements']
        # self.connect_edge = meshdict['connect_edge']
        # self.edge_length = meshdict
        # self.elements = meshdict['elements']
        # self.x = meshdict['x']
        # self.y = meshdict['y']

mesh = Mesh(meshdict)
md = Model(mesh)
md = utils.reorder_edges(md)
dS = S[:, -1, :] - S[:, 0, :]
dS_threshold = 1
print('dS.shape:', dS.shape)

# Number of nonconverged
nonconverged = np.zeros(dS.shape, dtype=int)
nonconverged[dS>=dS_threshold] = 1
nnc = np.sum(nonconverged, axis=1)

mtri = Triangulation(mesh.x/1e3, mesh.y/1e3, mesh.elements-1)

fig = plt.figure(figsize=(3, 4))
gs = GridSpec(1, 2, width_ratios=(100, 8), hspace=0.05,
    left=0.1, right=0.8, top=.95)
ax = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])
norm = colors.Normalize(vmin=0, vmax=512)
cmap = cmocean.cm.ice_r
xmax = 0
for i in range(len(mesh.connect_edge)):
    if nnc[i]>0:
        ax.plot(mesh.x[mesh.connect_edge[i,:]]/1e3, mesh.y[mesh.connect_edge[i,:]]/1e3,
            color=cmap(norm(nnc[i])))
        xmax = max(xmax, max(mesh.x[mesh.connect_edge[i,:]]))
nnc_max = np.max(nnc)
nnc_argmax = np.argmax(nnc)
edgex = np.mean(mesh.x[mesh.connect_edge[nnc_argmax,:]]/1e3)
edgey = np.mean(mesh.y[mesh.connect_edge[nnc_argmax,:]]/1e3)
ax.text(edgex, edgey, nnc_max)

ax.set_aspect('equal')
ax.set_xlabel
ax.set_xlim([0, 10])
ax.set_ylim([0, 25])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.set_label('Number of non-converged simulations')
cbar.set_ticks([0, 128, 256, 128+256, 512])

ax.set_xlabel('Distance from terminus (km)')
ax.set_ylabel('Distance across-glacier (km)')
fig.savefig('figures/channel_convergence_fraction.png', dpi=400)
fig.savefig('figures/channel_convergence_fraction.pdf', dpi=400)
