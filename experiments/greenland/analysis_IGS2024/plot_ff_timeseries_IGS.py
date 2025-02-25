"""
Basic plots
"""

import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.expanduser('~/SFU-code'))
from palettes.code import tools, palettes

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import cm
from matplotlib import colors as mpc
import cmocean

from src import utils

config = utils.import_config('../train_config.py')

simnum = 237
base = os.path.join(config.sim_dir, 'RUN/output_{:03d}/'.format(simnum))

# Read model outputs
ff = np.load(base + 'ff.npy').astype(np.float32)
Q = np.load(base + 'Q.npy').astype(np.float32)
phi = np.load(base + 'phi.npy').astype(np.float32)
time = np.load(base + 'time.npy').astype(np.float32)

# print('time', time)
ix = 205
print(time[ix])


cm2 = mpc.LinearSegmentedColormap.from_list('', cmocean.cm.gray(np.linspace(0.05, 1, 128)))
cmap = tools.join_cmaps(cmocean.cm.dense, cm2, average=0, N1=128, N2=64)

# Geometry and compute misc fields
bed = np.vstack(np.load(os.path.join(config.sim_dir, '../data/geom/IS_bed.npy')))
surf = np.vstack(np.load(os.path.join(config.sim_dir, '../data/geom/IS_surface.npy')))
thick = surf - bed
bed[thick<50] = surf[thick<50] - 50
thick = surf - bed
p_ice = 910*9.8*thick
phi_bed = 1000*9.8*bed
p_w = phi - phi_bed

# Compute triangulation
with open(os.path.join(config.sim_dir, config.mesh), 'rb') as meshin:
    mesh = pickle.load(meshin)
edges = mesh['connect_edge']
triangulation = tri.Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

# Find interesting nodes
x1 = -210e3
x1mask = np.abs(mesh['x'] - x1)<2e3
bedmask = bed.copy()
bedmask[~x1mask] = np.nan
node1 = np.nanargmin(bedmask.flatten())
print('Node 1:', node1)
print(mesh['x'][node1]/1e3, mesh['y'][node1]/1e3, bed[node1], surf[node1])

z1 = 1200
zmask = np.abs(surf.flatten()-z1)<25
bedmask = bed.copy()
bedmask[~zmask] = np.nan
bedmedian = np.nanmean(bedmask)
node2 = np.nanargmin(np.abs(bedmask - bedmedian))
print('Node 2:', node2)
print(mesh['x'][node2]/1e3, mesh['y'][node2]/1e3, bed[node2], surf[node2])

z2 = 1500
bedmask = bed.copy()
zmask = np.abs(surf.flatten()-z2)<25
bedmask[~zmask] = np.nan
node3 = np.nanargmin(bedmask)
print('Node 3:', node3)
print(mesh['x'][node3]/1e3, mesh['y'][node3]/1e3, bed[node3], surf[node3])

plot_nodes = [node1, node3]

area_ele = mesh['area']/1e6
with open('../issm/data/moulins/moulins_catchments.pkl', 'rb') as infile:
    basins = pickle.load(infile)

moulin_indices = np.array([basin['moulin'] for basin in basins])

fig = plt.figure(figsize=(10, 4))
gs = GridSpec(3, 3, height_ratios=(50, 100, 10), width_ratios=(150, 25, 125),
    hspace=0.1, wspace=0.1, left=0.02, bottom=0.05, right=0.975, top=0.95)
ax2 = fig.add_subplot(gs[1:,0])
gst = GridSpecFromSubplotSpec(2, 1, gs[:-1, 2])
ax1 = ax2.inset_axes((0.35, 0.7, 0.9, 0.8))
cax1 = ax2.inset_axes((0.05, 1.25, 0.3, 0.4/10))
cax2 = ax2.inset_axes((0.05, 1.2, 0.3, 0.4/10))
colors = cmocean.cm.algae([0.4, 0.75])

for ax in (ax1, ax2):

    sc = ax.tripcolor(triangulation, ff[:, ix], vmin=0, vmax=1.5, cmap=cmap)
    if ax is ax2:
        cbar = fig.colorbar(sc, shrink=0.75, pad=0.02, cax=cax1, orientation='horizontal')
        cbar.set_label('Flotation fraction')
        cax1.xaxis.tick_top()
        cax1.xaxis.set_label_position('top')
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    iii=0
    for node_index in plot_nodes:
        if ax is ax1:
            ms = 5
            mlw = 1
        else:
            ms = 10
            mlw = 2
        ax.plot(mesh['x'][node_index]/1e3, mesh['y'][node_index]/1e3, 
            marker='s', color=colors[iii], zorder=10, markeredgecolor='w',
            markersize=ms, markeredgewidth=mlw)
        iii+=1

    Qmin = 10
    Qmax = 200
    channel_mask = np.abs(Q[:, ix])>Qmin
    channel0 = edges[:, 0]
    channel1 = edges[:, 1]
    cx = np.array([mesh['x'][channel0[channel_mask]], mesh['x'][channel1[channel_mask]]]).T/1e3
    cy = np.array([mesh['y'][channel0[channel_mask]], mesh['y'][channel1[channel_mask]]]).T/1e3

    Qnorm = lambda x: min(0.99, max(0.01, (x-Qmin)/(Qmax-Qmin)))
    # Qcmap = cmocean.cm.
    Qcmap = palettes.get_cmap('BrownYellow')
    for i in range(len(cx)):
        Qscore = Qnorm(np.abs(Q[channel_mask, ix][i]))
        color = Qcmap(Qscore)
        alpha = 7./8.
        lwscore = ((1-alpha) + alpha*Qscore)*mlw*1.5
        ax.plot(cx[i], cy[i], color=color, linewidth=lwscore)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    ax.set_facecolor('none')

cnorm = mpc.Normalize(vmin=Qmin, vmax=Qmax)
cbar2 = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=Qcmap), cax=cax2, 
    orientation='horizontal')
cbar2.set_label('Channel discharge (m$^3$ s$^{-1}$)')
cticks = cbar2.get_ticks()
cticks[0] = Qmin
cbar2.set_ticks(cticks)

zmax = 1600
surf0 = surf[:, 0]
xmax = np.max(mesh['x'][surf0<=zmax])/1e3
ymax = np.max(mesh['y'][surf0<=zmax])/1e3
xmin = np.min(mesh['x'][surf0<=zmax])/1e3
ymin = np.min(mesh['y'][surf0<=zmax])/1e3
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])

ax1.set_xlim([xmin, np.max(mesh['x'])/1e3])

rect = Rectangle(xy=(xmin, ymin), width=(xmax-xmin), height=(ymax-ymin))
pc = PatchCollection([rect], facecolor='none', edgecolor='k',
    linestyle=':', linewidth=1)
ax1.add_collection(pc)


scale = Rectangle(xy=(xmin+3, ymin-5), width=50, height=1.5, zorder=15)
spc = PatchCollection([scale], facecolor='k', clip_on=False)
ax2.add_collection(spc)
ax2.text(xmin+3+0.5*50, ymin-2.5, '50 km', ha='center', va='bottom')

# colors = cmocean.cm.algae([0.2, 0.5, 0.8])
tticks = 1 + np.array([4, 5, 6, 7, 8, 9, 10])/12
ttick_labels = ['May', '', 'July', '', 'Sept', '', 'Nov']
for i,nindex in enumerate(plot_nodes):
    axt = fig.add_subplot(gst[i,0])
    axt.grid(linestyle=':', linewidth=0.5)
    axt.set_ylabel('Flotation fraction')
    axt.set_xlim([1 + 4/12, 1 + 10/12])
    axt.set_xticks(tticks, ttick_labels)
    axt.spines[['right', 'top']].set_visible(False)
    axt.axvline(time[ix], color='k', linestyle='dashed', linewidth=7/8, zorder=1)
    elevi = surf[nindex][0]
    axt.plot(time, ff[nindex, :], color=colors[i], label='{:.0f} m'.format(elevi))
    axt.set_ylim([0.75, 1.55])
    axt.text(1, 1, '{:.0f} m'.format(elevi), transform=axt.transAxes, 
        ha='right', va='top', fontweight='bold', color=colors[i])
    fig.savefig('figures/IGS_2024/ff_discharge_timeseries_{:02d}.png'.format(i), dpi=400)
