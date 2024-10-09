"""
Plot summary of synthetic GlaDS setup
"""

import os
import pickle

import numpy as np

fs = 8
import matplotlib
matplotlib.rc('font', size=fs)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
import cmocean


# Read data
surf = np.load('geom/synthetic_surface.npy')
bed = np.load('geom/synthetic_bed.npy')
temp = np.loadtxt('melt/KAN_L_2014_temp_clipped.txt', delimiter=',')
moulins = np.loadtxt('moulins/moulin_indices.csv')
with open('geom/synthetic_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)
mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

ff = np.load('../train/synthetic_ff.npy', mmap_mode='r')

fig = plt.figure(figsize=(6, 3))
gs = GridSpec(2, 2, left=0.085, right=0.975, bottom=0.1, top=0.95,
    hspace=0.2, wspace=0.3,
    height_ratios=(100, 75))
axs = np.array([
    [fig.add_subplot(gs[i,j]) for j in range(2)]
    for i in range(2)])

# (a) Temperature timeseries
tt = temp[:, 0]
T = temp[:, 1] - 0.005*390
axs[0,0].plot(tt*12/365, T, color='k')
axs[0,0].set_ylim([0, 12])
axs[0,0].set_yticks([0, 4, 8, 12])
axs[0,0].set_xticks([4, 6, 8, 10])
axs[0,0].set_xlim([4, 10])
axs[0,0].set_xticklabels(['May', 'July', 'Sep', 'Nov'])
axs[0,0].grid(linestyle=':')
axs[0,0].set_ylabel(r'Temperature ($^{\circ}{\rm{C}}$)')
axs[0,0].text(0.025, 0.95, '(a)', transform=axs[0,0].transAxes,
    fontweight='bold', ha='left', va='top')

# (b) 3D Domain perspective
axs[1,0].set_visible(False)
# ax3d = fig.add_subplot(gs[1,0], projection='3d', computed_zorder=False)
ax3d = fig.add_subplot(projection='3d', computed_zorder=False, facecolor='none')
ax3d.set_position(Bbox.from_extents(-0.2, -0.05, 0.75, 0.6))
ax3d.plot_trisurf(mtri, bed, cmap=cmocean.cm.turbid, vmin=300, vmax=365,
    edgecolor='none', linewidth=0., antialiased=False)
ax3d.plot_trisurf(mtri, surf, cmap=cmocean.cm.ice, edgecolor='#444444', linewidth=0.025, alpha=1,
    antialiased=True, vmin=0, vmax=2000, zorder=3)
ax3d.view_init(elev=20, azim=-125) #Works!
ax3d.set_box_aspect((4, 1, 1))
ax3d.set_aspect('equalxy')
ax3d.set_yticks([0, 12.5, 25], ['0.0', '12.5', '25.0'], rotation=30)
ax3d.set_zlim([300, 2000])
ax3d.set_xlabel('Distance from terminus (km)', labelpad=12)
ax3d.zaxis.set_rotate_label(False)
ax3d.set_zlabel('Elevation (m asl.)', rotation=90, labelpad=0)
fig.text(0.085, 0.35, '(b)',
    fontweight='bold', ha='left', va='top')

# (c) Flotation fraction timeseries
nodes = [-1, -1, -1]
xpos = [15e3, 30e3, 50e3]
ypos = [12.5e3, 12.5e3, 12.5e3]
nodexy = np.array([mesh['x'], mesh['y']]).T
nodes[0] = np.argmin( (nodexy[:, 0]-xpos[0])**2 + (nodexy[:, 1]-ypos[0])**2)
nodes[1] = np.argmin( (nodexy[:, 0]-xpos[1])**2 + (nodexy[:, 1]-ypos[1])**2)
nodes[2] = np.argmin( (nodexy[:, 0]-xpos[2])**2 + (nodexy[:, 1]-ypos[2])**2)
colors = cmocean.cm.algae([0.25, 0.5, 0.75])
tstep = 205
m = 10
nx = len(mesh['x'])
nt = 365

for i in range(len(nodes)):
    node = nodes[i]
    ax3d.plot(mesh['x'][node]/1e3, mesh['y'][node]/1e3, surf[node]+600,
        'o', markersize=5, zorder=5, color=colors[i], markeredgewidth=1, markeredgecolor='w')
    ax3d.plot([mesh['x'][node]/1e3, mesh['x'][node]/1e3],
            [mesh['y'][node]/1e3, mesh['y'][node]/1e3],
            [surf[node], surf[node]+500], color='k',
            linewidth=1, zorder=4)
            
    ax3d.plot([mesh['x'][node]/1e3, mesh['x'][node]/1e3],
            [mesh['y'][node]/1e3, mesh['y'][node]/1e3],
            [350, surf[node]], color='k',
            linewidth=1, linestyle=':', zorder=1)
ax3d.grid(linestyle=':')
ax3d.xaxis._axinfo['grid'].update({'linestyle':':', 'linewidth':0.5})
ax3d.yaxis._axinfo['grid'].update({'linestyle':':', 'linewidth':0.5})
ax3d.zaxis._axinfo['grid'].update({'linestyle':':', 'linewidth':0.5})

axs[0,1].plot(np.arange(nt)*12/365, ff[:, 10].reshape((nx, nt))[nodes[1],:], color=colors[1])
axs[0,1].set_ylim([0, 1.5])
axs[0,1].set_xticks([4, 6, 8, 10])
axs[0,1].set_xlim([4, 10])
axs[0,1].set_xticklabels(['May', 'July', 'Sep', 'Nov'])
axs[0,1].grid(linestyle=':')
axs[0,1].set_ylabel(r'$p_{\rm{w}}/p_{\rm{i}}$',)
axs[0,1].text(0.025, 0.95, '(c)', transform=axs[0,1].transAxes,
    fontweight='bold', ha='left', va='top')
axs[0,1].axvline(tstep*12/365, color='k', linestyle='dashed')

# (d) Flotation fraction map
axs[1,1].set_visible(False)
gs2 = GridSpecFromSubplotSpec(3, 3, gs[1,1],
    height_ratios=(50, 8, 100),
    width_ratios=(10, 100, 10),
    hspace=0.1)
ax = fig.add_subplot(gs2[-1,:])
cax = fig.add_subplot(gs2[-2,1])
tripc = ax.tripcolor(mtri, ff[:, 10].reshape((nx, nt))[:, tstep], 
    vmin=0, vmax=1, cmap=cmocean.cm.dense, rasterized=True)
ax.set_aspect('equal')
ax.set_xlim([0, 100])
ax.set_ylim([0, 25])
ax.set_yticks([0, 12.5, 25])
ax.text(0.025, 0.95, '(d)', transform=ax.transAxes,
    fontweight='bold', ha='left', va='top')
for i in range(3):
    node = nodes[i]
    ax.plot(mesh['x'][node]/1e3, mesh['y'][node]/1e3,
        's', markersize=5, color=colors[i],
        markeredgewidth='1', markeredgecolor='w')
ax.set_xlabel('Distance from terminus (km)', labelpad=0)
# ax.set_rasterized(True)
cbar = fig.colorbar(tripc, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cbar.set_label(r'$p_{\rm{w}}/p_{\rm{i}}$')

fig.savefig('domain_summary.png', dpi=400)
fig.savefig('domain_summary.pdf', dpi=400)