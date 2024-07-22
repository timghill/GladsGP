"""
Basic plots
"""

import pickle
import numpy as np
import os
import sys
# ISSM_DIR = os.getenv('ISSM_DIR')
# sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
# sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
# from issmversion import issmversion
# sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
# import devpath
# from read_netCDF import read_netCDF
# from meshconvert import meshconvert
# from parameterize import parameterize
# from GetAreas import GetAreas
import matplotlib
#matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib import colors as mpc
import cmocean

import rasterio as rs

from src import utils

config = utils.import_config('../train_config.py')

simnum = 237
base = os.path.join(config.sim_dir, 'RUN/output_{:03d}/'.format(simnum))

# Read model outputs
ff = np.load(base + 'ff.npy')
Q = np.load(base + 'Q.npy')
S = np.load(base + 'S.npy')
N = np.load(base + 'N.npy')
phi = np.load(base + 'phi.npy')
time = np.load(base + 'time.npy')

# print('time', time)
ix = 208
print(time[ix])

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
# md = read_netCDF('../issm/data/geom/IS_bamg.nc')
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

plot_nodes = [node1, node2, node3]

area_ele = mesh['area']/1e6
# candidates = np.arange(md.mesh.numberofvertices)[md.mesh.vertexonboundary==0]
# rng = np.random.default_rng(seed=549374)
# moulin_indices = rng.choice(candidates, size=50)
with open('../issm/data/moulins/moulins_catchments.pkl', 'rb') as infile:
    basins = pickle.load(infile)

moulin_indices = np.array([basin['moulin'] for basin in basins])

# FIGURES

# 1. Mean flotation fraction timeseries
fig, ax = plt.subplots()
ff_el = np.mean(ff[mesh['elements']-1, :], axis=1)
print(ff_el.shape)
ff_mean = np.sum(np.vstack(area_ele)*ff_el, axis=0)/np.sum(area_ele)
print(ff_mean.shape)
ax.plot(time, ff_mean)
ax.set_xlabel('Years')
ax.set_title('Mean flotation fraction')
fig.savefig('figures/mean_ff.png', dpi=400)

# 2. Total channel area timeseries
fig, ax = plt.subplots()
ax.plot(time, np.sum(S, axis=0))
ax.set_title('Total S (m$^2$)')
ax.set_xlabel('Years')
fig.savefig('figures/mean_S.png', dpi=400)

# 3. Flotation fraction and channel discharge
# fig, (ax1,ax2) = plt.subplots(figsize=(8, 6), nrows=2)
# fig = plt.figure(figsize=(6, 6))
# fig, ax1 = plt.subplots(figsize=(8, 4))
fig = plt.figure(figsize=(10, 4))
gs = GridSpec(3, 3, height_ratios=(50, 100, 10), width_ratios=(150, 25, 125),
    hspace=0.1, wspace=0.1, left=0.02, bottom=0.05, right=0.975, top=0.95)
# ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1:,0])
axt = fig.add_subplot(gs[:-1, 2])
# gs = GridSpec(4, 1, height_ratios=(10, 25, 100, 100),
#     hspace=-0.2, left=0.02, right=0.98, bottom=0.02, top=0.9)
# ax1 = fig.add_subplot(gs[2,0])
# ax2 = fig.add_subplot(gs[3,0])
# cax = fig.add_subplot(gs[0,0])

# ax2 = ax1.inset_axes((-0.6, -0.6, 1.4, 1.1))
ax1 = ax2.inset_axes((0.35, 0.7, 0.9, 0.8))
cax1 = ax2.inset_axes((0.05, 1.25, 0.3, 0.4/10))
cax2 = ax2.inset_axes((0.05, 1.2, 0.3, 0.4/10))

for ax in (ax1, ax2):

    sc = ax.tripcolor(triangulation, ff[:, ix], vmin=0, vmax=1, cmap=cmocean.cm.dense)
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
            ms = 3
            mlw = 0.5
        else:
            ms = 5
            mlw = 1
        colors = cmocean.cm.algae([0.2, 0.5, 0.8])
        ax.plot(mesh['x'][node_index]/1e3, mesh['y'][node_index]/1e3, 
            marker='s', color=colors[iii], zorder=10, markeredgecolor='w',
            markersize=ms, markeredgewidth=mlw)
        iii+=1
    # ax.set_title('Flotation fraction and channel discharge')
    # ax.plot(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3,
    #     linestyle='', marker='x', markersize=8, color='r')

    Qmin = 1
    Qmax = 200
    channel_mask = np.abs(Q[:, ix])>Qmin
    channel0 = edges[:, 0]
    channel1 = edges[:, 1]
    cx = np.array([mesh['x'][channel0[channel_mask]], mesh['x'][channel1[channel_mask]]]).T/1e3
    cy = np.array([mesh['y'][channel0[channel_mask]], mesh['y'][channel1[channel_mask]]]).T/1e3

    Qnorm = lambda x: min(0.99, max(0.01, (x-Qmin)/(Qmax-Qmin)))
    Qcmap = cmocean.cm.gray_r
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
print(cticks)
cticks[0] = Qmin
cbar2.set_ticks(cticks)

# surface = np.load('../issm/data/geom/IS_surface.npy')
# zmax = np.max(surf[moulin_indices, 0])+50
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


# fig.subplots_adjust(left=0.35, bottom=0.2, right=1.0, top=1.)
# fig.subplots_adjust(left=0.31, bottom=0.325, right=1.05, top=1.)
# fig.savefig('figures/ff_discharge.png', dpi=400)

with rs.open('../issm/data/geom/GimpIceMask_90m_2015_v1.2.tif') as geotiff:
    raster_mask = geotiff.read(1)

    xmin = geotiff.bounds.left
    xmax = geotiff.bounds.right
    ymin = geotiff.bounds.bottom
    ymax = geotiff.bounds.top
    nrows,ncols = geotiff.shape
    
x = np.linspace(xmin, xmax, ncols+1)[:-1]/1e3
y = np.linspace(ymin, ymax, nrows+1)[0:-1][::-1]/1e3
[xx, yy] = np.meshgrid(x, y)

# ax3 = ax1.inset_axes((0.825, 0.3, 0.3, 0.75))
# inc = 10
# ax3.contour(xx[::inc,::inc], yy[::inc,::inc], raster_mask[::inc,::inc], levels=(0,), colors='k', linewidths=0.4)
# ax3.set_xticks([])
# ax3.set_yticks([])
# ax3.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
# ax3.set_aspect('equal')
# outline = np.loadtxt('../issm/data/geom/IS_outline.csv', skiprows=1, quotechar='"', delimiter=',')
# # ax3.plot(outline[:, 1]/1e3, outline[:, 2]/1e3)
# xy = np.array([outline[:, 1], outline[:, 2]]).T
# pg = Polygon(xy[::10]/1e3, closed=True, color=cmocean.cm.dense(0.75))
# ax3.add_patch(pg)
# ax3.set_facecolor('none')


# fig, ax = plt.subplots(figsize=(6, 3))
colors = cmocean.cm.algae([0.2, 0.5, 0.8])
# colors = ['r', 'g', 'b']
for i,nindex in enumerate(plot_nodes):
    elevi = surf[nindex][0]
    axt.plot(time, ff[nindex, :], color=colors[i], label='{:.0f} m'.format(elevi))
axt.grid(linestyle=':', linewidth=0.5)
axt.set_xlabel('Year')
axt.set_ylabel('Flotation fraction')
axt.set_xlim([1 + 4/12, 1 + 10/12])
# ax.set_ylim([0.5, 1])
# axt.set_ylim([0.5, 1.5])
axt.legend()
axt.axvline(time[ix], color='k', linestyle='dashed', linewidth=7/8, zorder=1)
# fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
fig.savefig('figures/ff_discharge_timeseries.png', dpi=400)


fig, ax = plt.subplots(figsize=(8, 4))
npc = ax.tripcolor(triangulation, N[:, ix]/1e6, vmin=0, cmap=cmocean.cm.ice)
cbar = fig.colorbar(npc)
ax.tricontour(triangulation, N[:, -1]/1e6, levels=(0.,), colors='r')
ax.set_title('N (MPa)')
ax.set_aspect('equal')
fig.savefig('figures/N.png', dpi=400)

fig, ax = plt.subplots(figsize=(8, 4))
npc = ax.tripcolor(triangulation, p_w[:, ix]/1e6, cmap=cmocean.cm.thermal)
ax.tricontour(triangulation, p_w[:, ix]/1e6, levels=(0.,), colors='k')
cbar = fig.colorbar(npc)
ax.set_title(r'$p_{\rm{w}}$ (MPa)')
ax.set_aspect('equal')
fig.savefig('figures/pw.png', dpi=400)

fig, ax = plt.subplots(figsize=(8, 4))
npc = ax.tripcolor(triangulation, bed[:, 0], cmap=cmocean.cm.deep)
cbar = fig.colorbar(npc)
outlet_nodes = np.array([296, 34, 38])
ax.plot(mesh['x'][outlet_nodes]/1e3, mesh['y'][outlet_nodes]/1e3,
    marker='x', color='r', linestyle='')
ax.set_title('Bed (m asl.)')
ax.set_aspect('equal')
fig.savefig('figures/bed.png', dpi=400)

fig, ax = plt.subplots()
ax.hist(p_w[:, ix]/1e6)
ax.set_title(r'$p_{\rm{w}}$ (MPa)')
fig.savefig('figures/hist_pw.png', dpi=400)


# fig, ax = plt.subplots()
# # md.geometry.base = bed
# # md.geometry.bed =bed
# # md.geometry.surface = surf
# # md.geometry.thickness = thick
# bx = mesh['x'][md.mesh.vertexonboundary==1]
# by = mesh['y'][md.mesh.vertexonboundary==1]
# # bz = md.geometry.bed[md.mesh.vertexonboundary==1]
# # bt = md.geometry.thickness[md.mesh.vertexonboundary==1]

# phi_bndry = 1000*9.8*bz + 910*9.8*bt

# ax.plot(bx, by, linestyle='', marker='o')
# for i in range(len(md.mesh.vertexonboundary)):
#     if md.mesh.vertexonboundary[i]==1:
#         ax.text(mesh['x'][i], mesh['y'][i], '{}'.format(i))
# ax.set_title('Boundary vertices')
# ax.set_aspect('equal')
# fig.savefig('figures/boundary.png', dpi=400)

# fig, ax = plt.subplots()
# ax.plot(phi_bndry)
# ax.set_title(r'$\phi$ on boundary')
# fig.savefig('figures/phi_boundary.png', dpi=400)

print('Min thickness', np.min(thick))

fig,ax = plt.subplots()
vv = np.load('../issm/data/velocity/IS_vel.npy')
vv_pc = ax.tripcolor(triangulation, (vv), cmap=cmocean.cm.speed, norm=matplotlib.colors.LogNorm())
print('Max speed:', np.max(vv))
print('Min speed:', np.min(vv))
ax.set_aspect('equal')
fig.colorbar(vv_pc)
ax.set_title('Surface velocity (m/a)')


# plt.show()
