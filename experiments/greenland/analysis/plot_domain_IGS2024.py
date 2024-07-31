import os
import sys

import pickle
import numpy as np
import os

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib import colors as mpc
import cmocean
import rasterio as rs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

# Geometry and compute misc fields
bed = np.vstack(np.load('../issm/data/geom/IS_bed.npy'))
surf = np.vstack(np.load('../issm/data/geom/IS_surface.npy'))
thick = surf - bed
bed[thick<50] = surf[thick<50] - 50
thick = surf - bed

aws_xy = [-217706.690013, -2504221.345267]

# Compute triangulation
# md = read_netCDF('IS_bamg.nc')
with open('../issm/data/geom/IS_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)

# md = model()
# md = meshconvert(md,mesh['elements'], mesh['x'], mesh['y'])
# triangulation = tri.Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
triangulation = tri.Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

# Find interesting nodes

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

# plot_nodes = [node1, node2, node3]
plot_nodes = [node1, node3]

with open('../issm/data/moulins/moulins_catchments.pkl', 'rb') as infile:
    basins = pickle.load(infile)

moulin_indices = np.array([basin['moulin'] for basin in basins])

fig = plt.figure(figsize=(8, 5))

gs = GridSpec(1, 2, width_ratios=(40, 100),
    hspace=0.35, left=0.1, bottom=0.4, right=0.95, top=1.05)
ax1 = fig.add_subplot(gs[0, 1])

ax2 = ax1.inset_axes((-0.75, -0.95, 1.6, 1.3))

h3 = 0.9
ax3 = ax1.inset_axes((-0.65, 1-h3 + 0.05, 0.3, h3))
cax = ax2.inset_axes((1.02, 0.1, 0.025, 0.63))
pos = np.array([296, 34, 38])
for ax in (ax1, ax2):

    sc = ax.tripcolor(triangulation, thick.flatten(), vmin=0, vmax=2500, cmap=cmocean.cm.ice_r,
        edgecolor='#aaaaaa', linewidth=0.1)
    if ax is ax2:
        cbar = fig.colorbar(sc, shrink=0.75, pad=0.02, cax=cax, orientation='vertical')
        cbar.set_label('Ice thickness (m)')
    if ax is ax1:
        ms = 5
        mlw = 1
    else:
        ms = 10
        mlw = 2
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    ax.plot(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3,
        marker='.', color='k', markersize=ms/2, linestyle='', label='Moulins')
    iii=0
    for node_index in plot_nodes:
        colors = cmocean.cm.algae([0.4, 0.75])
        ax.plot(mesh['x'][node_index]/1e3, mesh['y'][node_index]/1e3, 
            marker='s', color=colors[iii], zorder=10, markeredgecolor='w',
            markersize=ms, markeredgewidth=mlw, linestyle='',
            label='{:.0f} m asl.'.format(surf[node_index][0]))
        iii+=1
    
    ax.plot(mesh['x'][pos]/1e3, mesh['y'][pos]/1e3, linestyle='',
        marker='*', color='k', markersize=ms/1.5, label=r'$p_{\rm{w}}=0$ outlets')
    ax.plot(aws_xy[0]/1e3, aws_xy[1]/1e3, 'm^', markersize=ms/1.5, label='KAN_L AWS')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    ax.set_facecolor('none')

zmax = 1850
surf0 = surf[:, 0]
xmax = np.max(mesh['x'][surf0<=zmax])/1e3
ymax = np.max(mesh['y'][surf0<=zmax])/1e3
xmin = np.min(mesh['x'][surf0<=zmax])/1e3
ymin = np.min(mesh['y'][surf0<=zmax])/1e3
ax2.set_xlim([xmin-1, xmax])
ax2.set_ylim([ymin, ymax])

ax1.set_xlim([xmin-2, np.max(mesh['x'])/1e3])

print('Length of area below zmax:', xmax - xmin)

rect = Rectangle(xy=(xmin, ymin), width=(xmax-xmin), height=(ymax-ymin))
pc = PatchCollection([rect], facecolor='none', edgecolor='k',
    linestyle=':', linewidth=1)
ax1.add_collection(pc)


scale = Rectangle(xy=(xmin+0, ymin+2), width=50, height=1.5, zorder=15)
spc = PatchCollection([scale], facecolor='k', clip_on=False)
ax2.add_collection(spc)
ax2.text(xmin+0+0.5*50, ymin+2+2.5, '50 km', ha='center', va='bottom')

ax2.legend(bbox_to_anchor=(0, 0.15, 0.5, 0.8), 
    frameon=False, loc='lower left', borderpad=0, borderaxespad=0)



scale2 = Rectangle(xy=(mesh['x'].max()/1e3-10-100, ymax-10), width=100, height=5, zorder=15)
spc2 = PatchCollection([scale2], facecolor='k', clip_on=False)
ax1.add_collection(spc2)
ax1.text(mesh['x'].max()/1e3-10-0.5*100, ymax-10+5+2, '100 km', ha='center', va='bottom')

with rs.open('../issm/data/geom/GimpIceMask_90m_2015_v1.2.tif') as geotiff:
    raster_mask = geotiff.read(1)

    rxmin = geotiff.bounds.left
    rxmax = geotiff.bounds.right
    rymin = geotiff.bounds.bottom
    rymax = geotiff.bounds.top
    nrows,ncols = geotiff.shape
    
x = np.linspace(rxmin, rxmax, ncols+1)[:-1]/1e3
y = np.linspace(rymin, rymax, nrows+1)[0:-1][::-1]/1e3
[xx, yy] = np.meshgrid(x, y)

inc = 10
ax3.contour(xx[::inc,::inc], yy[::inc,::inc], raster_mask[::inc,::inc], 
    levels=(0,), colors='k', linewidths=0.35)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
ax3.set_aspect('equal')
outline = np.loadtxt('../issm/data/geom/IS_outline.csv', skiprows=1, quotechar='"', delimiter=',')
# ax3.plot(outline[:, 1]/1e3, outline[:, 2]/1e3)
xy = np.array([outline[:, 1], outline[:, 2]]).T
ax3.tripcolor(triangulation, thick.flatten(), vmin=0, vmax=2500, cmap=cmocean.cm.ice_r,
        edgecolor='none')
pg = Polygon(xy[::10]/1e3, closed=True, facecolor='none', edgecolor='b', linewidth=1)
ax3.add_patch(pg)
ax3.set_facecolor('none')
ax3.plot(aws_xy[0]/1e3, aws_xy[1]/1e3, 'm^', markersize=3)

fig.text(0.025, 0.975, 'a',
    fontweight='bold', va='top', ha='left')

fig.text(0.38, 0.975, 'b',
    fontweight='bold', va='top', ha='left')

ax1.text(xmin+2, ymin+2, 'c', va='bottom', ha='left', zorder=10)
# ax1.plot(xmaymin+1, 'rx')
# print(xmin, xmax)

fig.text(0.025, 0.55, 'c', fontweight='bold',
    va='bottom', ha='left')

fig.savefig('figures/IGS_2024/greenland_domain_summary.png', dpi=400)
