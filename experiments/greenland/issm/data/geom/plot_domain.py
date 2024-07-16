import os
import sys

import pickle
import numpy as np
import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
from issmversion import issmversion
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from model import model
# from read_netCDF import read_netCDF
from meshconvert import meshconvert
from parameterize import parameterize
from GetAreas import GetAreas
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


# Geometry and compute misc fields
bed = np.vstack(np.load('IS_bed.npy'))
surf = np.vstack(np.load('IS_surface.npy'))
thick = surf - bed
bed[thick<50] = surf[thick<50] - 50
thick = surf - bed

# Compute triangulation
# md = read_netCDF('IS_bamg.nc')
with open('IS_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)

md = model()
md = meshconvert(md,mesh['elements'], mesh['x'], mesh['y'])
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

plot_nodes = [node1, node2, node3]

# area_ele = GetAreas(mesh['elements'], mesh['x'], mesh['y'])/1e6
# candidates = np.arange(md.mesh.numberofvertices)[md.mesh.vertexonboundary==0]
# rng = np.random.default_rng(seed=549374)
# moulin_indices = rng.choice(candidates, size=50)
with open('../moulins/moulins_catchments.pkl', 'rb') as infile:
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

for ax in (ax1, ax2):

    sc = ax.tripcolor(triangulation, thick.flatten(), vmin=0, vmax=2500, cmap=cmocean.cm.ice_r,
        edgecolor='#aaaaaa', linewidth=0.1)
    if ax is ax2:
        cbar = fig.colorbar(sc, shrink=0.75, pad=0.02, cax=cax, orientation='vertical')
        cbar.set_label('Ice thickness (m)')
        # cax.xaxis.tick_top()
        # cax.xaxis.set_label_position('top')
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
        colors = cmocean.cm.algae([0.2, 0.45, 0.7])
        ax.plot(mesh['x'][node_index]/1e3, mesh['y'][node_index]/1e3, 
            marker='s', color=colors[iii], zorder=10, markeredgecolor='w',
            markersize=ms, markeredgewidth=mlw)
        ax.plot(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3,
            marker='.', color='k', markersize=ms/2, linestyle='')
        iii+=1
    # ax.set_title('Flotation fraction and channel discharge')
    # ax.plot(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3,
    #     linestyle='', marker='x', markersize=8, color='r')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    ax.set_facecolor('none')

# cnorm = mpc.Normalize(vmin=Qmin, vmax=Qmax)
# cbar2 = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=Qcmap), cax=cax2, 
#     orientation='horizontal')
# cbar2.set_label('Channel discharge (m$^3$ s$^{-1}$)')
# cticks = cbar2.get_ticks()
# print(cticks)
# cticks[0] = Qmin
# cbar2.set_ticks(cticks)

zmax = 1850
surf0 = surf[:, 0]
xmax = np.max(mesh['x'][surf0<=zmax])/1e3
ymax = np.max(mesh['y'][surf0<=zmax])/1e3
xmin = np.min(mesh['x'][surf0<=zmax])/1e3
ymin = np.min(mesh['y'][surf0<=zmax])/1e3
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])

ax1.set_xlim([xmin, np.max(mesh['x'])/1e3])

print('Length of area below zmax:', xmax - xmin)


rect = Rectangle(xy=(xmin, ymin), width=(xmax-xmin), height=(ymax-ymin))
pc = PatchCollection([rect], facecolor='none', edgecolor='k',
    linestyle=':', linewidth=1)
ax1.add_collection(pc)


scale = Rectangle(xy=(xmin+10, ymin+2), width=50, height=2.5, zorder=15)
spc = PatchCollection([scale], facecolor='k', clip_on=False)
ax2.add_collection(spc)
ax2.text(xmin+10+0.5*50, ymin+2+2.5, '50 km', ha='center', va='bottom')



scale2 = Rectangle(xy=(mesh['x'].max()/1e3-10-100, ymax-10), width=100, height=5, zorder=15)
spc2 = PatchCollection([scale2], facecolor='k', clip_on=False)
ax1.add_collection(spc2)
ax1.text(mesh['x'].max()/1e3-10-0.5*100, ymax-10+5+2, '100 km', ha='center', va='bottom')

# fig.subplots_adjust(left=0.35, bottom=0.2, right=1.0, top=1.)
# fig.subplots_adjust(left=0.31, bottom=0.325, right=1.05, top=1.)
# fig.savefig('figures/ff_discharge.png', dpi=400)

with rs.open('GimpIceMask_90m_2015_v1.2.tif') as geotiff:
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
outline = np.loadtxt('IS_outline.csv', skiprows=1, quotechar='"', delimiter=',')
# ax3.plot(outline[:, 1]/1e3, outline[:, 2]/1e3)
xy = np.array([outline[:, 1], outline[:, 2]]).T
ax3.tripcolor(triangulation, thick.flatten(), vmin=0, vmax=2500, cmap=cmocean.cm.ice_r,
        edgecolor='none')
pg = Polygon(xy[::10]/1e3, closed=True, facecolor='none', edgecolor='b', linewidth=1)
ax3.add_patch(pg)
ax3.set_facecolor('none')

fig.text(0.025, 0.975, 'a',
    fontweight='bold', va='top', ha='left')

fig.text(0.38, 0.975, 'b',
    fontweight='bold', va='top', ha='left')

ax1.text(xmin+2, ymin+2, 'c', va='bottom', ha='left', zorder=10)
# ax1.plot(xmaymin+1, 'rx')
# print(xmin, xmax)

fig.text(0.025, 0.55, 'c', fontweight='bold',
    va='bottom', ha='left')

fig.savefig('greenland_domain_summary.png', dpi=400)

# Plot slope
fig, ax = plt.subplots(figsize=(8, 4))
slope = np.load('IS_surface_slope.npy')
tripc = ax.tripcolor(triangulation, slope, cmap=cmocean.cm.matter, vmin=0, vmax=0.1)
cbar = fig.colorbar(tripc)
cbar.set_label('Slope')
ax.set_aspect('equal')
fig.savefig('greenland_domain_slope.png', dpi=400)