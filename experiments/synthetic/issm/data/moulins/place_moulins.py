"""
Place moulins using elevation-dependent density and accounting
for highly nonuniform triangle areas
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import patches
from matplotlib.collections import PatchCollection
from scipy import stats
from shapely.geometry.polygon import Polygon

import pickle

# Specify moulin distribution parameters
# Density parameters are computed from Yang et al. (2016) 
# supraglacial drainage map,
# https://doi.org/10.1002/2016JF003927
mu = 1138.25114463
sigma = 280.12484997
amp = 42.12111318
print('amp:', amp)

# Read in the mesh and elevation
surf = np.load('../geom/synthetic_surface.npy')

# Define elevation bins
dz = 100
# zz = np.arange(0, 3000+dz, dz)
zz = np.arange(300, 1900+dz, dz)
zc = 0.5*(zz[:-1] + zz[1:])

# md = read_netCDF('../geom/synthetic_mesh.nc')
with open('../geom/synthetic_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)
# Mesh
meshx = mesh['x']
meshy = mesh['y']
elements = mesh['elements']-1
vonboundary = mesh['vertexonboundary']

mtri = Triangulation(meshx/1e3, meshy/1e3, elements)

polys = [Polygon([(meshx[el[i]], meshy[el[i]]) for i in range(3)]) for el in elements]
# areas = np.array([pol.area for pol in polys])/1e6
area_ele = mesh['area']/1e6
print('Total catchment area (km2):', area_ele.sum())

surf_ele = np.mean(surf[elements], axis=1)
print('Elevation interpolated onto elements')
print(surf_ele.shape)
print(surf_ele)
print(surf_ele.min())
print(surf_ele.max())

densities = np.zeros(zc.shape)
areas = 0*zc
for i in range(len(zc)):
    zi = zc[i]
    msk = np.logical_and(surf_ele>=zz[i], surf_ele<zz[i+1])
    areas[i] = np.sum(area_ele[msk])

print('AREAS by elevation bin:')
print(areas)

densities = amp*stats.norm.pdf(zc, loc=mu, scale=sigma)
print('Density:', densities)
n_moulins = (densities*areas).astype(int)
print('Moulins per bin:', n_moulins)
print('Total moulins:', n_moulins.sum())

moulin_indices = []
rng = np.random.default_rng(seed=506839)
for i in range(len(zc)):
    ni = n_moulins[i]
    band_mask = np.logical_and(
        surf>=zz[i], surf<zz[i+1])
    band_mask = np.logical_and(band_mask, vonboundary==0)
    candidate_nodes = np.where(band_mask)[0]
    # print(candidate_nodes)
    for j in range(ni):
        ix = rng.choice(candidate_nodes, size=1)[0]
        moulin_indices.append(ix)
        dx = meshx[candidate_nodes] - meshx[ix]
        dy = meshy[candidate_nodes] - meshy[ix]
        dd = np.sqrt(dx**2 + dy**2)/1e3
        candidate_nodes = candidate_nodes[dd>=1]

moulin_indices = np.sort(moulin_indices)

np.savetxt('moulin_indices.csv', moulin_indices, fmt='%d')
print(moulin_indices)

mx = meshx[moulin_indices]
my = meshy[moulin_indices]
dxarr = np.vstack(mx) - mx
dyarr = np.vstack(my) - my
darr = np.sqrt(dxarr**2 + dyarr**2)/1e3
darr[darr==0] = np.nan
print(np.nanmin(darr))

nodex = meshx[elements]
nodey = meshy[elements]

elx = np.mean(nodex, axis=1)
ely = np.mean(nodey, axis=1)
print('elx.shape', elx.shape)

dx = np.vstack(elx) - mx
dy = np.vstack(ely) - my
dd = np.sqrt(dx**2 + dy**2)
print('dx.shape:', dx.shape)

catchment_nums = np.argmin(dd, axis=1)
catchment_nums[elx<5e3] = -1
print('Catchment nums:', catchment_nums)
print(catchment_nums.shape)

dmin = 2.5e3
catchment_outlets = np.array([], dtype=int)
for i in range(len(moulin_indices)):
    catchment_ix = np.where(catchment_nums==i)[0]
    zi = surf_ele[catchment_ix]
    # print('catchment_ix:', catchment_ix)
    # print('zi:', zi)
    # print('catchment_outlets', catchment_outlets)
    if len(catchment_outlets)>0:
        # Distance from other moulins
        distx = np.vstack(meshx[catchment_outlets]) - elx[catchment_ix]
        disty = np.vstack(meshy[catchment_outlets]) - ely[catchment_ix]
        d2 = np.min(np.sqrt(distx**2 + disty**2), axis=0)
    else:
        # Set the distance to > dmin so we pass
        d2 = dmin*np.ones(len(catchment_ix)) + 1

    # Penalize elevation for elements that are close to 
    # already placed moulins
    zi[d2<dmin] = zi[d2<dmin] + 1e6 - 1e-2*d2[d2<dmin]
    if (zi>1e6).all():
        print('WARNING: All penalized elevations are inf')
    is_boundary = np.max(vonboundary[elements[catchment_ix]], axis=1)>0
    zi[is_boundary] = zi[is_boundary] + 1e10
    # Find the element number of the lowest element value
    # print('zi:', zi)
    ele_ii = catchment_ix[np.argmin(zi)]
    z_ii = surf[elements[ele_ii]]
    moulin_ii = elements[ele_ii, np.argmin(z_ii)]
    old_outlets = catchment_outlets
    catchment_outlets = np.zeros(i+1, dtype=int)
    catchment_outlets[0:i] = old_outlets
    catchment_outlets[i] = moulin_ii

print('Outlets:', catchment_outlets)

moulin_dx = np.vstack(meshx[catchment_outlets]) - meshx[catchment_outlets]
moulin_dy = np.vstack(meshy[catchment_outlets]) - meshy[catchment_outlets]
moulin_dd = np.sqrt(moulin_dx**2 + moulin_dy**2)
moulin_dd[moulin_dd==0] = np.nan
print('Moulin distances:', np.nanmin(moulin_dd))


fig, ax = plt.subplots()
ax.plot(zc, n_moulins, color='b', label='Moulins')
ax2 = ax.twinx()
ax2.plot(zc, areas, color='r', label='Area')
ax.grid()
ax.set_xlabel('Elevation (m asl.)')
ax.set_ylabel('Moulins')
ax2.set_ylabel('Area (m$^2$)')
ax.legend()

fig.savefig('moulins.png', dpi=400)

fig, ax = plt.subplots(figsize=(8, 4))
ax.tripcolor(mtri, surf, vmin=0, vmax=2500)
ax.plot(meshx[moulin_indices]/1e3, meshy[moulin_indices]/1e3, linestyle='', marker='x', color='r')
ax.set_aspect('equal')
fig.savefig('moulin_map.png', dpi=400)


fig, ax = plt.subplots(figsize=(8, 3))
ax.tripcolor(mtri, catchment_nums, vmin=0, vmax=len(moulin_indices), cmap='YlGnBu_r',
    edgecolor='w', linewidth=0.01)
ax.plot(meshx[moulin_indices]/1e3, meshy[moulin_indices]/1e3, 
    linestyle='', marker='.', color='k', markersize=5)
ax.plot(meshx[catchment_outlets]/1e3, meshy[catchment_outlets]/1e3,
    linestyle='', marker='x', color='r', markersize=5, linewidth=1)
ax.set_aspect('equal')
ax.set_xlim([0, 100])
ax.set_ylim([0, 25])

fig.savefig('catchment_map.png', dpi=400)

catchment_info = []
for i in range(len(moulin_indices)):
    cinfo = {}
    cinfo['area'] = np.sum(area_ele[catchment_nums==i])
    cinfo['elements'] = np.where(catchment_nums==i)[0]
    cinfo['moulin'] = catchment_outlets[i]
    cinfo['area_units'] = 'km2'
    catchment_info.append(cinfo)

with open('moulins_catchments.pkl', 'wb') as outfile:
    pickle.dump(catchment_info, outfile)

# CHECKS
num_eles = np.sum([len(ci['elements']) for ci in catchment_info])
print('Number of catchment elements:', num_eles)
print('Number of elements below 5 km:', len(surf_ele[elx<5e3]))
print('Total number of elements:', len(surf_ele))