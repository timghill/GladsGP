"""
Integrate surface melt within catchments
"""

import os
import sys
import pickle

import numpy as np
import cmocean
from matplotlib import pyplot as plt
import matplotlib

# Constants
lapse_rate = -5e-3  # deg C m-1
DDF = 0.01/86400    # 0.01 m w.e./K/day

# Read sea-level temp
temparr = np.loadtxt('KAN_L_2014_temp_clipped.txt', delimiter=',')
tt = temparr[:, 0]
temp = temparr[:, 1]


# Read elevation
elev = np.load('../geom/synthetic_surface.npy')
# md = read_netCDF('../geom/synthetic_mesh.pkl')
with open('../geom/synthetic_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)
elev_els = np.mean(elev[mesh['elements']-1], axis=1)

T_distributed = temp + lapse_rate*np.vstack(elev_els)
melt_distributed = DDF*T_distributed
melt_distributed[T_distributed<0] = 0
print(melt_distributed.shape)

# Read catchments
with open('../moulins/moulins_catchments.pkl', 'rb') as fid:
    basins = pickle.load(fid)

# area_ele = GetAreas(md.mesh.elements, md.mesh.x, md.mesh.y)
area_ele = mesh['area']

n_basins = len(basins)
surf_inputs = np.zeros((n_basins, len(temp)))
for i in range(n_basins):
    areas = area_ele[basins[i]['elements']]
    meltrate = melt_distributed[basins[i]['elements']]
    basin_inputs = np.sum(np.vstack(areas)*meltrate, axis=0)
    surf_inputs[i,:] = basin_inputs

fig, ax = plt.subplots()
ax.plot(tt, temp)
ax.grid()
ax.set_xlabel('Day of year')
ax.set_ylabel('Sea-level temperature (C)')
fig.savefig('sea_level_temperature.png', dpi=400)

[xx, ll] = np.meshgrid(tt, np.arange(len(area_ele)))
[_, bb] = np.meshgrid(tt, np.arange(n_basins))

fig, ax = plt.subplots()
pc = ax.pcolormesh(xx, ll, melt_distributed*86400, cmap=cmocean.cm.rain)
cbar = fig.colorbar(pc)
cbar.set_label('Melt rate (m w.e./day)')
ax.set_title('Element melt rates')
ax.set_xlabel('Day of year')
ax.set_ylabel('Element index')
fig.savefig('element_melt_rates', dpi=400)

fig, ax = plt.subplots()
pc = ax.pcolormesh(tt, bb, surf_inputs, cmap=cmocean.cm.rain)
cbar = fig.colorbar(pc)
cbar.set_label('Surface inputs (m$^3$/s)')
ax.set_title('Integrated catchment inputs')
ax.set_xlabel('Day of year')
ax.set_ylabel('Basin index')
fig.savefig('basin_integrated_inputs.png', dpi=400)

issm_inputs = np.zeros((surf_inputs.shape[0]+1, surf_inputs.shape[1]))
issm_inputs[:-1] = surf_inputs
issm_inputs[-1, :] = tt

np.savetxt('basin_integrated_inputs.csv', issm_inputs, delimiter=',', fmt='%.6e')
