""" Compute synthetic surface and bed elevations"""

import os
import sys

import numpy as np

# Import ISSM paths
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from read_netCDF import read_netCDF

md = read_netCDF('synthetic_mesh.nc')

surf = 6*( np.sqrt(md.mesh.x + 5e3) - np.sqrt(5e3)) + 390
bed = 350*np.ones_like(md.mesh.x)

np.save('synthetic_surface.npy', surf)
np.save('synthetic_bed.npy', bed)