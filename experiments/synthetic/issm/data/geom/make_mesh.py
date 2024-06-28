"""
Make ISSM meshes
"""
import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion
from model import model
from meshconvert import meshconvert
from solve import solve
from setmask import setmask
from parameterize import parameterize
from triangle import *
from bamg import *
from write_netCDF import write_netCDF
from plotmodel import *

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec

import cmocean

## Point to domain outline file
outline = 'synthetic_outline.exp'
meshfile = 'synthetic_mesh.nc'
min_length = 750

## Mesh characteristics
md = model()
md = triangle(md, outline, min_length)
print('Made mesh with numberofvertices:', md.mesh.numberofvertices)

if os.path.exists(meshfile):
    os.remove(meshfile)
write_netCDF(md, meshfile)

fig, ax = plt.subplots(figsize=(8, 3))
mtri = Triangulation(md.mesh.x, md.mesh.y, md.mesh.elements-1)
ax.tripcolor(mtri, 0*md.mesh.x, facecolor='none', edgecolor='k')
ax.set_aspect('equal')
ax.set_xlim([0, 100e3])
ax.set_ylim([0, 25e3])
fig.savefig('synthetic_mesh.png', dpi=600)