"""
Make ISSM meshes
"""
import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
# sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
# sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
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
# matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean
# from save import *

from make_surface_bed import interp_surf_bed

## Point to domain outline file
outline = 'IS_outline.exp'
meshfile = 'IS_mesh.nc'
min_length = 500
max_length = 5e3

min_elev = 1000
ELA_elev = 2000

## Mesh characteristics

## Make a draft mesh to interpolate surface elevations
md = model()
md = bamg(md, 'domain', 'IS_outline.exp', 'hmin', max_length, 'hmax', max_length, 'anisomax', 1.1)
print('Made draft mesh with numberofvertices:', md.mesh.numberofvertices)

## Elevation-dependent refinement
elev, bed = interp_surf_bed(np.array([md.mesh.x, md.mesh.y]).T)
area = min_length + (max_length - min_length)*(elev-min_elev)/(ELA_elev-min_elev)
area[elev<min_elev] = min_length
area[elev>ELA_elev] = max_length

## Make the refined mesh
md = bamg(md, 'hVertices', area, 'anisomax', 1.1)
print('Refined mesh to have numberofvertices:', md.mesh.numberofvertices)
if os.path.exists(meshfile):
    os.remove(meshfile)
write_netCDF(md, meshfile)
