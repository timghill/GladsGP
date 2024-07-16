"""
Make ISSM meshes
"""
import os
import sys
import pickle
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
from plotmodel import *
from GetAreas import GetAreas

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec

import cmocean

from src.utils import reorder_edges

## Point to domain outline file
outline = 'synthetic_outline.exp'
meshfile = 'synthetic_mesh.pkl'
min_length = 750

## Mesh characteristics
md = model()
md = triangle(md, outline, min_length)
print('Made mesh with numberofvertices:', md.mesh.numberofvertices)

# Compute the nodes connected to each edge
connect_edge = reorder_edges(md)

# Compute edge lengths
x0 = md.mesh.x[connect_edge[:,0]]
x1 = md.mesh.x[connect_edge[:,1]]
dx = x1 - x0
y0 = md.mesh.y[connect_edge[:,0]]
y1 = md.mesh.y[connect_edge[:,1]]
dy = y1 - y0
edge_length = np.sqrt(dx**2 + dy**2)

print(md.mesh)
areas = GetAreas(md.mesh.elements, md.mesh.x, md.mesh.y)


if os.path.exists(meshfile):
    os.remove(meshfile)

meshdict = {}
meshdict['x'] = md.mesh.x
meshdict['y'] = md.mesh.y
meshdict['elements'] = md.mesh.elements
meshdict['connect_edge'] = connect_edge
meshdict['edge_length'] = edge_length
meshdict['area'] = areas
meshdict['vertexonboundary'] = md.mesh.vertexonboundary
meshdict['numberofelements'] = md.mesh.numberofelements
meshdict['numberofvertices'] = md.mesh.numberofvertices

# Write dictionary
with open(meshfile, 'wb') as mesh:
    pickle.dump(meshdict, mesh)


fig, ax = plt.subplots(figsize=(8, 3))
mtri = Triangulation(md.mesh.x, md.mesh.y, md.mesh.elements-1)
ax.tripcolor(mtri, 0*md.mesh.x, facecolor='none', edgecolor='k')
ax.set_aspect('equal')
ax.set_xlim([0, 100e3])
ax.set_ylim([0, 25e3])
fig.savefig('synthetic_mesh.png', dpi=600)