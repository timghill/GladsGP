"""
Interpolate topography onto the triangular mesh
"""

import numpy as np
from scipy import interpolate
from scipy import signal
import rasterio as rs

import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
# sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
# sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
# from issmversion import issmversion
# from model import model
# from meshconvert import meshconvert
# from solve import solve
# from setmask import setmask
# from parameterize import parameterize
# from triangle import *
# from bamg import *
from write_netCDF import write_netCDF
from read_netCDF import read_netCDF
# from plotmodel import *
# from matplotlib import pyplot as plt

def interp_surf_bed(xy, 
    surface_file='BedMachineGreenland_StudyArea_surface.tif',
    bed_file='BedMachineGreenland_StudyArea_bed.tif'):

    surf_fid = rs.open(surface_file)
    bed_fid = rs.open(bed_file)

    surf = surf_fid.read(1)
    bed = bed_fid.read(1)

    xmin = surf_fid.bounds.left
    xmax = surf_fid.bounds.right
    ymin = surf_fid.bounds.bottom
    ymax = surf_fid.bounds.top
    nrows,ncols = surf_fid.shape

    surf_fid.close()
    bed_fid.close()

    x = np.linspace(xmin, xmax, ncols+1)[:-1]
    y = np.linspace(ymin, ymax, nrows+1)[0:-1][::-1]

    # [xx,yy] = np.meshgrid(x,y)
    points = (x, y)

    surf_interp = interpolate.interpn(points, surf.T, xy, bounds_error=False, fill_value=-9999)
    bed_interp = interpolate.interpn(points, bed.T, xy, bounds_error=False, fill_value=-9999)

    return surf_interp, bed_interp

if __name__=='__main__':
    md = read_netCDF('IS_mesh.nc')
    xi = np.array([md.mesh.x, md.mesh.y]).T
    surf,bed = interp_surf_bed(xi)
    np.save('IS_surface.npy', surf)
    np.save('IS_bed.npy', bed)
    
