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
    vel_file='GrIMP_multiyear_vel_mosaic_vv.tif'):

    vel_fid = rs.open(vel_file)

    vv = vel_fid.read(1)

    xmin = vel_fid.bounds.left
    xmax = vel_fid.bounds.right
    ymin = vel_fid.bounds.bottom
    ymax = vel_fid.bounds.top
    nrows,ncols = vel_fid.shape

    vel_fid.close()

    x = np.linspace(xmin, xmax, ncols+1)[:-1]
    y = np.linspace(ymin, ymax, nrows+1)[0:-1][::-1]

    points = (x, y)

    vv_interp = interpolate.interpn(points, vv.T, xy, bounds_error=False, fill_value=-9999)

    return vv_interp

if __name__=='__main__':
    md = read_netCDF('../geom/IS_bamg.nc')
    xi = np.array([md.mesh.x, md.mesh.y]).T
    vv = interp_surf_bed(xi)
    np.save('IS_vel.npy', vv)
    
