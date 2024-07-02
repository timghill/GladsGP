#import os
#import sys
#ISSM_DIR = os.getenv('ISSM_DIR')
#sys.path.append(os.path.join(ISSM_DIR, 'src/m/netcdf/'))
#import devpath

from src.read_netCDF import read_netCDF
from src.write_netCDF import write_netCDF

def read_nc(fname):
    md = read_netCDF(fname)
    return md

def write_nc(md, fname):
    write_netCDF(md, fname)
    return fname
