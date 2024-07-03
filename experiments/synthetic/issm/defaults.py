import pickle

import numpy as np

import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, '/bin'))
sys.path.append(os.path.join(ISSM_DIR, '/lib'))
from issmversion import issmversion
from hydrologyglads import hydrologyglads
from generic import generic

# Vectors for convenience
xvec = md.mesh.x
onevec = 0*xvec + 1

# Calving
md.calving.calvingrate=0*onevec

# # Friction - need to specify but not used
md.friction.coefficient = onevec
md.friction.p = np.ones((md.mesh.numberofelements, 1))
md.friction.q = np.ones((md.mesh.numberofelements, 1))

# Geometry
bed = 350 + 0*xvec
surf = 390 + 6*(np.sqrt(xvec+5e3) - np.sqrt(5e3))
thick = surf - bed
md.geometry.base = bed
md.geometry.bed =bed
md.geometry.surface = surf
md.geometry.thickness = thick

# Constants
md.materials.rheology_B = (2.4e-24)**(-1/3)*onevec
md.initialization.temperature = (273)*onevec
md.materials.rheology_n = 3
md.materials.rho_freshwater = 1e3
md.materials.rho_ice = 910
md.materials.mu_water = md.materials.rho_freshwater * 1.793e-6
md.constants.g = 9.8

# HYDROLOGY
# parameters
md.hydrology = hydrologyglads()
md.hydrology.sheet_conductivity = 0.05*onevec
md.hydrology.sheet_alpha = 3./2.
md.hydrology.sheet_beta = 2.0
md.hydrology.cavity_spacing = 10
md.hydrology.bump_height = 0.5*onevec
md.hydrology.channel_sheet_width = md.hydrology.cavity_spacing
md.hydrology.omega = 1/2000
md.hydrology.englacial_void_ratio = 1e-4
md.hydrology.rheology_B_base = (2.4e-24)**(-1./3.)*onevec
md.hydrology.istransition = 1
md.hydrology.ischannels = 1
md.hydrology.channel_conductivity = 0.5*onevec
md.hydrology.channel_alpha = 5./4.
md.hydrology.channel_beta = 3./2.
md.hydrology.creep_open_flag = 0
# md.hydrology.requested_outputs = ['default']
md.hydrology.requested_outputs = [
        'HydraulicPotential',
        'EffectivePressure',
        'HydrologySheetThickness',
        'ChannelDischarge',
        'ChannelArea',
        # 'HydrologySheetDischarge',
        'HydrologyWaterVx',
        'HydrologyWaterVy',
]

# INITIAL CONDITIONS
md.initialization.watercolumn = 0.2*md.hydrology.bump_height*onevec
md.initialization.channelarea = 0*np.zeros((md.mesh.numberofedges, 1))

phi_bed = md.constants.g*md.materials.rho_freshwater*md.geometry.base
p_ice = md.constants.g*md.materials.rho_ice*md.geometry.thickness
md.initialization.hydraulic_potential = phi_bed + p_ice

md.initialization.ve = 30*onevec
md.initialization.vx = -30*onevec
md.initialization.vy = 0*onevec

# BOUNDARY CONDITIONS
md.hydrology.spcphi = np.nan*onevec
pos = np.where(np.logical_and(
    md.mesh.vertexonboundary,
    md.mesh.x==np.min(md.mesh.x, axis=-1)))
md.hydrology.spcphi[pos] = phi_bed[pos]
md.hydrology.neumannflux = np.zeros((md.mesh.numberofelements, 1))

# FORCING

with open('../data/moulins/moulins_catchments.pkl', 'rb') as moulins_file:
    basins = pickle.load(moulins_file)
moulin_indices = np.array([basin['moulin'] for basin in basins])

md.hydrology.melt_flag = 1
md.basalforcings.groundedice_melting_rate = 0.05*onevec
md.basalforcings.geothermalflux = 0
# Read one year of moulin inputs from file and repeat as many times as necessary
moulin_inputs = np.loadtxt('../data/melt/basin_integrated_inputs.csv', delimiter=',')
tt_days = moulin_inputs[-1, :].astype(int)
moulin_input_rate = moulin_inputs[:-1, :]

# # Steady moulin inputs for testing
# mean_moulin_input = np.mean(moulin_input_rate, axis=1)
# md.hydrology.moulin_input = np.zeros(md.mesh.numberofvertices)
# md.hydrology.moulin_input[moulin_indices] = mean_moulin_input

n_reps = 2
dt = 1/365
tt_years = np.arange(0, 2+dt, dt)
tt_seconds = tt_years * md.constants.yts
md.hydrology.moulin_input = np.zeros((md.mesh.numberofvertices+1, len(tt_seconds)))
for i in range(n_reps):
    md.hydrology.moulin_input[np.vstack(moulin_indices), tt_days + i*365] = moulin_inputs[:-1, :]
md.hydrology.moulin_input[-1] = tt_years

# TOLERANCES
md.stressbalance.restol = 1e-3
md.stressbalance.reltol = np.nan
md.stressbalance.abstol = np.nan
md.stressbalance.maxiter = 100

# TIMESTEPPING
hour = 3600
day = 86400
dt_hours = 2
out_freq = 24*1/dt_hours
# out_freq = 1
md.timestepping.time_step = dt_hours*hour/md.constants.yts
md.timestepping.final_time = 2

md.settings.output_frequency = out_freq

#md.settings.interp_forcing = True
md.settings.cycle_forcing = True

md.transient.deactivateall()
md.transient.ishydrology = True

md.verbose.solution = True
md.miscellaneous.name = 'output'

md.cluster = generic('np', 1)

SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
if SLURM_TMPDIR:
    md.cluster.executionpath = SLURM_TMPDIR
else:
    cwd = os.getcwd()
    expath = os.path.join(cwd, 'TMP/')
    if not os.path.exists(expath):
        os.makedirs(expath)
    md.cluster.executionpath = expath

print(md.cluster.executionpath)
