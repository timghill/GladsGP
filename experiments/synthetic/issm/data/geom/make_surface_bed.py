""" Compute synthetic surface and bed elevations"""

import os
import sys
import pickle

import numpy as np

with open('synthetic_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)

surf = 6*( np.sqrt(mesh['x'] + 5e3) - np.sqrt(5e3)) + 390
bed = 350*np.ones_like(mesh['x'])

np.save('synthetic_surface.npy', surf)
np.save('synthetic_bed.npy', bed)

