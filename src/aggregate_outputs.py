"""
Collect outputs from individual simulations into .npy arrays.

Command line interface

    python -m src.aggregate_outputs issm [config] [njobs] [--all]
"""

import os
import sys
import importlib
import argparse
import pickle

import numpy as np

from . import definitions

def collect_issm_results(config, njobs, dtype=np.float32, save_all=False):
    """
    Collect outputs from individual simulations into .npy arrays.

    Parameters
    ----------
    config : module
             Imported configuration file
    
    njobs : int
            Number of jobs to look for in the results directory
    
    dtype : type
            Data type to cast outputs into. Recommend np.float32
    
    save_all : bool
               If save_all is True, save values for all scalar
               variable definitions in addition to the mean.
               If save_all is False, save values for only
               the mean of each variable across definitions.
    """

    # Definitions for special quantities of interest
    fluxgate_positions = np.arange(5e3, 35e3, 5e3)
    R_thresholds = np.array([0.5, 1.])
    S_thresholds = 0.5*np.pi*R_thresholds**2
    
    with open(config.mesh, 'rb') as meshin:
        mesh = pickle.load(meshin)
    
    nodes = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements'].astype(int)-1
    connect_edge = mesh['connect_edge'].astype(int)
    edge_length = mesh['edge_length']

    # Construct file patterns
    jobids = np.arange(1, njobs+1)

    resdir = 'RUN/output_{:03d}/'
    respattern = os.path.join(resdir, '{}.npy')
    aggpattern = '{exp}_{}.npy'.format('{}', exp=config.exp)

    all_ff = np.zeros((len(nodes)*365, njobs), dtype=dtype)
    all_channel_frac = np.zeros((len(fluxgate_positions), njobs), dtype=dtype)
    all_channel_length = np.zeros((len(R_thresholds), njobs), dtype=dtype)
    all_transit_time = np.zeros((len(fluxgate_positions), njobs), dtype=dtype)
    for i,jobid in enumerate(jobids):
        print('Job %d' % jobid)
        ff = np.load(respattern.format(jobid, 'ff'))
        all_ff[:, i] = ff.flatten()

        Q = np.load(respattern.format((jobid), 'Q'))
        S = np.load(respattern.format((jobid), 'S'))
        vx = np.load(respattern.format((jobid), 'vx'))/365/86400
        hs = np.load(respattern.format((jobid), 'h_s'))
        q = np.abs(vx*hs)
        
        # Fraction of channelized discharge
        channel_discharge_timeseries = definitions.channel_discharge(nodes, connect_edge, 
            np.abs(Q), fluxgate_positions)
        sheet_discharge_timeseries = definitions.sheet_discharge(nodes,
            q, fluxgate_positions)

        channel_discharge = np.sum(channel_discharge_timeseries, axis=1)
        sheet_discharge = np.sum(sheet_discharge_timeseries, axis=1)
        channel_frac = channel_discharge/(channel_discharge + sheet_discharge)
        # channel_frac = channel_discharge
        all_channel_frac[:, i] = channel_frac

        # Channel network length
        channel_length_ts = definitions.channel_network_length(edge_length,
            S, S_thresholds)
        channel_length = np.max(channel_length_ts, axis=1)/1e3
        all_channel_length[:, i] = channel_length

        # Sheet transit time
        melt_start = 154
        melt_end   = 266

        T = definitions.sheet_transit_time(nodes, np.abs(vx), 
            fluxgate_positions)
        T_meltseason = np.mean(T[:, melt_start:melt_end], axis=1)/365/86400
        T_winter = np.mean(T[:, melt_start-30:melt_start-15], axis=1)/365/86400
        T_mean = np.mean(T, axis=1)/365/86400
        all_transit_time[:, i] = np.log10(T_mean)
    
    # Compute average of channel frac and log transit time
    append_channel_frac = np.zeros((all_channel_frac.shape[0]+1, all_channel_frac.shape[1]))
    append_transit_time = np.zeros((all_transit_time.shape[0]+1, all_transit_time.shape[1]))

    append_channel_frac[:all_channel_frac.shape[0], :] = all_channel_frac
    append_channel_frac[-1,:] = np.mean(all_channel_frac, axis=0)
    append_transit_time[:all_transit_time.shape[0], :] = all_transit_time
    append_transit_time[-1, :] = np.mean(all_transit_time, axis=0)

    np.save(aggpattern.format('ff'), all_ff)

    if save_all:
        save_channel_frac = append_channel_frac
        save_channel_length = all_channel_length
        save_transit_time = all_transit_time
    else:
        save_channel_frac = append_channel_frac[-1].squeeze()
        save_channel_length = all_channel_length[0]
        save_transit_time = all_transit_time[-1]

    np.save(aggpattern.format('channel_frac'), save_channel_frac)
    np.save(aggpattern.format('channel_length'), save_channel_length)
    np.save(aggpattern.format('log_transit_time'), save_transit_time)


def main():
    """
    Command-line interface to collect simulation outputs
    """
    desc = 'Collect simulation outputs'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('version', help='"matlab" or "issm" GlaDS')
    parser.add_argument('config', help='Path to experiment config file')
    parser.add_argument('njobs', help='Number of jobs', type=int)
    parser.add_argument('--all', action='store true')
    args = parser.parse_args()

    if args.version not in ['matlab', 'issm']:
        raise ValueError('version must be one of "matlab" or "issm"')
    elif args.version=='matlab':
        raise NotImplementedError('"matlab" version is not implemented')

    if not os.path.exists(args.config):
        raise OSError('Configuration file "{}" does not exist'.format(args.config))
    
    path, name = os.path.split(args.config)
    if path:
        abspath = os.path.abspath(path)
        sys.path.append(abspath)
    module, ext = os.path.splitext(name)
    config = importlib.import_module(module)
    if args.version=='issm':
        collect_issm_results(config, args.njobs)
    return

if __name__=='__main__':
    main()
