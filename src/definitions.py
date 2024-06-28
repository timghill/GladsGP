"""
Reference definitions of GlaDS quantities of interest:

    channel_discharge
    
    sheet_discharge
    
    channel_network_length
    
    sheet_transit_time
"""

import numpy as np
from scipy import interpolate

def channel_discharge(nodes, connect_edge, Q, flux_gates):
    """
    Channel discharge across flux gates.

    Parameters
    ----------
    nodes : (numberofvertices, 2) array
            (x, y) coordinates of mesh nodes

    connect_edge : (numberofedges, 2) array
                   Node indices connected to each mesh edge

    Q : (numberofvertices, nt) array
        Array of absolute channel discharge. Note: default ISSM
        units are m3.s-1

    flux_gates : (n_flux_gates,) array
                 Flux gate positions (specified in meters)

    Returns
    -------
    discharge : (n_flux_gates, n_time)
                Timeseries of channel discharge across each flux gate,
                in the same units as Q (recommended m.s-3)
    """
    discharge = np.zeros((len(flux_gates), Q.shape[1]))
    for i_edge in range(len(connect_edge)):
        for i_xpos in range(len(flux_gates)):
            xb = flux_gates[i_xpos]
            xnodes = nodes[connect_edge[i_edge, :], 0]
            if xnodes[0]>xb and xnodes[1]<=xb:
                discharge[i_xpos, :] = discharge[i_xpos, :] + Q[i_edge, :]
            if xnodes[0]<=xb and xnodes[1]>xb:
                discharge[i_xpos, :] = discharge[i_xpos, :] + Q[i_edge, :]
    return discharge

def sheet_discharge(nodes, q, flux_gates):
    """
    Sheet discharge integrated across flux gates.

    Parameters
    ----------
    nodes : (numberofvertices, 2) array
            Array of (x, y) coordinates of mesh nodes

    q : (numberofvertices, nt) array
        Sheet flux. Note: if q is computed from HydrologyWaterVx, ...Vy,
        the units will be m2.day-1

    flux_gates : (n_flux_gates,) array
                 Flux gate positions (specified in meters)

    Returns
    -------
    discharge : (n_flux_gates, n_time)
                Timeseries of sheet discharge integrated across each flux gate,
                in the same units as q
    """
    discharge = np.zeros((len(flux_gates), q.shape[1]))
    for i in range(len(flux_gates)):
        xb = flux_gates[i]
        xb_mask = np.abs(nodes[:, 0]-xb)<2.5e3
        F = interpolate.LinearNDInterpolator(nodes[xb_mask, :],
            q[xb_mask, :])
        dy = 50
        yq = np.vstack(np.arange(0, 25e3, dy))
        xq = 0*yq + xb
        q_regular = F(xq, yq)
        discharge[i, :] = np.sum(q_regular*dy, axis=0)
    return discharge

def channel_network_length(edge_length, Z, thresholds):
    """
    Total length of subglacial channel network.

    Compute total subglacial channel network length for an arbitrary
    threshold (i.e., area or discharge).

    Parameters
    ----------
    edge_length : (numberofedges,) array
                  Length of each edge in mesh
    
    Z : (numberofedges, n_time)
        Quantity to use to threshold edge length calculation
        (e.g., discharge or area)
    
    thresholds : (n,) array
                 Values to use as thresholds on Z to define channel
                 network
    
    Returns
    -------
    length : (n_time) array
    """
    network_length = np.zeros((len(thresholds), Z.shape[1]))
    for i in range(len(thresholds)):
        edge_len = np.vstack(edge_length)*np.ones(Z.shape)
        edge_len[Z<thresholds[i]] = 0
        network_length[i, :] = np.sum(edge_len, axis=0)
    return network_length

def sheet_transit_time(nodes, v, flux_gates):
    """
    Instantaneous sheet transit time below specified flux gates

    Parameters
    ----------
    nodes : (numberofvertices, 2) array
            (x, y) coordinates of mesh nodes
    
    v : (numberofvertices, n_time) array
        Sheet-flow velocity. Note: default ISSM units are m.day-1
    
    flux_gates : (n_flux_gates,) array
                 Flux gate positions (specified in meters)
    
    Returns
    -------
    transit_time: (n_flux_gates, n_time) array
    """
    transit_time = np.zeros((len(flux_gates), v.shape[1]))
    for i in range(len(flux_gates)):
        xb = flux_gates[i]
        dx = 1e3
        xleft = np.arange(0, xb, dx)
        for j in range(len(xleft)):
            mask = np.logical_and(nodes[:, 0]>=xleft[j],
                nodes[:, 0]<(xleft[j]+dx))
            vel_mean = np.mean(v[mask, :], axis=0)
            transit_time[i, :] = transit_time[i, :] + dx/vel_mean
    return transit_time