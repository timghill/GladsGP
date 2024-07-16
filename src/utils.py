"""
Extra tools and utils for GP emulation project
"""

from pydoc import importfile

import time

import numpy as np
from scipy import stats

def import_config(conf_path):
    """
    Import a configuration file from path conf_path

    Parameters
    ----------
    conf_path : str
                Path to a valid python *.py file with GP model specification
    
    Returns
    -------
    module instance with configuration file globals
    """
    return importfile(conf_path)

def saltelli_sensitivity_indices(func, n_dim, m, bootstrap=True):
    """
    Compute first-order and total sensitivity indices following Saltelli et al.

    This function mimics scipy.stats.sobol_indices but is faster (at least a
    factor of ~5x) because it assumes the shape of the data.

    Parameters
    ----------
    func : callable
           Function with call signature
               func(x: (N, n_dim) ) --> (N, p)
           Where N=2**m is the number of points and p is
           the dimensionality of the output of a single func
           calculation (i.e., the number of PCs)
    
    n_dim : int
            Number of input dimensions
    
    m : int
        Use 2**m samples
    
    bootstrap: bool
               If bootstrap is True, use bootstrap resampling to
               compute 95% confidence intervals on sensitivity indices
    
    See also: scipy.stats.sobol_indices
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sobol_indices.html#scipy.stats.sobol_indices
    """
    t0 = time.perf_counter()
    sampler = stats.qmc.Sobol(d=2*n_dim)
    N = 2**5
    AB = sampler.random_base2(m=int(m))
    A = AB[:, n_dim:]
    B = AB[:, :n_dim]
    f_A = func(A)
    f_B = func(B)
    first_order = np.zeros((f_A.shape[1], n_dim))
    total_index = np.zeros((f_A.shape[1], n_dim))
    print('f_A.shape:', f_A.shape)
    f_AB = np.zeros((n_dim, f_A.shape[0], f_A.shape[1]))
    print('f_AB.shape:', f_AB.shape)
    var = np.var([f_A, f_B], axis=(0, 1))
    for ix in range(n_dim):
        C = B.copy()
        C[:, ix] = A[:, ix]
        f_C = func(C)
        f_AB[ix] = f_C

        V_i = np.mean(f_A*(f_C - f_B) , axis=0)
        E_i = 0.5*np.mean( (f_B - f_C)**2, axis=0)
        S_i = V_i/var
        T_i = E_i/var
        first_order[:, ix] = S_i
        total_index[:, ix] = T_i
    
    t1 = time.perf_counter()
    print('\tElapsed time:', t1-t0)

    if bootstrap:
        def first_order_statistic(arg):
            f_A_ = f_A[arg, :]
            f_B_ = f_B[arg, :]
            f_AB_ = f_AB[:, arg, :]
            V_ix = np.mean(f_A_*(f_AB_ - f_B_), axis=(1,))
            V_ix[V_ix<0] = 0
            # print('V_ix.shape', V_ix.shape)
            var = np.var([f_A_, f_B_], axis=(0, 1))
            # print('var:', var.shape)
            return (V_ix/var).T
        
        def total_index_statistic(arg):
            f_A_ = f_A[arg, :]
            f_B_ = f_B[arg, :]
            f_AB_ = f_AB[:, arg, :]
            # V_ix = np.mean(f_A_*(f_AB_ - f_B_), axis=(1,))
            E_ix = 0.5*np.mean( (f_B_ - f_AB_)**2, axis=(1,))
            E_ix[E_ix<0] = 0
            # print('V_ix.shape', V_ix.shape)
            var = np.var([f_A_, f_B_], axis=(0, 1))
            # print('var:', var.shape)
            return (E_ix/var).T

        res_first = stats.bootstrap([np.arange(2**m)], first_order_statistic,
            n_resamples=9999)
        res_total = stats.bootstrap([np.arange(2**m)], total_index_statistic,
            n_resamples=9999)
        res = { 'first_order':res_first, 
                'total_index':res_total,
        }
    else:
        res = None
    return (first_order, total_index, res)


def PCA_saltelli_sensitivity_indices(func, n_dim, m, pcvar, bootstrap=True):
    """
    Compute first-order and total sensitivity indices following Saltelli et al.

    This function mimics scipy.stats.sobol_indices but is faster (at least a
    factor of ~5x) because it assumes the shape of the data.

    Parameters
    ----------
    func : callable
           Function with call signature
               func(x: (N, n_dim) ) --> (N, p)
           Where N=2**m is the number of points and p is
           the dimensionality of the output of a single func
           calculation (i.e., the number of PCs)
    
    n_dim : int
            Number of input dimensions
    
    m : int
        Use 2**m samples
    
    bootstrap: bool
               If bootstrap is True, use bootstrap resampling to
               compute 95% confidence intervals on sensitivity indices
    
    See also: scipy.stats.sobol_indices
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sobol_indices.html#scipy.stats.sobol_indices
    """
    t0 = time.perf_counter()
    sampler = stats.qmc.Sobol(d=2*n_dim)
    N = 2**5
    AB = sampler.random_base2(m=int(m))
    A = AB[:, n_dim:]
    B = AB[:, :n_dim]
    f_A = func(A)
    f_B = func(B)
    first_order = np.zeros((f_A.shape[1], n_dim))
    total_index = np.zeros((f_A.shape[1], n_dim))
    print('f_A.shape:', f_A.shape)
    f_AB = np.zeros((n_dim, f_A.shape[0], f_A.shape[1]))
    print('f_AB.shape:', f_AB.shape)
    var = np.var([f_A, f_B], axis=(0, 1))
    for ix in range(n_dim):
        C = B.copy()
        C[:, ix] = A[:, ix]
        f_C = func(C)
        f_AB[ix] = f_C

        V_i = np.mean(f_A*(f_C - f_B) , axis=0)
        E_i = 0.5*np.mean( (f_B - f_C)**2, axis=0)
        S_i = V_i/var
        T_i = E_i/var
        first_order[:, ix] = S_i
        total_index[:, ix] = T_i
    
    t1 = time.perf_counter()
    print('\tElapsed time:', t1-t0)

    if bootstrap:
        def first_order_statistic(arg):
            f_A_ = f_A[arg, :]
            f_B_ = f_B[arg, :]
            f_AB_ = f_AB[:, arg, :]
            V_ix = np.mean(f_A_*(f_AB_ - f_B_), axis=(1,))
            V_ix[V_ix<0] = 0
            # print('V_ix.shape', V_ix.shape)
            var = np.var([f_A_, f_B_], axis=(0, 1))
            # print('var:', var.shape)
            return (V_ix/var).T
        
        def total_index_statistic(arg):
            f_A_ = f_A[arg, :]
            f_B_ = f_B[arg, :]
            f_AB_ = f_AB[:, arg, :]
            # V_ix = np.mean(f_A_*(f_AB_ - f_B_), axis=(1,))
            E_ix = 0.5*np.mean( (f_B_ - f_AB_)**2, axis=(1,))
            E_ix[E_ix<0] = 0
            # print('V_ix.shape', V_ix.shape)
            var = np.var([f_A_, f_B_], axis=(0, 1))
            # print('var:', var.shape)
            return (E_ix/var).T
        
        def general_first_order_statistic(arg):
            pc_indices = first_order_statistic(arg)
            return np.sum(pc_indices*np.vstack(pcvar), axis=0)
        
        def general_total_index_statistic(arg):
            pc_indices = total_index_statistic(arg)
            return np.sum(pc_indices*np.vstack(pcvar), axis=0)

        res_first = stats.bootstrap([np.arange(2**m)], first_order_statistic,
            n_resamples=9999)
        res_total = stats.bootstrap([np.arange(2**m)], total_index_statistic,
            n_resamples=9999)
        res_gen_first = stats.bootstrap([np.arange(2**m)], general_first_order_statistic,
            n_resamples=9999)
        res_gen_total = stats.bootstrap([np.arange(2**m)], general_total_index_statistic,
            n_resamples=9999)
        res = { 'first_order':res_first, 
                'total_index':res_total,
                'general_first_order':res_gen_first,
                'general_total_index':res_gen_total,
        }
    else:
        res = None
    
    gen_first = np.sum(first_order.T*pcvar, axis=1)
    gen_total = np.sum(total_index.T*pcvar, axis=1)
    return (first_order, total_index, gen_first, gen_total, res)


def width_average(mesh, x, dx=2):
    """Width-average (nx, nt) array x"""
    xedge = np.arange(0, 100+dx, dx)
    xmid = 0.5*(xedge[1:] + xedge[:-1])
    xavg = np.zeros((len(xmid), x.shape[1]))
    for i in range(len(xavg)):
        xi = xmid[i]
        mask = np.abs(mesh['x']/1e3 - xi)<dx/2
        xavg[i] = np.nanmean(x[mask,:],axis=0)
    return xavg, xedge

def reorder_edges(md):
    maxnbf = 3*md.mesh.numberofelements
    edges = np.zeros((maxnbf, 3)).astype(int)
    exchange = np.zeros(maxnbf).astype(int)

    head_minv = -1*np.ones(md.mesh.numberofvertices).astype(int)
    next_face = np.zeros(maxnbf).astype(int)
    nbf = 0
    for i in range(md.mesh.numberofelements):
        for j in range(3):
            v1 = md.mesh.elements[i,j]-1
            if j==2:
                v2 = md.mesh.elements[i,0]-1
            else:
                v2 = md.mesh.elements[i,j+1]-1
            
            if v2<v1:
                v3 = v2
                v2 = v1
                v1 = v3
            
            exists = False
            e = head_minv[v1]
            while e!=-1:
                if edges[e, 1]==v2:
                    exists=True
                    break
                e = next_face[e]
            
            if not exists:
                edges[nbf,0] = v1
                edges[nbf,1] = v2
                edges[nbf,2] = i
                if v1!=md.mesh.elements[i,j]-1:
                    exchange[nbf] = 1
                
                next_face[nbf] = head_minv[v1]
                head_minv[v1] = nbf
                nbf = nbf+1

    edges = edges[:nbf]
    pos = np.where(exchange==1)[0]
    v3 = edges[pos, 0]
    edges[pos, 0] = edges[pos, 1]
    edges[pos, 1] = v3

    edges = edges[:, :2]
    return edges