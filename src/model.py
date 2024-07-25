"""
Fit GP models for different subsets of training data and different
choices for the number of principal components.
"""

import argparse
import os
import time

import numpy as np
from numpy import linalg

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaParam

from src.utils import import_config
from src import svd

def init_model(t_std, y_sim, exp, p, data_dir='data/', 
    sd_threshold=1e-6, recompute=False):
    """
    Initialize SepiaData and SepiaModel instances,
    compute PC coefficients and basis and save
    to data directory

    Parameters
    ----------
    t_std : (n simulations, t_dim)
            Standardized simulation design matrix

    y_sim : (n_simulations, y_dim)
            Non-standardized simulation output matrix
    
    exp : str
          Name for simulation, used to generate file paths
    
    p : int
        Number of principal components to retain
    
    data_dir : str
               Directory to save PCA matrices to
    
    sd_threshold : float
                   Lower bound to standard deviation of simulation data
                   for scaling to avoid dividing by zero
    
    recompute : bool
                Force recompute SVD matrices
    
    Returns:
    --------
    SepiaData, SepiaModel
    """
    # Initialize SepiaData instance with our data
    y_ind_sim = np.linspace(0, 1, y_sim.shape[1])
    data = SepiaData(t_sim=t_std, y_sim=y_sim, y_ind_sim=y_ind_sim)

    # Compute a custom basis
    mu_y = np.mean(y_sim, axis=0)
    sd_y = np.std(y_sim, ddof=1, axis=0)
    # We have some nodes near the terminus that have no variation, so the
    # scaling produces nan's. Set a minimum standard deviation to avoid this
    sd_y[sd_y<sd_threshold] = sd_threshold

    # No transformations for x, t, they are already scaled in [0, 1]
    t_dim = t_std.shape[1]
    data.transform_xt(t_notrans=np.arange(t_dim))

    # Transform y with given mean, sd
    data.standardize_y(y_mean=mu_y, y_sd=sd_y)
    y_std = (y_sim - mu_y)/sd_y
    assert np.allclose(y_std, data.sim_data.y_std)

    # Compute PCA basis
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Check for PCA files, do not recompute them unless recompute==True
    pca_fpattern = os.path.join(data_dir, 'pca_{}_{}.npy')
    pca_fstat = np.array([os.path.exists(pca_fpattern.format(exp, arr)) for arr in ['U', 'S', 'Vh']])
    pmax = 25
    if recompute or not pca_fstat.all():
        # U,S,Vh = linalg.svd(y_std, full_matrices=False)
        U,S,Vh = svd.randomized_svd(y_std, pmax, k=0, q=1)
        U = U[:, :pmax]
        Vh = Vh[:pmax, :]
        np.save(pca_fpattern.format(exp, 'U'), U)
        np.save(pca_fpattern.format(exp, 'S'), S)
        np.save(pca_fpattern.format(exp, 'Vh'), Vh)

    # Always use saved SVD matrices
    U = np.load(pca_fpattern.format(exp, 'U'))
    S = np.load(pca_fpattern.format(exp, 'S'))
    Vh = np.load(pca_fpattern.format(exp, 'Vh'))
    S2 = S**2
    prop_var = S2/np.sum(S2)
    cumul_var = np.cumsum(S2)
    print('SVD proportion of variance:', prop_var[:10])

    # Construct the basis K, this is the scaling SEPIA uses by default
    K = np.diag(S[:p]) @ Vh[:p] / np.sqrt(y_sim.shape[0])
    data.create_K_basis(K=K)
    print('K.shape', K.shape)

    # Now make the SepiaModel instances since we are done with the data
    model = SepiaModel(data)
    return data,model

def load_model(train_config, m, p, dtype=np.float32):
    """
    Load SepiaData and SepiaModel instances from disk

    Parameters
    ----------
    train_config : module
                   Imported experiment configuration
    
    m : int
        Number of simultions
    
    p : int
        Number of princpal components
    
    dtype: type, optional (default np.float32)
           Data type to cast simulation to into. Accepts any
           valid numpy dtype=dtype.

    Returns:
    --------
    SepiaData, SepiaModel
    """
    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)[:m]

    y_sim = np.load(train_config.Y_physical).T.astype(dtype)[:m]

    data_dir = os.path.join(train_config.data_dir, 'models')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    models = []
    model_name = '{}_n{:03d}_p{:02d}'.format(train_config.exp, m, p)
    m_name = '{}_n{:03d}'.format(train_config.exp, m)
    model_path = os.path.join(data_dir, model_name)
    sepia_data, model = init_model(t_std=t_std, y_sim=y_sim, 
        exp=m_name, 
        p=p, data_dir=data_dir, recompute=False)
    print('Restoring from:', model_path)
    model.restore_model_info(model_path)
    return sepia_data, model

def fit_models(train_config, n_sims, n_pcs, 
    dtype=np.float32, recompute=False):
    """
    Fit GP model using Metropolis MCMC sampling and save to
    data directory

    Parameters
    ----------
    train_config : module
                   Imported experiment configuration
    
    n_sims : list of int
             Number of simultions
    
    n_pcs : list of int
            Number of princpal components
    
    dtype: type, optional (default np.float32)
           Data type to cast simulation to into. Accepts any
           valid numpy dtype=dtype
    
    recompute : bool, optional (default False)
                Force recompute of PC coefficients
                and basis even if found on disk
    
    Returns
    -------
    models : list of Sepia Model
             Models with parameters sampled using MCMC
    """
    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(train_config.X_standard, delimiter=',', max_rows=1,
        dtype=str, comments=None)

    y_sim = np.load(train_config.Y_physical).T.astype(dtype)

    data_dir = os.path.join(train_config.data_dir, 'models')
    print(train_config.data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    models = []
    dts_pca = []
    dts_mcmc = []
    all_nsims = []
    all_npcs = []
    for i in range(len(n_sims)):
        for k in range(len(n_pcs)):
            m = n_sims[i]
            p = n_pcs[k]
            all_nsims.append(m)
            all_npcs.append(p)
            ti_std = t_std[:m, :]
            yi_phys = y_sim[:m, :]
            
            t0 = time.perf_counter()
            model_name = '{}_n{:03d}_p{:02d}'.format(train_config.exp, m, p)
            m_name = '{}_n{:03d}'.format(train_config.exp, m)
            sepia_data, model = init_model(t_std=ti_std, y_sim=yi_phys, 
                exp=m_name, 
                p=p, data_dir=data_dir, recompute=recompute)
            t1 = time.perf_counter()
            dts_pca.append(t1 - t0)

            sim_data = sepia_data.sim_data
            w = np.dot(np.linalg.pinv(sim_data.K).T, sim_data.y_std.T).T
            y_sim_std_hat = np.dot(w, sim_data.K)
            pc_resid = sim_data.y_std - y_sim_std_hat
            pc_var = np.var(pc_resid)
            pc_prec = 1/pc_var
            print('PC PRECISION:', pc_prec)
            gamma_a = 50
            gamma_b = gamma_a/pc_prec
            model.params.lamWOs = SepiaParam(val=pc_prec, name='lamWOs', 
                val_shape=(1, 1), dist='Gamma', params=[gamma_a, gamma_b], 
                bounds=[1., np.inf], mcmcStepParam=10, mcmcStepType='Uniform')
            model.params.mcmcList = [model.params.betaU, 
                model.params.lamUz, model.params.lamWs, model.params.lamWOs]

            t0 = time.perf_counter()
            model.tune_step_sizes(100, 5)
            model.do_mcmc(512)
            t1 = time.perf_counter()
            dts_mcmc.append(t1 - t0)
            model.save_model_info(os.path.join(data_dir, model_name))
            models.append(model)
        
    dt = np.array([all_nsims, all_npcs, dts_pca, dts_mcmc]).T
    np.savetxt(os.path.join(data_dir, 'timing.csv'),
        dt, delimiter=',', fmt='%.3f', header='sims,PCs,PCA (seconds),MCMC (seconds)')
    
    return models