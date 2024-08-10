"""
Compute time to make predictions in a fair way

usage: time_predictions.py [-h] conf_file test_file
"""

import argparse
import time
from src.utils import import_config
from src.model import load_model

import numpy as np

from sepia.SepiaPredict import SepiaEmulatorPrediction

def main(config, test_config, dtype=np.float32):
    """
    Mock GP fitting and predicting to fairly time the model without some
    of the overhead etc. from the main fitting function

    Parameters
    ----------
    config : module
             Configuration file loaded as a module
    
    recompute : bool, optional
                Force recompute fields even if file already exists
    
    dtype : optional (np.float32)
            Data type for GP predictions and CV error calculations            
    """
    # Load data and initialize model
    t_std = np.loadtxt(config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(config.X_physical, delimiter=',', max_rows=1,
        dtype=str, comments=None)
    t_names= [tn.strip('#') for tn in t_names]
    t_std = t_std[:config.m, :]
    y_train_sim = np.load(config.Y_physical, mmap_mode='r').T[:config.m, :].astype(dtype)

    t_test_std = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)[:test_config.m :]
    y_test_sim = np.load(test_config.Y_physical, mmap_mode='r').T[:test_config.m, :].astype(dtype)

    dt_emulator = []
    dt_error = []
    dt_y = []
    data,model = load_model(config, config.m, config.p)

    # Using 32 posterior samples. Appropriate for a lot of uses,
    # especially comparing multiple GPs. May want to use more samples
    # (64, 128) for "best" predictions
    samples = model.get_samples(numsamples=32, nburn=256)
    for key in samples.keys():
        samples[key] = samples[key].astype(dtype)
    
    print('CONFIGURATION:')
    print('m =', config.m)
    print('p = ', config.p)
    print('samples = ', len(samples['lamWOs']))
    m = config.m
    m_pred = test_config.m
    n = model.data.sim_data.y.shape[1]
    mu_y = np.mean(model.data.sim_data.y, axis=0)
    sd_y = np.std(model.data.sim_data.y, ddof=1, axis=0)
    sd_y[sd_y<1e-6] = 1e-6

    for i in range(m_pred):
        xi = t_test_std[i:i+1]
        print('Sample {}/{}:'.format(i+1, m_pred))
        
        # Start timing
        t0 = time.perf_counter()
        # Timing includes making predictions of PCs and transforming
        # back into physical space
        preds = SepiaEmulatorPrediction(samples=samples,
            model=model, t_pred=xi)
        preds.w = preds.w.astype(dtype)
        emulator_preds = preds.get_y()

        # Intermediate reference point to partition timing
        t1 = time.perf_counter()

        # Time to sample PC error term
        error_preds = np.zeros(emulator_preds.shape, dtype=np.float32)
        for j in range(error_preds.shape[0]):
            error_preds[j] = sd_y*np.random.normal(scale=1/np.sqrt(samples['lamWOs'][j])).astype(np.float32)
        
        # Construct the full predictions
        y_preds = emulator_preds + error_preds

        # End timing
        t2 = time.perf_counter()
        dt_emulator.append(t1-t0)
        dt_error.append(t2-t1)
        dt_y.append(t2-t0)
    
    print('Mean timing:')
    print('Emulator-only:', np.mean(dt_emulator))
    print('Error-only:', np.mean(dt_error))
    print('Full prediction distribution:', np.mean(dt_y))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file')
    parser.add_argument('test_file')
    args = parser.parse_args()
    config = import_config(args.conf_file)
    test_config = import_config(args.test_file)
    main(config, test_config)
