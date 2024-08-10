"""
Explore how to include PC truncation error in emulator predictions
"""
import os

import argparse
import numpy as np

from matplotlib import pyplot as plt

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction
from sepia.SepiaPrior import SepiaPrior

from src.utils import import_config
from src import model as md

def main(train_config, test_config, m, p, dtype=np.float32):

    # What was the MCMC mean value?
    data,model = md.load_model(train_config, m, p)
    smp = model.get_samples(64, nburn=64)
    est_prec = smp['lamWOs'].mean()
    print(smp['lamWOs'].mean())
    print(np.var(smp['lamWOs']))

    print('Nugget mean:')
    print(smp['lamWs'].mean())

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(train_config.X_standard, delimiter=',', max_rows=1,
        dtype=str, comments=None)

    y_sim = np.load(train_config.Y_physical, mmap_mode='r')
    y_sim = y_sim[:, :m].T.astype(np.float32)

    # Get variance of PC residuals
    K = data.sim_data.K
    K = K/np.vstack(np.linalg.norm(K, axis=1))
    y_reconstructed = y_sim @ K.T @ K
    y_resid = y_sim - y_reconstructed
    print(y_resid.shape)
    truncation_variance = np.var(y_resid)
    print('Truncation variance:', truncation_variance)
    print('Simulation variance:', np.var(y_sim))

    t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)[32:33,:]
    y_test = np.load(test_config.Y_physical, mmap_mode='r')[:, 32].astype(np.float32)

    data_dir = os.path.join(train_config.data_dir, 'models')

    ti_std = t_std[:m, :]
    yi_phys = y_sim[:m, :]
    model_name = '{}_n{:03d}_p{:02d}'.format(train_config.exp, m, p)
    m_name = '{}_n{:03d}'.format(train_config.exp, m)

    nt = 365
    nx = int(len(y_test)/nt)
    node = 1845

    pca_var = [1e6, est_prec, 100, 1/truncation_variance]
    labels = [
        r'No PC truncation error ($\lambda = ${:.1e})', r'MCMC estimated ($\lambda = ${:.1e})',
        r'Prior mean ($\lambda = ${:.1e})', r'Truncation residual ($\lambda = ${:.1e})',
    ]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True, sharey=False)
    for i,var in enumerate(pca_var):
        sepia_data, model = md.init_model(t_std=ti_std, y_sim=yi_phys, 
            exp=m_name, p=p, data_dir=data_dir, recompute=False)

        model.params.lamWOs.fixed = np.array([[True]])
        model.params.lamWOs.val = np.array([[var]])
        model.params.lamWOs.prior = SepiaPrior(model.params.lamWOs, dist='Uniform',
            params=[0, 2*var], bounds=[0, 2*var])

        model.tune_step_sizes(50, 10)

        model.do_mcmc(128)
        samples = model.get_samples(64, nburn=64)

        preds = SepiaEmulatorPrediction(t_pred=t_test, samples=samples, model=model)
        preds.w = preds.w.astype(np.float32)
        ypred = preds.get_y()
        ypred_lq = np.quantile(ypred, 0.025, axis=0).reshape((nx, nt))
        ypred_uq = np.quantile(ypred, 0.975, axis=0).reshape((nx, nt))
        ypred_mean = np.mean(ypred)
        ypred = ypred + np.random.normal(loc=0, scale=np.sqrt(1/var), size=ypred.shape)

        yt = y_test.reshape((nx, nt))
        axs.flat[i].fill_between(x=np.arange(nt),
            y1=ypred_lq[node] - 1.96/np.sqrt(var), 
            y2=ypred_uq[node] + 1.96/np.sqrt(var),
            color='tab:blue', alpha=0.3)
        axs.flat[i].plot(yt[node,:], color='k', linewidth=2)

        for j in range(min(8, ypred.shape[0])):
            yp = ypred[j].reshape((nx, nt))
            axs.flat[i].plot(yp[node, :], color='gray', alpha=0.75, linewidth=0.5)
        
        axs.flat[i].grid(linestyle=':')
        axs.flat[i].set_title(labels[i].format(var))
        axs.flat[i].set_xlim([140, 260])
    
    
    fig.savefig('figures/simulator_precision.png', dpi=600)
    # plt.show()



    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('m', type=int)
    parser.add_argument('p', type=int)
    args = parser.parse_args()
    
    train_config = import_config(args.train_config)
    test_config = import_config(args.test_config)
    main(train_config, test_config, args.m, args.p)