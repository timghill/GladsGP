"""
Compute prediction error for different numbers of simulations and different
choices for the number of principal components given a common test set
"""

import os
import argparse

fs = 8
import matplotlib
matplotlib.rc('font', size=fs)

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import cmocean

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia import SepiaPlot
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction

from src import utils
from src.model import load_model

flierprops = {'marker':'+', 'markersize':2, 'markeredgewidth':0.6}

def plot_marginal_loss(path, n_sims, n_pcs, m_ref, p_ref):
    """
    Plot GP prediction RMSE, std residuals for n_sims and n_pcs
    """
    fig, axs = plt.subplots(figsize=(6, 4), ncols=4, nrows=2)

    # 1 For number of PCs
    ax1,ax2,ax3,ax4 = axs[0]
    RMSE = None
    MAPE = None
    CI = None
    full_cis = np.zeros(len(n_pcs))
    coverage = np.zeros(len(n_pcs))
    labelsize = 6
    for i in range(len(n_pcs)):
        p = n_pcs[i]
        performance = np.loadtxt(path.format(m_ref, p), delimiter=',')
        if RMSE is None:
            n_train = performance.shape[0]
            RMSE = np.zeros((len(n_pcs), n_train))
            MAPE = np.zeros((len(n_pcs), n_train))
            CI = np.zeros((len(n_pcs), n_train))
            cov = np.zeros((len(n_pcs), n_train))
        RMSE[i,:] = performance[:,0]
        MAPE[i,:] = performance[:,1]
        lower = performance[:, 2]
        upper = performance[:, 3]
        CI[i,:] = upper - lower
        full_cis[i] = performance[0, 4]
        cov[i,:] = performance[:,5]
        coverage[i] = np.mean(cov[i])

    metrics = (RMSE.T, 100*MAPE.T, CI.T, 100*cov.T)
    labels = ('RMSE', 'MAPE (%)', '95% prediction interval', 'Coverage (%)')
    alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    dys = (0.05, 2, 0.1, 5)
    labelpads = [2, -2, 0, -4]
    # cbars = [0, 0, 0]
    # colors = cmocean.cm.amp(np.linspace(0.25, 1, len(n_sims)))
    medianprops = {'color':'#000000'}
    boxprops = {'edgecolor':'none'}
    # fc = ['#356575', '#6295A20.024, 0.', '#80B9AD', '#B3E2A7']
    fc = [(0.272, 0.259, 0.539), (0.420, 0.431, 0.812),
        (0.647, 0.318, 0.580), (0.858, 0.478, 0.791)]
    for i in range(len(metrics)):
        ax = axs[0,i]
        ax.grid(linestyle=':', linewidth=0.5)
        boxprops['facecolor'] = fc[i]
        boxes = ax.boxplot(metrics[i], tick_labels=n_pcs, patch_artist=True,
            medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
            flierprops=flierprops, whiskerprops={'linewidth':0.65})
        ax.set_ylabel(labels[i], labelpad=labelpads[i])
        ymax = np.max(metrics[i])
        dy = dys[i]
        upper = dy*np.ceil(ymax/dy)
        ylim = ax.get_ylim()
        ax.set_ylim([0, upper])
        ax.text(0.15, 0.9, alphabet[i], transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        xtlabels = np.array(n_pcs).astype(str)
        xtlabels[1::2] = ''
        ax.set_xticks(n_pcs, xtlabels)
    
    axs[0,2].plot(np.arange(1, len(n_pcs)+1), full_cis, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)

    axs[0,-1].plot(np.arange(1, len(n_pcs)+1), 100*coverage, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)
    axs[0,-1].set_ylim([0, 100])
    
    fig.text(0.5, 0.52, 'Number of PCs', ha='center')

    # 2 Number of simulations
    ax1,ax2,ax3,ax4 = axs[1]
    RMSE = None
    MAPE = None
    CI = None
    full_cis = np.zeros(len(n_sims))
    coverage = np.zeros(len(n_sims))
    for i in range(len(n_sims)):
        m = n_sims[i]
        performance = np.loadtxt(path.format(m,p_ref), delimiter=',')
        if RMSE is None:
            n_train = performance.shape[0]
            RMSE = np.zeros((len(n_sims), n_train))
            MAPE = np.zeros((len(n_sims), n_train))
            CI = np.zeros((len(n_sims), n_train))
            cov = np.zeros((len(n_sims), n_train))
        RMSE[i,:] = performance[:,0]
        MAPE[i,:] = performance[:,1]
        lower = performance[:, 2]
        upper = performance[:, 3]
        CI[i,:] = upper - lower
        full_cis[i] = performance[0, 4]
        cov[i,:] = performance[:,5]
        coverage[i] = np.mean(cov[i])

    metrics = (RMSE.T, 100*MAPE.T, CI.T, 100*cov.T)
    # labels = ('RMSE', 'MAPE (%)', '95% prediction interval width', 'Coverage (%)')
    # medianprops = {'color':'#000000'}
    # boxprops = {'edgecolor':'none'}
    # # fc = ['#8a0A0A', '#BF3131', '#DA6262']
    # # fc = cmocean.cm.balance([0.85, 0.75, 0.15, 0.25])
    # fc = ['#356575', '#6295A2', '#80B9AD', '#B3E2A7']
    for i in range(len(metrics)):
        ax = axs[1,i]
        ax.grid(linestyle=':', linewidth=0.5)
        boxprops['facecolor'] = fc[i]
        boxes = ax.boxplot(metrics[i], tick_labels=n_sims, patch_artist=True,
            medianprops=medianprops, boxprops=boxprops, showcaps=False, showfliers=True,
            flierprops=flierprops, whiskerprops={'linewidth':0.65})
        ax.set_ylabel(labels[i], labelpad=labelpads[i])
        ymax = np.max(metrics[i])
        dy = dys[i]
        upper = dy*np.ceil(ymax/dy)
        ylim = ax.get_ylim()
        ax.set_ylim([0, upper])
        ax.text(0.15, 0.9, alphabet[i+4], transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold')
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fs)
        xtlabels = np.array(n_sims).astype(str)
        xtlabels[0::2] = ''
        ax.set_xticks(np.arange(1,len(n_sims)+1), xtlabels)
        
    axs[1,2].plot(np.arange(1, len(n_sims)+1), full_cis, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)
    axs[1,-1].plot(np.arange(1, len(n_sims)+1), 100*coverage, 
        linestyle='', marker='.', color='#000000', markersize=4, zorder=10)

    ax2 = axs[1,-1].twinx()
    ax2.spines['top'].set_visible(False)
    cputime = 0.5*n_sims
    ax2.plot(np.arange(1, len(n_sims)+1), cputime,
        color='#000000', marker='.', markersize=3, linewidth=0.65)
    ax2.set_ylabel('CPU-hours', rotation=-90, labelpad=8)
    ax2.tick_params(axis='both', labelsize=fs)

    axs[0,-1].set_ylim([0, 110])
    axs[1,-1].set_ylim([0, 110])
    
    fig.text(0.5, 0.025, 'Number of Simulations', ha='center')
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.92, top=0.975, wspace=0.5, hspace=0.3)
    return fig

def plot_joint_loss(path, n_sims, n_pcs, linestyle='solid'):
    """
    Plot GP prediction RMSE, std residuals for n_sims and n_pcs
    """
    fig, axs = plt.subplots(figsize=(6, 3), ncols=2)
    ax1,ax2 = axs
    RMSE = np.zeros((len(n_sims), len(n_pcs)))
    MAPE = np.zeros((len(n_sims), len(n_pcs)))
    RMSE_lower = np.zeros((len(n_sims), len(n_pcs)))
    RMSE_upper = np.zeros((len(n_sims), len(n_pcs)))
    MAPE_lower = np.zeros((len(n_sims), len(n_pcs)))
    MAPE_upper = np.zeros((len(n_sims), len(n_pcs)))
    AIC = np.zeros((len(n_sims), len(n_pcs)))
    BIC = np.zeros((len(n_sims), len(n_pcs)))
    for i in range(len(n_sims)):
        for j in range(len(n_pcs)):
            m = n_sims[i]
            p = n_pcs[j]
            performance = np.loadtxt(path.format(m,p), delimiter=',')

            rmse = performance[:,0]
            mape = performance[:,1]

            RMSE[i,j] = np.sqrt(np.nanmean(rmse**2))
            MAPE[i,j] = np.nanmean(mape)

            qq = 0.25
            RMSE_lower[i,j] = np.nanquantile(rmse, qq)
            RMSE_upper[i,j] = np.nanquantile(rmse, 1-qq)

            MAPE_lower[i,j] = np.nanquantile(mape, qq)
            MAPE_upper[i,j] = np.nanquantile(mape, 1-qq)

            mean_rss = RMSE[i,j]**2
            k = p*(8+1)
            AIC[i,j] = 2*k + m*np.log(mean_rss)
            BIC[i,j] = k*np.log(m) + m*np.log(mean_rss)


    metrics = (RMSE, 100*MAPE)
    lower = (RMSE_lower, 100*MAPE_lower)
    upper = (RMSE_upper, 100*MAPE_upper)
    labels = ('RMSE', 'MAPE (%)', 'Max error')
    dys = (0.05, 2.5)
    alphabet = ('a', 'b')
    cbars = [0, 0, 0]
    colors = cmocean.cm.deep(np.linspace(0.15, 0.9, len(n_sims)))
    for i in range(len(metrics)):
        ax = axs[i]
        ax.grid(linestyle=':', linewidth=0.5)
        for j in range (len(n_sims)):
            if linestyle=='solid':
                ax.plot(n_pcs, metrics[i][j,:], color=colors[j], label=n_sims[j])

            ymax = np.max(upper[i])
            dy = dys[i]
            ub = dy*np.ceil(ymax/dy)
            ax.set_xlim([n_pcs[0]-0.5, n_pcs[-1]+0.5])
            ylim = ax.get_ylim()
            ax.set_ylim([0, ub])

            span_rects = []
            mean_rects = []
            dx = 0.1
            dsim = 0.05
            sim_offset = (2-j)*dsim
            for l in range(len(n_pcs)):
                R = Rectangle((n_pcs[l] - dx/2 + sim_offset, lower[i][j,l]), 
                    width=dx, height=(upper[i][j,l] - lower[i][j,l]))
                span_rects.append(R)

                R2 = Rectangle((n_pcs[l] - 0.1, metrics[i][j,l]), 
                    width=0.2, height=ymax/100)
                mean_rects.append(R2)
                
            span_pcol = PatchCollection(span_rects, color=colors[j], edgecolor='none',
                alpha=0.5)
            ax.add_collection(span_pcol)

            if linestyle=='bar':
                mean_pcol = PatchCollection(mean_rects, color=colors[j], edgecolor='none',
                        alpha=1.)
                ax.add_collection(mean_pcol)

        ax.set_ylabel(labels[i])
        ax.text(0.05, 0.95, alphabet[i], transform=ax.transAxes,
            ha='left', va='top', fontweight='bold')
    
    fig.text(0.5, 0.025, 'Number of PCs', ha='center')
    ax1.legend(loc='upper right', frameon=False, ncols=2)
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.975, top=0.95, wspace=0.2)

    # AIC = AIC - np.min(AIC, axis=0)
    # BIC = BIC - np.min(BIC, axis=0)
    fig2, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4.5))
    for i in range(len(n_sims)):    
        AIC[i] = AIC[i] - np.min(AIC[i])
        BIC[i] = BIC[i] - np.min(BIC[i])
        ax1.plot(n_pcs, AIC[i], label=n_sims[i], color=colors[i], linewidth=2)
        ax2.plot(n_pcs, BIC[i], label=n_sims[i], color=colors[i], linewidth=2)
        aic_minpc = np.argmin(AIC[i])
        bic_minpc = np.argmin(BIC[i])
        ax1.plot(n_pcs[aic_minpc], AIC[i,aic_minpc], marker='x', color=colors[i])
        ax2.plot(n_pcs[bic_minpc], BIC[i,bic_minpc], marker='x', color=colors[i])
    ax1.grid()
    ax2.grid()
    ax1.set_xticks(n_pcs)
    ax2.set_xticks(n_pcs)
    ax1.set_ylabel(r'$\Delta$AIC')
    ax2.set_ylabel(r'$\Delta$BIC')
    
    fig2.text(0.5, 0.15, 'Number of PCs', ha='center')
    ax1.legend(loc='upper left', frameon=False, ncols=2,
        bbox_to_anchor=(0, -0.5, 1, 0.425))
    fig2.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.95, wspace=0.3)
    return fig, fig2


def compute_test_error(train_config, test_config, n_sims, n_pcs, 
    quantile=0.025, dtype=np.float32, test=False):

    t_std = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1,
        comments=None).astype(dtype)
    t_names = np.loadtxt(train_config.X_standard, delimiter=',', max_rows=1,
        dtype=str, comments=None)

    y_sim = np.load(train_config.Y_physical).T.astype(dtype)

    x_pred = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1,
        comments=None)[:test_config.m].astype(dtype)
    y_test = np.load(test_config.Y_physical).T.astype(dtype)
    y_test = y_test[:test_config.m]
    
    data_dir = os.path.join(train_config.data_dir, 'architecture')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_pattern = os.path.join(data_dir, 'performance_n{:03d}_p{:02d}.csv')
    txtargs = dict(delimiter=',')

    scale = 'columnwise'
    n_dim = t_std.shape[1]
    sampler = stats.qmc.LatinHypercube(n_dim, 
        optimization='random-cd', scramble=True, seed=42186)
    if test:
        t_integrate = sampler.random(n=16)
    else:
        t_integrate = sampler.random(n=200)

    for i in range(len(n_sims)):
        for k in range(len(n_pcs)):
            m = n_sims[i]
            p = n_pcs[k]
            print('m={}, p={}'.format(m,p))
            ti_std = t_std[:m, :]
            yi_phys = y_sim[:m, :]
            sepia_data, model = load_model(train_config, m, p)
            print(sepia_data)

            n = model.data.sim_data.y.shape[1]
            mu_y = np.mean(model.data.sim_data.y, axis=0)
            sd_y = np.std(model.data.sim_data.y, ddof=1, axis=0)
            sd_y[sd_y<1e-6] = 1e-6

            if test:
                samples = model.get_samples(6)
            else:
                samples = model.get_samples(64)
            
            for key in samples.keys():
                samples[key] = samples[key].astype(dtype)
            ypred_mean = np.zeros((test_config.m, yi_phys.shape[1]), dtype=dtype)
            ypred_lq = np.zeros((test_config.m, yi_phys.shape[1]), dtype=dtype)
            ypred_uq = np.zeros((test_config.m, yi_phys.shape[1]), dtype=dtype)

            n_per_batch = 4
            n_batches = int(np.ceil(len(x_pred)/n_per_batch))
            batch_indices = np.array_split(np.arange(len(x_pred)), n_batches)
            print(batch_indices)
            print('Using {} batches of ~{}'.format(n_batches, n_per_batch))
            for j in range(n_batches):
                print('Test Batch {}/{}'.format(j+1, n_batches))
                tj_pred = x_pred[batch_indices[j],:]
                preds = SepiaEmulatorPrediction(t_pred=tj_pred, 
                    samples=samples, model=model)
                preds.w = preds.w.astype(np.float32)
                ypreds = preds.get_y()
                error_preds = np.zeros(ypreds.shape, dtype=np.float32)
                for l_pred in range(len(batch_indices[j])):
                    for l_sample in range(error_preds.shape[0]):
                        err_sd = 1/np.sqrt(samples['lamWOs'][l_sample])
                        error_preds[l_sample][l_pred] = sd_y*np.random.normal(scale=err_sd)
                ypred_mean[batch_indices[j]] = np.mean(ypreds, axis=0)
                ypred_lq[batch_indices[j]] = np.quantile(ypreds + error_preds, quantile, axis=0)
                ypred_uq[batch_indices[j]] = np.quantile(ypreds + error_preds, 1-quantile, axis=0)

            test_confint = np.zeros(len(t_integrate), dtype=dtype)
            n_batches = int(np.ceil(len(t_integrate)/n_per_batch))
            batch_indices = np.array_split(np.arange(len(t_integrate)), n_batches)
            for j in range(n_batches):
                print('Integrate Batch {}/{}'.format(j+1, n_batches))
                tj_integrate = t_integrate[batch_indices[j], :]
                preds = SepiaEmulatorPrediction(t_pred=tj_integrate, 
                    samples=samples, model=model)
                preds.w = preds.w.astype(np.float32)
                ypreds = preds.get_y()
                error_preds = np.zeros(ypreds.shape, dtype=np.float32)
                for l_pred in range(len(batch_indices[j])):
                    for l_sample in range(error_preds.shape[0]):
                        err_sd = 1/np.sqrt(samples['lamWOs'][l_sample])
                        error_preds[l_sample][l_pred] = sd_y*np.random.normal(scale=err_sd)
                yint_lq = np.quantile(ypreds + error_preds, quantile, axis=0)
                yint_uq = np.quantile(ypreds + error_preds, 1-quantile, axis=0)
                test_confint[batch_indices[j]] = np.mean(yint_uq - yint_lq)

            pred_resid = ypred_mean - y_test
            pred_rmse = np.sqrt(np.mean(pred_resid**2, axis=1))
            print('RMSE:', np.sqrt(np.mean(pred_rmse**2)))
            lq = np.quantile(y_test, 0.1)
            inner_mape = np.abs(pred_resid/y_test)
            inner_mape[y_test<lq] = np.nan
            pred_mape = np.nanmean(inner_mape, axis=1)

            is_covered = np.logical_and(
                y_test>=ypred_lq, y_test<=ypred_uq)
            frac_covered = is_covered.sum(axis=1)/is_covered.shape[1]

            pred_lq = np.mean(ypred_lq, axis=1)
            pred_uq = np.mean(ypred_uq, axis=1)
            integrated_ci = np.mean(test_confint)
            confint = integrated_ci*np.ones(pred_rmse.shape)

            pred_arr = np.array([
                pred_rmse, 
                pred_mape, 
                pred_lq, 
                pred_uq, 
                confint, 
                frac_covered
            ]).T
            csv_file = csv_pattern.format(m, p)
            pred_header = 'RMSE,MAPE,Lower quantile,Upper quantile,Integrated confidence interval,Fraction covered'
            np.savetxt(csv_file, pred_arr, header=pred_header,
                delimiter=',', fmt='%.6e')
    return

def main(train_config, test_config, n_sims, n_pcs, 
    recompute=False, test=False):

    if recompute:
        compute_test_error(train_config, test_config, n_sims, n_pcs, test=test)

    path = os.path.join(train_config.data_dir, 'architecture/performance_n{:03d}_p{:02d}.csv')
    print('path:', path)
    fig1, fig2 = plot_joint_loss(path, n_sims, n_pcs)
    if not os.path.exists(train_config.figures):
        os.makedirs(train_config.figures)
    fig1.savefig(os.path.join(train_config.figures, 'nsim_npcs_error.png'),
        dpi=400)
    fig1.savefig(os.path.join(train_config.figures, 'nsim_npcs_error.pdf'))
    
    fig3 = plot_marginal_loss(path, np.array(n_sims), np.array(n_pcs), train_config.m, train_config.p)
    fig3.savefig(os.path.join(train_config.figures, 'nsim_boxplot.png'), dpi=400)
    fig3.savefig(os.path.join(train_config.figures, 'nsim_boxplot.pdf'))
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_conf')
    parser.add_argument('test_conf')
    parser.add_argument('--npc', nargs='+', type=int, required=True)
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    parser.add_argument('--recompute', '-r', 
        help='Refit models and recompute prediction error',
        action='store_true')
    parser.add_argument('--test', '-t', action='store_true')
    args = parser.parse_args()
    train_config = utils.import_config(args.train_conf)
    test_config = utils.import_config(args.test_conf)
    main(train_config, test_config, n_sims=args.nsim, n_pcs=args.npc,
        recompute=args.recompute, test=args.test)
    
