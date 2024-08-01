"""
Statistics to summarize emulator performance
"""

import argparse
import os

import numpy as np
from src.utils import import_config

def summarize_performance(train_config, test_config, nsim, npc):
    for i,m in enumerate(nsim):
        for j,p in enumerate(npc):
            perf_file = np.loadtxt('data/architecture/performance_n{:03d}_p{:02d}.csv'.format(m,p),
                delimiter=',')
            rmse = perf_file[:,0]
            mape = perf_file[:,1]

            print('m={}, p={}'.format(m,p))

            total_rmse = np.sqrt(np.mean(rmse**2))
            total_mape = np.mean(np.abs(mape))

            print('RMSE:', total_rmse)
            print('(5%, 95%):', np.quantile(rmse, [0.05, 0.95]))

            print('MAPE:', total_mape)
            print('(5%, 95%):', np.quantile(mape, [0.05, 0.95]))

            print('Integrated uncertainty:', perf_file[0,4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--nsim', nargs='+', type=int)
    parser.add_argument('--npc', nargs='+', type=int)
    args = parser.parse_args()
    train_config = import_config(args.train_config)
    test_config = import_config(args.test_config)
    summarize_performance(train_config, test_config, args.nsim, args.npc)

if __name__=='__main__':
    main()