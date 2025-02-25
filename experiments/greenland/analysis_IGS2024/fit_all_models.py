"""
Fit GP models for different subsets of training data and different
choices for the number of principal components.
"""

import argparse
from src import utils
from src import model

def main():
    """
    Command-line interface to src.model tools for model fitting

    usage: fit_all_models.py [-h] --npc NPC [NPC ...] --nsim NSIM [NSIM ...] [--recompute] config_file
    
    positional arguments:
        config_file

    options:
        -h, --help            show help message and exit
        --npc NPC [NPC ...]
        --nsim NSIM [NSIM ...]
        --recompute, -r
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--npc', nargs='+', type=int, required=True)
    parser.add_argument('--nsim', nargs='+', type=int, required=True)
    parser.add_argument('--recompute', '-r', action='store_true')
    args = parser.parse_args()
    config = utils.import_config(args.config_file)
    model.fit_models(train_config=config, n_sims=args.nsim,
        n_pcs=args.npc, recompute=args.recompute)

if __name__=='__main__':
    main()
