"""
Script in development for doing distributed HPO on Cori.
"""

# System
import os
import logging
import argparse
import yaml
from copy import deepcopy
import subprocess

# Externals
import numpy as np

# Locals
from utils.slurm import SlurmJob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('hpo.py')
    add_arg = parser.add_argument
    add_arg('--n-trials', type=int, default=64)
    add_arg('--n-nodes', type=int, default=4)
    add_arg('--n-epochs', type=int, default=16)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def config_logging(verbose=False):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def build_config(base_config, output_dir,
                 dropout, optimizer, lr,
                 batch_size, n_epochs, lr_warmup_epochs):
    config = deepcopy(base_config)
    config['output_dir'] = output_dir
    config['model']['dropout'] = dropout
    config['optimizer']['name'] = optimizer
    config['optimizer']['lr'] = lr
    config['training']['batch_size'] = batch_size
    config['training']['n_epochs'] = n_epochs
    config['training']['lr_warmup_epochs'] = lr_warmup_epochs
    return config

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
    return config

def write_config(config, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        yaml.dump(config, f)

def submit_task(task_cmd, n_nodes=1):
    """Submit a task to run in the job allocation with srun"""
    cmd = 'srun -l -N %i %s' % (n_nodes, task_cmd)
    return subprocess.Popen(cmd.split(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)

def main():

    # Initialization
    args = parse_args()
    config_logging(args.verbose)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load the base configuration
    base_config = load_config('configs/cifar10_resnet.yaml')

    # Generate hyper-parameters
    logging.info('Preparing %i HPO trials with %i nodes each', args.n_trials, args.n_nodes)
    output_dir_base = os.path.expandvars('$SCRATCH/sc18-dl-tutorial/cifar-resnet-hpo')
    output_dirs = [os.path.join(output_dir_base, 'hp_%i' % i) for i in range(args.n_trials)]
    dropout = np.random.rand(args.n_trials)
    optimizer = np.random.choice(['Adam', 'Nadam', 'RMSprop', 'Adadelta'], size=args.n_trials)
    lr = np.random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005], size=args.n_trials)
    batch_size = np.random.choice([32, 64, 128, 256], size=args.n_trials)
    lr_warmup_epochs = np.random.randint(args.n_epochs, size=args.n_trials)

    # Prepare the configs
    configs = [
        build_config(base_config,
                     output_dir=output_dirs[i],
                     optimizer=str(optimizer[i]),
                     lr=float(lr[i]),
                     dropout=float(dropout[i]),
                     batch_size=int(batch_size[i]),
                     n_epochs=args.n_epochs,
                     lr_warmup_epochs=int(lr_warmup_epochs[i]))
        for i in range(args.n_trials)
    ]

    # Submit the tasks
    results = []
    for i, config in enumerate(configs):
        logging.info('Submitting HP %i: %s', i, config)
        output_dir = config['output_dir']
        
        # Write the configuration to file
        os.makedirs(output_dir, exist_ok=True)
        config_file = os.path.join(output_dir, 'config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Submit the task
        results.append(submit_task('python ./train.py -d %s' % config_file,
                                   n_nodes=args.n_nodes))

    # Wait and gather all the results
    logging.info('Waiting for tasks to complete...')
    outputs = [r.communicate() for r in results]

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
