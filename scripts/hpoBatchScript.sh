#!/bin/bash
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/cifar10-hpo-%j.out

. scripts/setup.sh
python hpo.py
