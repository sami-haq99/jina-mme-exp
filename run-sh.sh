#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:2
#SBATCH -p compute
#SBATCH -J mm_ss
#SBATCH -t 23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/mtqe/env-mtqe/bin/activate


python ic-gemma3b.py 
