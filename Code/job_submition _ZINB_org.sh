#!/bin/bash
#SBATCH --job-name=exp_ZINB_org
#SBATCH --time=05:00:00				  # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=4             # number of cores
#SBATCH --partition=cpu2021
#SBATCH --mem=64000				  # max memory (default unit is MB) per node
#SBATCH --output=exp_ZINB_org%j.out		  # file name for the output
#SBATCH --error=exp_ZINB_org%j.err		  # file name for errors
					                  # %j gets replaced by the job number
#SBATCH --mail-user=hamid.hamidi@ucalgary.ca  # mail job notifications here
#SBATCH --mail-type=ALL				  # what to notify about



###for conda 
source ~/software/init_conda
conda activate pytorch
export R_HOME=/home/hamid.hamidi/miniconda3/envs/R/lib/R

## run test
python /work/long_lab/Hamid/Code/experiment_ZINB_org.py


