#!/bin/bash
#SBATCH --job-name=exp_scale_scvi
#SBATCH --time=04:00:00				  # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=2             # number of cores
#SBATCH --partition=gpu-v100 --gres=gpu:1  # type and number of GPU(s) per node
#SBATCH --mem=32000				  # max memory (default unit is MB) per node
#SBATCH --output=exp_scale_scvi%j.out		  # file name for the output
#SBATCH --error=exp_scale_scvi%j.err		  # file name for errors
					                  # %j gets replaced by the job number
#SBATCH --mail-user=hamid.hamidi@ucalgary.ca  # mail job notifications here
#SBATCH --mail-type=ALL				  # what to notify about



###for conda 
source ~/software/init_conda
conda activate dlearn


## run test
python /work/long_lab/Hamid/Code/experiment_scvi.py


