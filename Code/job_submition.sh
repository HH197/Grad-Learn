#!/bin/bash
#SBATCH --job-name=exp_scale_ZINB
#SBATCH --time=04:00:00				  # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=2             # number of cores
#SBATCH --partition=gpu-v100 --gres=gpu:1  # type and number of GPU(s) per node
#SBATCH --mem=32000				  # max memory (default unit is MB) per node
#SBATCH --output=exp_scale_ZINB%j.out		  # file name for the output
#SBATCH --error=exp_scale_ZINB%j.err		  # file name for errors
					                  # %j gets replaced by the job number
#SBATCH --mail-user=***@ucalgary.ca  # mail job notifications here
#SBATCH --mail-type=ALL				  # what to notify about


##scp experiment.py data_prep.py ZINB_grad.py ***@arc.ucalgary.ca:/work/long_lab/Hamid/Code

SLURM_TMPDIR=/scratch/${SLURM_JOB_ID}

## load modules

module load python/anaconda3-2018.12 cuda/11.3.0

## setup virtual environment
## compute canada only uses virtualenv and pip
## do not use conda as conda will write files to home directory
python3 -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

## install project dependencies
pip install --no-index --upgrade pip
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113imp
pip install numpy
pip install pandas
pip install scipy
pip install -U scikit-learn
pip install h5py
pip install pyro-ppl
pip install scvi-tools

## run test
python /work/long_lab/Hamid/Code/experiment.py


## clean up by stopping virtualenv
deactivate



###for conda 
source ~/software/init_conda
conda activate dlearn



