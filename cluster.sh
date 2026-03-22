#!/bin/bash -l
#SBATCH --job-name=car1
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ag2682 # Replace $PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M

hostname
date
# sleep 10800
# # Activate Conda
source /home/sm3934/miniconda3/etc/profile.d/conda.sh
conda activate ipm_rcrl_env
which python
export PYTHONPATH=/project/ag2682/sm3934/RCRL_on_RMDP:$PYTHONPATH



/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py --env CartPoleCostEnv --run 1 --seed 1
#/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py --env CartPoleCostEnv --run 2 --seed 2
#/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py --env CartPoleCostEnv --run 3 --seed 3

#/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py --env CartPolePerturbedEnv --run 1 --seed 1
#/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py --env CartPolePerturbedEnv --run 2 --seed 2
#/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py --env CartPolePerturbedEnv --run 3 --seed 3
