#!/bin/bash -l
#SBATCH --job-name=cp1
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ag2682 # Replace $PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=15:00:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M


hostname
date
# sleep 10800
# # Activate Conda
source /home/sm3934/miniconda3/etc/profile.d/conda.sh
conda activate ipm_rcrl_env
which python
export PYTHONPATH=/project/ag2682/sm3934/RCRL_on_RMDP:$PYTHONPATH


# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_pd_rcrl.py --run 1 --seed 1 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_pd_2.py --run 1 --seed 1 --max_train_steps 5000
/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_ipm_rcmdp_rcrl_max.py --run 17 --seed 1 --env CartPolePerturbedEnv --persistent_eps 2.0 --beta 25.0 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_ipm_rcmdp_rcrl_max.py --run 1 --seed 1 --env CartPoleCostEnv --persistent_eps 2.0 --beta 25.0 --max_train_steps 5000

