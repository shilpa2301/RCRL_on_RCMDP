#!/bin/bash -l
#SBATCH --job-name=sg_9
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ag2682 # Replace $PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M


hostname
date
# sleep 10800
# # Activate Conda
source /home/sm3934/miniconda3/etc/profile.d/conda.sh
conda activate rpcrl_env
which python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=/project/ag2682/sm3934/RCRL_on_RMDP:$PYTHONPATH


/home/sm3934/miniconda3/envs/rpcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RCMDP/code_ipm_rcmdp_rcrl_max_safety_gym_rpcrl.py --config envs/env_configs/safety_gym_circle.yaml
