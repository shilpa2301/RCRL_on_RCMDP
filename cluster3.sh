#!/bin/bash -l
#SBATCH --job-name=per_hcsp3
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ag2682 # Replace $PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M


hostname
date
# sleep 10800
# # Activate Conda
source /home/sm3934/miniconda3/etc/profile.d/conda.sh
conda activate rpcrl_env
which python
export PYTHONPATH=/project/ag2682/sm3934/RCRL_on_RMDP:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_pd_rcrl.py --run 1 --seed 1 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_pd_2.py --run 1 --seed 1 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_ipm_rcmdp_rcrl_max.py --run 1 --seed 1 --env CartPolePerturbedEnv --persistent_eps 2.0 --beta 25.0 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_ipm_rcmdp_rcrl_max.py --run 1 --seed 1 --env CartPoleCostEnv --persistent_eps 2.0 --beta 25.0 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/rpcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RCMDP/code_ipm_rcmdp_rcrl_max_safety_gym_sparse.py --config /project/ag2682/sm3934/RCRL_on_RMDP/envs/env_configs/safety_gym_circle.yaml

/home/sm3934/miniconda3/envs/rpcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RCMDP/code_ipm_rcmdp_rcrl_max_hc_sparse.py --run 3 --persistent_eps 0.2 --K_epochs 5 --max_train_steps 10000 --warm_start_episode 500 --lr_cost 5e-4 --entropy_coef 0.001 --weight_reg 0.001 --seed 3 --env HalfCheetahForwardObstaclePerturbed --lr_c 5e-4 --sigma_gravity 0.7
