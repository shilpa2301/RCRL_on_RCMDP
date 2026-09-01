#!/bin/bash -l
#SBATCH --job-name=hum_percmdp
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ag2682 # Replace $PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00  # D-HH:MM:SS
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


# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_pd_rcrl.py --run 1 --seed 1 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_pd_2.py --run 1 --seed 1 --max_train_steps 5000
# /home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/code_ipm_rcmdp_rcrl_max.py --run 1 --seed 1 --env CartPolePerturbedEnv --persistent_eps 2.0 --beta 25.0 --max_train_steps 5000
/home/sm3934/miniconda3/envs/rpcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RCMDP/code_ipm_rcmdp_rcrl_max_humanoid_sparse.py --run 1 --persistent_eps 0.5 --beta 300000 --K_epochs 5 --max_train_steps 8000 --warm_start_episode 300 --lr_cost 1e-4 --lr_a 1e-3 --entropy_coef 0.01 --weight_reg 0.001 --seed 6 --env HumanoidForwardObstaclePerturbed --sigma_gravity 0.7 --lr_c 5e-4 


