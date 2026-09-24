#!/bin/bash -l
#SBATCH --job-name=hum_b300
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


/home/sm3934/miniconda3/envs/rpcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RCMDP/code_ipm_rcmdp_rcrl_max_humanoid_feasible_policy.py --run 23 --persistent_eps 1.0 --beta 300 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 300 --lr_cost 5e-4 --lr_a 1e-3 --entropy_coef 0.01 --weight_reg 0.001 --seed 2 --env HumanoidWithCostPerturbed --sigma_gravity 0.7 --lr_c 5e-4 
