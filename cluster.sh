#!/bin/bash -l
#SBATCH --job-name=cart_per1
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ag2682 # Replace $PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M

hostname
date
# sleep 10800
# # Activate Conda
source /home/sm3934/miniconda3/etc/profile.d/conda.sh
conda activate ipm_rcrl_env
which python
export PYTHONPATH=/project/ag2682/sm3934/RCRL_on_RMDP:$PYTHONPATH



# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2     /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra/config.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra > run_job_idtra_checkpoints_sub100_req_output.log 2>&1

/home/sm3934/miniconda3/envs/ipm_rcrl_env/bin/python /project/ag2682/sm3934/RCRL_on_RMDP/ipm_rcmdp_rcrl.py