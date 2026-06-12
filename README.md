#Robust Peak-Constrained Reinforcementy Learning (RP-CRL)

# Dependencies

## Installation:
Install the environment using requirements.txt or ipm_rcrl_env_latest.yaml

## Mujoco
- Ensure that [Mujoco](https://mujoco.org/) is installed and properly configured on your system. Mujoco is required for running the environments.

## Safe-Control-Gym
- **Safe-Control-Gym** is already integrated and should work fine. If you encounter any issues, you can find the repository and installation instructions here: [Safe-Control-Gym GitHub Repository](https://github.com/ManUtdMoon/safe-control-gym).


# Important Commands

## HALFCHEETAH ENV
```bash
python code_ipm_rcmdp_rcrl_max_hc.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 500 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.0 --seed 5 --env HalfCheetahWithPos --lr_c 5e-3 

python code_ipm_rcmdp_rcrl_max_hc.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 500 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.001 --seed 5 --env HalfCheetahWithPosPerturbed --lr_c 5e-3 --sigma_gravity 2.0
```

## REACHER ENV
```bash
python code_ipm_rcmdp_rcrl_max_reacher.py --run 2 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.0 --sigma_gravity 0.0 --env ReacherWithCost --seed 1

python code_ipm_rcmdp_rcrl_max_reacher.py --run 2 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.001 --sigma_gravity 0.7 --env ReacherWithCostPerturbed --seed 1
```

## SWIMMER ENV
```bash
python code_ipm_rcmdp_rcrl_max_swimmer.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.001 --sigma_viscosity 0.0 --env SwimmerWithPos --seed 1

python code_ipm_rcmdp_rcrl_max_swimmer.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.001 --sigma_viscosity 0.7 --env SwimmerWithPosPerturbed --seed 1
```

## QUADROTOR ENV
```bash
python code_ipm_rcmdp_rcrl_max_quadrotor2.py --run 2 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 5e-4 --lr_c 5e-4 --lr_a 5e-4 --entropy_coef 0.007 --weight_reg 0.0 --seed 2 --batch_size 1024 --mini_batch_size 256
```

## ANT ENV
```bash
python code_ipm_rcmdp_rcrl_max_ant.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 700 --lr_cost 1e-3 --lr_a 1e-3 --entropy_coef 0.001 --weight_reg 0.0 --seed 2 --env AntCost --lr_c 1e-3 

python code_ipm_rcmdp_rcrl_max_ant.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 700 --lr_cost 1e-3 --lr_a 1e-3 --entropy_coef 0.001 --weight_reg 0.001 --seed 2 --env AntCostPerturbed --sigma_gravity 0.7 --lr_c 1e-3 
```

## HUMANOID ENV
```bash
python code_ipm_rcmdp_rcrl_max_humanoid.py --run 1 --persistent_eps 1.0 --beta 300000 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 300 --lr_cost 5e-4 --lr_a 1e-3 --entropy_coef 0.01 --weight_reg 0.0 --seed 6 --env HumanoidWithCost --lr_c 5e-4 

python code_ipm_rcmdp_rcrl_max_humanoid.py --run 1 --persistent_eps 1.0 --beta 300000 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 300 --lr_cost 5e-4 --lr_a 1e-3 --entropy_coef 0.01 --weight_reg 0.001 --seed 6 --env HumanoidWithCostPerturbed --sigma_gravity 0.7 --lr_c 5e-4 
```

## WALKER2D ENV
```bash
python code_ipm_rcmdp_rcrl_max_walker2d.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 500 --lr_cost 5e-4 --lr_a 1e-3 --entropy_coef 0.01 --weight_reg 0.0 --seed 6 --env Walker2dWithCost --lr_c 5e-4 

python code_ipm_rcmdp_rcrl_max_walker2d.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 6000 --warm_start_episode 500 --lr_cost 1e-3 --lr_a 1e-3 --entropy_coef 0.001 --weight_reg 0.001 --seed 2 --env Walker2dWithCostPerturbed --sigma_gravity 0.7 --lr_c 5e-4 
```

## PRIMAL DUAL (WORK IN PROGRESS)
```bash
python code_ipm_rcmdp_primal_dual_swimmer.py --run 1 --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.0 --sigma_viscosity 0.0 --env SwimmerWithPos --seed 1
```

