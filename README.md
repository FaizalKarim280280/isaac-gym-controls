# isaac-gym-controls

## Installation

### 1. IsaacGym

Refer to <a href="https://medium.com/@piliwilliam0306/install-isaac-gym-on-ubuntu-22-04-8ebf4b86e6f7">Install Isaac Gym on Ubuntu 22.04</a>

```
cd IsaacGym_Preview_4_Package/isaacgym
conda env create -f rlgpu_conda_env.yml
conda activate rlgpu_conda_env

cd ./python
pip install -e .
```

### 2. CGN_pytorch

Refer to <a href="https://github.com/sebbyjp/cgn_pytorch">Contact-GraspNet Pytorch</a>

```
conda create --name cgn python=3.9
conda activate cgn
cd cgn_pytorch/cgn_pytorch
pip install -r requirements_2.txt

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric
```

### 3. IsaacGymEnvs

Refer to <a href="https://github.com/isaac-sim/IsaacGymEnvs/tree/main?tab=readme-ov-file#installation">IsaacGymEnvs Installation</a>

```
conda activate rlgpu_conda_env
cd IsaacGymEnvs
pip install -e .
```

## Instructions to run IsaacGym

```
python3 single_arm.py
```

[1] Capture Images and exit? \
[2] Run Ik \
[3] Grasp Object \
[4] Exit

## Instructions to run IsaacGymEnvs

```
python3 train.py task=Cartpole
```

The task can be changed to othe tasks such as ```Ant, Humanoid, FrankaCubeStack``` etc. Refer to <a href="https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/rl_examples.md">IsaacGymEnvs Tasks</a>. 

## For Practice

1. Try changing the goal poses of IK and note if the arm is able to reach there or not. Note: In some cases the arm will struggle to reach certain goal poses because of joint limits. 

2. Currently, the grasp object only grasps and lifts the object to the lift pose. Try extending the current code to also place the object to a final goal pose after the object has been lifted up. 

## Migration from Isaac Gym to Isaac Lab

Refer to this artcle <a href="https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html">Isaac Lab Documentation</a>

