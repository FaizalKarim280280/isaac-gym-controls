import torch
from isaacgym import gymtorch
import math
from isaacgym.torch_utils import quat_conjugate, quat_mul
from icecream import ic


class IKController:
    def __init__(self, gym, sim, device):
        self.gym = gym
        self.sim = sim
        self.device = device
        self.kp = torch.tensor([10.0, 10, 10, 20, 20, 20.0], device=device).unsqueeze(0)
        self.damping = 0.2
        
    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
        
    def compute_error(self, current_eef_pose, target_eef_pose):
        pos_error = target_eef_pose[:, :3] - current_eef_pose[:, :3]
        orn_error = self.orientation_error(target_eef_pose[:, 3:], current_eef_pose[:, 3:])
        
        dpose = torch.cat([pos_error, orn_error], dim=-1)
        return dpose
        
    def compute_pos_action(self, current_eef_pose, target_eef_pose, j_eef):
        # ic(j_eef.shape)
        self.dpose = self.compute_error(current_eef_pose, target_eef_pose).squeeze(0)
        
        
        j_eef_T = torch.transpose(j_eef, 1, 2).squeeze(0)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        
        # ic(self.dpose.shape, j_eef.shape, j_eef_T.shape, lmbda.shape)
        
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ self.dpose).view(1, 7)
        return u
