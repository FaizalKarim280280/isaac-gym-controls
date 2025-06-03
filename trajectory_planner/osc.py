import torch
from isaacgym import gymtorch
import math
from isaacgym.torch_utils import quat_conjugate, quat_mul
from icecream import ic


class OSC:
    def __init__(self, gym, sim, device):
        self.gym = gym
        self.sim = sim
        self.kp = 2 * torch.tensor([10.0, 10, 10, 20, 20, 20.0], device=device)
        self.kv = 20
        
    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


    def compute_pos_action(self, current_eef_pose, 
                        target_eef_pose,
                        mm, j_eef, dof_vel):
        
        current_eef_pos = current_eef_pose[:, 0:3]
        current_eef_quat = current_eef_pose[:, 3:7]
        target_eef_pos = target_eef_pose[:, 0:3]
        target_eef_quat = target_eef_pose[:, 3:7]
        
        current_eef_quat /= torch.norm(current_eef_quat, dim=-1)
        
        # Compute errors
        pos_err = (target_eef_pos - current_eef_pos)
        orn_err = self.orientation_error(target_eef_quat, current_eef_quat)


        dpose = torch.cat([pos_err, orn_err], dim=-1).squeeze(0)

        # Operational Space Control law
        m_inv = torch.inverse(mm)
        m_eef = torch.inverse(j_eef @ m_inv @ j_eef.transpose(1, 2))
        
        # print('=' * 50)
        # print(j_eef.transpose(1, 2).shape, m_eef.shape)
        # print(self.kp.shape, dpose.shape, dof_vel.shape, mm.shape)
        u = j_eef.transpose(1, 2) @ m_eef @ (self.kp * dpose) - self.kv * mm @ dof_vel.squeeze(0)
        return u
    