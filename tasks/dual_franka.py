from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from tasks.base_task import BaseTask
import math

ROOT_HALF = math.sqrt(1/2)


class DualFranka(BaseTask):
    def __init__(self,
                 args,
                 device):
        
        super().__init__(args, device)
        self.franka_asset_file = 'franka_description/robots/franka_panda_gripper_extended.urdf'
        self.num_envs = self.args.num_envs
        self.num_per_row = int(math.sqrt(self.num_envs))
        self.spacing = 1.0
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        self.envs = []
        # self.franka_default_dof_state = [0, -1.1752, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]
        self.franka_default_dof_state = np.zeros(9, gymapi.DofState.dtype)
        self.franka_default_dof_state["pos"][:9] = [0, -1.1752, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]

        self.joints_lower_limit = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.04, -0.04])
        self.joints_upper_limit = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
        self.left_gripper_idxs = []
        self.right_gripper_idxs = []
        
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def load_everything(self, ):
        self.load_assets()
        self.create_actors()
        self.load_camera()
        
        self.gym.prepare_sim(self.sim)
        
        self.load_state_tensors()
        
                
    def load_camera(self):
        cam_pos = gymapi.Vec3(0.0, -2.0, 1)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def load_assets(self):
        
        self.franka_left_asset, self.franka_left_pose = self.load_franka_asset(
            urdf_path=self.franka_asset_file,
            pose_p=[-0.45, 0.35, 0],
            pose_q=[0, 0, -ROOT_HALF, ROOT_HALF])
        
        self.franka_right_asset, self.franka_right_pose = self.load_franka_asset(
            urdf_path=self.franka_asset_file,
            pose_p=[0.45, 0.35, 0],
            pose_q=[0, 0, -ROOT_HALF, ROOT_HALF])
        
        # load the object also
        
        self.object_asset, self.object_pose = self.load_object_asset(
            urdf_path='da/chair/chair.urdf',
            pose_p=[0, 0, 0],
            pose_q=[0, 0, 0, 1.0])
        
    def create_actors(self,):
        
        # self.franka_default_dof_state["pos"][:7] = self.franka_mids[:7]
        
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, 
                                      self.env_lower,
                                      self.env_upper, 
                                      self.num_per_row)
            
            self.franka_left_handle = self.gym.create_actor(env, self.franka_left_asset, self.franka_left_pose,
                                                        "franka_left", i, 1, 0)
            
            self.franka_right_handle = self.gym.create_actor(env, self.franka_right_asset,
                                                             self.franka_right_pose,
                                                             'franka_right', i, 2, 0)
            
            self.object_handle = self.gym.create_actor(env, self.object_asset, self.object_pose,
                                                       'object', i, 0, 0)
            
            
            
            self.gym.set_actor_dof_states(env, self.franka_left_handle, 
                                          self.franka_default_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_states(env, self.franka_right_handle,
                                            self.franka_default_dof_state, gymapi.STATE_ALL)
            
            
            self.gym.set_actor_dof_properties(env, self.franka_left_handle, self.franka_dof_props)
            self.gym.set_actor_dof_properties(env, self.franka_right_handle, self.franka_dof_props)
            
            self.left_gripper_idxs.append(
                self.gym.find_actor_rigid_body_index(env, self.franka_left_handle, "panda_grip_site", gymapi.DOMAIN_SIM)
            )
            
            self.right_gripper_idxs.append(
                self.gym.find_actor_rigid_body_index(env, self.franka_right_handle, "panda_grip_site", gymapi.DOMAIN_SIM)
            )
            
            
            self.envs.append(env)
            
        self.hand_left_handle = self.gym.find_actor_rigid_body_handle(env, 
                                                                    self.franka_left_handle, 
                                                                    "panda_grip_site")
        self.hand_right_handle = self.gym.find_actor_rigid_body_handle(env,
                                                                    self.franka_right_handle,
                                                                    "panda_grip_site")
        
        self.left_gripper_handle = self.gym.get_asset_rigid_body_dict(self.franka_left_asset)['panda_grip_site']
        self.right_gripper_handle = self.gym.get_asset_rigid_body_dict(self.franka_right_asset)['panda_grip_site']
            
    def load_state_tensors(self):
        self.left_jac = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, 'franka_left'))
        self.left_eef_jac = self.left_jac[:, self.left_gripper_handle - 1, :]
        
        self.right_jac = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, 'franka_right'))
        self.right_eef_jac = self.right_jac[:, self.right_gripper_handle - 1, :]
        
        self.left_mm = self.gym.acquire_mass_matrix_tensor(self.sim, 'franka_left')
        self.left_mm = gymtorch.wrap_tensor(self.left_mm)
        
        self.right_mm = self.gym.acquire_mass_matrix_tensor(self.sim, 'franka_right')
        self.right_mm = gymtorch.wrap_tensor(self.right_mm)
        
        self.rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self.rb_states)
        
        self.dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self.dof_states).reshape(self.num_envs, -1, 2)
        
        self.left_eef_pose = self.rb_states[self.left_gripper_idxs, :7]
        self.right_eef_pose = self.rb_states[self.right_gripper_idxs, :7]
        
    def get_eef_pose(self):
        return self.rb_states[self.left_gripper_idxs, :7], self.rb_states[self.right_gripper_idxs, :7]
        
    def get_eef_vel(self):
        return self.rb_states[self.left_gripper_idxs, 7:], self.rb_states[self.right_gripper_idxs, 7:]
    
    def get_dof_states_left(self):
        return self.dof_states[:, :9, 1].unsqueeze(-1)
    def get_dof_states_right(self):
        return self.dof_states[:, 9:, 1].unsqueeze(-1)