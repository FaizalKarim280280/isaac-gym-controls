from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from tasks.base_task import BaseTask
import math

class SingleFranka(BaseTask):
    def __init__(self,
                 custom_parameters,
                 device):
        
        super().__init__(custom_parameters, device)
        self.franka_asset_file = 'franka_description/robots/franka_panda_gripper_extended.urdf'
        self.num_envs = self.args.num_envs
        self.num_per_row = int(math.sqrt(self.num_envs))
        self.spacing = 1.0
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        self.envs = []
        self.franka_default_dof_state = [-0.0007, -0.0017, -0.0023, -0.9484, 0.0207, 1.1335, -0.0027, 0.0346, 0.0346]
        self.joints_lower_limit = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.04, -0.04])
        self.joints_upper_limit = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
        self.gripper_idx = []
        self.camera_handles = []
        
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
        cam_pos = gymapi.Vec3(2.0, 0.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def load_assets(self):
        self.table_asset, self.table_pose = self.load_table_asset(
            pose_p=[0, 0, 1.0])
        
        self.stand_asset, self.stand_pose = self.load_franka_stand_asset(
            pose_p=[-0.5, 0, self.table_pose.p.z + 0.1/2 + 0.05/2])
        
        self.franka_asset, self.franka_pose = self.load_franka_asset(
            urdf_path=self.franka_asset_file,
            pose_p=[-0.45, 0.0, self.table_pose.p.z + 0.05/2 + 0.1],
            pose_q=[0, 0, 0, 1.0])
        
        self.cube_asset, self.cube_pose = self.load_object_asset(
            urdf_path='da/cube.urdf',
            pose_p=[0.1, -0.2, 1.075],
            pose_q=[0, 0, 0, 1],
        )
        
    def create_actors(self,):
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, 
                                      self.env_lower,
                                      self.env_upper, 
                                      self.num_per_row)

            # table
            self.table_handle = self.gym.create_actor(env, self.table_asset, self.table_pose,
                                                       "table", i, 0, 0)
            self.gym.set_rigid_body_segmentation_id(env, self.table_handle, 0, 1)
            self.gym.set_rigid_body_color(env, self.table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.686, 0.588, 1.))
            
            # franka stand
            self.stand_handle = self.gym.create_actor(env, self.stand_asset, self.stand_pose,
                                                       "stand", i, 0, 0)
            self.gym.set_rigid_body_segmentation_id(env, self.stand_handle, 0, 2)

            # franka arm
            self.franka_handle = self.gym.create_actor(env, self.franka_asset, self.franka_pose,
                                                        "franka", i, 1, 0)
            num_rigid_bodies = self.gym.get_actor_rigid_body_count(env, self.franka_handle)
            for j in range(num_rigid_bodies):
                self.gym.set_rigid_body_segmentation_id(env, self.franka_handle, j, 3)
            
            # cube
            self.cube_handle = self.gym.create_actor(env, self.cube_asset, self.cube_pose,
                                                         "cube", i, 2, 0)
            
            num_rigid_bodies = self.gym.get_actor_rigid_body_count(env, self.cube_handle)
            for j in range(num_rigid_bodies):
                self.gym.set_rigid_body_segmentation_id(env, self.cube_handle, j, 4)    
            
            
            self.gym.set_actor_dof_properties(env, self.franka_handle, self.franka_dof_props)
            self.gym.set_actor_dof_states(env, self.franka_handle, 
                                          self.franka_default_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env, self.franka_handle, self.franka_default_dof_state)
            self.hand_handle = self.gym.find_actor_rigid_body_handle(env, 
                                                                     self.franka_handle, 
                                                                     "panda_grip_site")
            
            self.gripper_idx.append(
                self.gym.find_actor_rigid_body_index(env, self.franka_handle, "panda_grip_site", gymapi.DOMAIN_SIM)
            )
            self.envs.append(env)
            
            self.gym.set_rigid_body_color(env, self.cube_handle, 1, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
            
            camera_props = gymapi.CameraProperties()
            # print(camera_props)
            # exit()
            camera_props.width = 512
            camera_props.height = 512
            # print(camera_props.near_plane, camera_props.far_plane)
            # exit()
            # camera_props.horizontal_fov = 75.0
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.set_camera_location(camera_handle, env, gymapi.Vec3(0.75, 0, 1.75), gymapi.Vec3(0.25, 0, 1.25))

            self.camera_handles.append(camera_handle)
                    
    def load_state_tensors(self):
        self.jac = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, 'franka'))
        self.eef_jac = self.jac[:, self.hand_handle - 1, :, :7]
        # print(self.eef_jac.shape)
        # exit()
        
        self.mm = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, 'franka'))[..., :7, :7]
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.dof_states = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)).reshape(self.num_envs, -1, 2)
        
    def get_eef_pose(self):
        return self.rb_states[self.gripper_idx, :7]
    
    def get_dof_states(self):
        return self.dof_states[:, :9, 0]
    
    def get_dof_vel(self):
        return self.dof_states[:, :9, 1]