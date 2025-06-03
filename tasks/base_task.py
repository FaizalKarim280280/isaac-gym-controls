from isaacgym import gymapi
from isaacgym.torch_utils import *

class BaseTask:
    def __init__(
        self,
        args,
        device='cpu'):
        
        self.args = args
        self.device = device
        # self.custom_args = []
        # self.custom_args.extend(custom_parameters)
        # self.args = gymutil.parse_arguments(custom_parameters=custom_parameters)
        self.args.use_gpu_pipeline = False
        
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        
        self.sim = self.gym.create_sim(self.args.compute_device_id, 
                                       self.args.graphics_device_id, 
                                       self.args.physics_engine, 
                                       self.sim_params)
        
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        if not self.args.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        else:
            self.viewer = None
        
    def create_pose(self, p, q):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*p)
        pose.r = gymapi.Quat(*q)
        return pose
    
    def load_franka_asset(self, 
                          urdf_path, 
                          pose_p=[0, 0, 0],
                          pose_q=[0, 0, 0, 1.0],
                          asset_options=None):
        
        if asset_options is None:
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
        
        pose = self.create_pose(pose_p, pose_q)
        
        franka_asset = self.gym.load_asset(self.sim, './assets/urdf', urdf_path, asset_options)

        # get joint limits and ranges for Franka
        self.franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = self.franka_dof_props['lower']
        franka_upper_limits = self.franka_dof_props['upper']
        self.franka_ranges = franka_upper_limits - franka_lower_limits
        self.franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
        self.franka_num_dofs = len(self.franka_dof_props)

        # # set default DOF states
        # default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        # default_dof_state["pos"][:7] = franka_mids[:7]

        # set DOF control properties (except grippers)
        # self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        # self.franka_dof_props["stiffness"][:7].fill(0.0)
        # self.franka_dof_props["damping"][:7].fill(0.0)

        # set DOF control properties for grippers
        if self.args.do_ik:
            self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            self.franka_dof_props["stiffness"].fill(400.0)
            self.franka_dof_props["damping"].fill(40.0)
            
        else:
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.franka_dof_props["stiffness"][:7].fill(10.0)
            self.franka_dof_props["damping"][:7].fill(2.0)
            
            self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            self.franka_dof_props["stiffness"][7:].fill(400.0)
            self.franka_dof_props["damping"][7:].fill(40.0)
        
        return franka_asset, pose
    
    def load_franka_stand_asset(self, 
                                pose_p=[0, 0, 0],
                                pose_q=[0, 0, 0, 1.0],
                                asset_options=None):
        
        if asset_options is None:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            
        pose = self.create_pose(pose_p, pose_q)
        franka_stand_asset = self.gym.create_box(self.sim, 0.2, 0.2, 0.1, asset_options)
        return franka_stand_asset, pose
            
    def load_table_asset(self,
                         table_dims=[1.2, 1.2, 0.05], 
                         asset_options=None,
                         pose_p=[0, 0, 0],
                         pose_q=[0, 0, 0, 1.0]):
        
        if asset_options is None:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
        
        pose = self.create_pose(pose_p, pose_q)
        table_asset = self.gym.create_box(self.sim, table_dims[0], table_dims[1], table_dims[2], asset_options)
        return table_asset, pose
    
    def load_object_asset(self, 
                          urdf_path=None,
                          pose_p=[0, 0, 0],
                          pose_q=[0, 0, 0, 1.0],
                          asset_options=None):
        
        if asset_options is None:
            asset_options = gymapi.AssetOptions()
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 500_000

        pose = self.create_pose(pose_p, pose_q)
        object_asset = self.gym.load_asset(self.sim, './assets/urdf', urdf_path, asset_options)
        return object_asset, pose
    
    def draw_axes(self, env, pos, rot):
        if self.viewer is None:
            return
        
        posx = (pos + quat_apply(rot, torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
        pos = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [pos[0], pos[1], pos[2], posx[0], posx[1], posx[2]], [1, 0.0, 0.0])
        self.gym.add_lines(self.viewer, env, 1, [pos[0], pos[1], pos[2], posy[0], posy[1], posy[2]], [0.0, 1, 0.0])
        self.gym.add_lines(self.viewer, env, 1, [pos[0], pos[1], pos[2], posz[0], posz[1], posz[2]], [0.0, 0.0, 1.0])
        
        
        
        


        
