from tasks.single_franka import SingleFranka
import torch
import os
from scipy.spatial.transform import Rotation as R
from utils.viz import draw_axes, clear_axes
from utils.utils import torch_gen_pre_grasp
from isaacgym import gymutil, gymtorch
from trajectory_planner.ik import IKController
from trajectory_planner.osc import OSC
from isaacgym import gymapi
import numpy as np

def main():
    custom_parameters = [
        {'name': '--num_envs', 'type': int, 'default': 1},
        {"name": "--do_ik", "type": bool, "default": False, "help": "Use inverse kinematics to control Franka."},
        {"name": "--add_object", "type": bool, "default": False, "help": "Add monitor object which the arm will hold."}
    ]
    
    #! Initialize the counter and flags
    counter = 0
    capture_image = False
    run_ik = False
    grasp_object = False
    pre_grasp_pose_reached = False
    grasp_pose_reached = False
    lift_object = False
    gripper_amt = 0.04
    #! ======================================================================================================================
    
    print('=' * 100)
    print('=' * 100)
    
    #! Input loop for user interaction
    while True:
        input_str = """[1] Capture Images and exit?\n[2] Run Ik\n[3] Grasp Object \n[4] Exit"""
        
        print(input_str)
        c = input()
        
        if c.lower().strip() == '1':
            print('Capturing images...')
            CAPTURED_IMAGES_PATH = './captured_images'
            os.makedirs(CAPTURED_IMAGES_PATH, exist_ok=True)
            capture_image = True
            break
        elif c.lower().strip() == '2':
            print('Running IK...')
            run_ik = True
            break
        elif c.lower().strip() == '3':
            print('Grasping Object...')
            grasp_object = True
            break
        elif c.lower().strip() == '4':
            print('Exiting...')
            exit()
    #! ======================================================================================================================
    
    args = gymutil.parse_arguments(
        description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
        custom_parameters=custom_parameters,
    )

    args.device = 'cpu'
    args.headless = False
    args.sim_device = 'cpu'
    args.pipeline = 'cpu'
    args.graphics_device_id = 0
    
    #! Create the task instance and load everything
    task = SingleFranka(args, device=args.device)
    
    args.do_ik = True
    controller = IKController(
        gym=task.gym,
        sim=task.sim,
        device=args.device)
    
    task.load_everything()
    #! =======================================================================================================================
    
    goal_pos = [0.1, -0.1, 0.5 + 1]
    goal_pos = torch.tensor([goal_pos], device=args.device).repeat(args.num_envs, 1)
    goal_rot = R.from_quat([0.707, 0, 0.707, 0.0]).as_quat()
    goal_rot = torch.from_numpy(goal_rot).to(args.device).repeat(args.num_envs, 1).float()
    
    goal_pose = torch.cat([goal_pos, goal_rot], dim=-1).to(args.device)
    pos_action = torch.zeros(task.num_envs, 9, device=args.device)
    
    
    #! Main Simluation loop
    while not task.gym.query_viewer_has_closed(task.viewer):

        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        
        task.gym.refresh_rigid_body_state_tensor(task.sim)
        task.gym.refresh_dof_state_tensor(task.sim)
        task.gym.refresh_jacobian_tensors(task.sim)
        task.gym.refresh_mass_matrix_tensors(task.sim)
        task.gym.render_all_camera_sensors(task.sim)
        
        #! [2] Run IK
        if run_ik and counter >= 100:
            draw_axes(task.gym, task.viewer, task.envs[0], goal_pose[0, :3], goal_pose[0, 3:7])
            current_eef_pose = task.get_eef_pose()
            
            pos_action = torch.zeros(task.num_envs, 9, device=args.device)
            
            dof_pos = task.get_dof_states()[..., :7]
            
            pos_action[:, :7] = dof_pos + 0.1 * (controller.compute_pos_action(current_eef_pose, 
                                                                                goal_pose,
                                                                                task.eef_jac))
            pos_action[:, 7:] = torch.Tensor([[0.04, 0.04]] * task.num_envs).to(args.device)
            
            draw_axes(task.gym, task.viewer, task.envs[0],
                    current_eef_pose[0, :3], 
                    current_eef_pose[0, 3:7])
            
            task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(pos_action))
            
        #! [3] Grasp Object 
        if grasp_object and counter >= 100:
            
            # grasp_pos = [0.1, -0.1, 0.2 + 1]
            grasp_pos = [0.12155217 - 0.03, -0.22480735,  1.4207604 - 0.05]
            grasp_pos = torch.tensor([grasp_pos], device=args.device).repeat(args.num_envs, 1)
            # grasp_rot = R.from_quat([1, 0, 0, 0.0]).as_quat()
            grasp_rot = R.from_quat([-0.61183335, -0.053706,  0.05885671,  0.78696347]).as_quat()
            grasp_rot = torch.from_numpy(grasp_rot).to(args.device).repeat(args.num_envs, 1).float()
            
            grasp_pose = torch.cat([grasp_pos, grasp_rot], dim=-1).to(args.device) # user defined grasp pose 
            pre_grasp_pose = torch_gen_pre_grasp(grasp_pose, offset=0.2) # generate pre-grasp pose
                        
            current_eef_pose = task.get_eef_pose() # get current end-effector pose
            dof_pos = task.get_dof_states()[..., :7] # get current joint positions
            
            if not pre_grasp_pose_reached: # first go till pre-grasp
                draw_axes(task.gym, task.viewer, task.envs[0], pre_grasp_pose[0, :3], pre_grasp_pose[0, 3:7])
                pos_action[:, :7] = dof_pos + 0.1 * (controller.compute_pos_action(current_eef_pose,
                                                                                    pre_grasp_pose,
                                                                                    task.eef_jac))
                pos_action[:, 7:] = torch.Tensor([[0.04, 0.04]] * task.num_envs).to(args.device)
                
                if torch.norm(current_eef_pose[:, :3] - pre_grasp_pose[:, :3]) < 0.01:
                    pre_grasp_pose_reached = True
                    print("Pre-grasp pose reached!")
            
            elif pre_grasp_pose_reached and not grasp_pose_reached: # next, go till grasp pose
                draw_axes(task.gym, task.viewer, task.envs[0], grasp_pose[0, :3], grasp_pose[0, 3:7])
                pos_action[:, :7] = dof_pos + 0.1 * (controller.compute_pos_action(current_eef_pose,
                                                                                    grasp_pose,
                                                                                    task.eef_jac))
                pos_action[:, 7:] = torch.Tensor([[0.04, 0.04]] * task.num_envs).to(args.device)
                
                if torch.norm(current_eef_pose[:, :3] - grasp_pose[:, :3]) < 0.01:
                    grasp_pose_reached = True
                    print("Grasp pose reached!")
                    
            elif grasp_pose_reached and not lift_object: # then, close the grippers
                print(f"Grasping object..., gripper_amt: {gripper_amt}")

                gripper_amt = max(0.0001, gripper_amt - 0.0003)
                pos_action[:, :7] = dof_pos + 0.1 * (controller.compute_pos_action(current_eef_pose,
                                                                grasp_pose,
                                                                task.eef_jac))
                pos_action[:, 7:] = torch.Tensor([[gripper_amt, gripper_amt]] * task.num_envs).to(args.device)
                
                if gripper_amt == 0.0001:
                    lift_object = True
                    print('Object grasped')
                    
            elif lift_object: # finally, lift the object
                lift_pose = grasp_pose.clone()
                lift_pose[:, 2] = lift_pose[:, 2] + 0.35
                lift_pose[:, 0] = lift_pose[:, 0] + 0.1
                lift_pose[:, 3:7] = torch.from_numpy(R.from_euler('xyz', [0, 0, -35], degrees=True).as_quat())
                
                draw_axes(task.gym, task.viewer, task.envs[0], lift_pose[0, :3], lift_pose[0, 3:7])
                
                pos_action[:, :7] = dof_pos + 0.1 * (controller.compute_pos_action(current_eef_pose,
                                                                                    lift_pose,
                                                                                    task.eef_jac))
                pos_action[:, 7:] = torch.Tensor([[gripper_amt, gripper_amt]] * task.num_envs).to(args.device)
            
            
            task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(pos_action))
        
        #! [1] Capture Images and exit
        if capture_image and counter == 5:
            rgb_filename = f'{CAPTURED_IMAGES_PATH}/rgb.png'
            task.gym.write_camera_image_to_file(task.sim, 
                                                task.envs[0], 
                                                task.camera_handles[0], 
                                                gymapi.IMAGE_COLOR, 
                                                rgb_filename)
            depth_filename = f'{CAPTURED_IMAGES_PATH}/depth.npy'
            depth = task.gym.get_camera_image(task.sim, 
                                        task.envs[0], 
                                        task.camera_handles[0], 
                                        gymapi.IMAGE_DEPTH)
            np.save(depth_filename, depth)

            seg_filename = f'{CAPTURED_IMAGES_PATH}/segmentation.npy'
            seg = task.gym.get_camera_image(task.sim, 
                                        task.envs[0], 
                                        task.camera_handles[0], 
                                        gymapi.IMAGE_SEGMENTATION)
            
            np.save(seg_filename, seg)
            
            projection_matrix = np.matrix(task.gym.get_camera_proj_matrix(task.sim, 
                                                                          task.envs[0], 
                                                                          task.camera_handles[0]))

            view_matrix = np.matrix(task.gym.get_camera_view_matrix(task.sim, 
                                                                task.envs[0], 
                                                                task.camera_handles[0]))
            
            np.save(f'{CAPTURED_IMAGES_PATH}/projection_matrix.npy', projection_matrix)
            np.save(f'{CAPTURED_IMAGES_PATH}/view_matrix.npy', view_matrix)
            
            print(f'Captured images and saved to {CAPTURED_IMAGES_PATH}')
            break
            
        task.gym.step_graphics(task.sim)
        task.gym.draw_viewer(task.viewer, task.sim, False)
        task.gym.sync_frame_time(task.sim)
        
        counter += 1
            
        clear_axes(task)
    
    task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)


if __name__ == "__main__":
    main()