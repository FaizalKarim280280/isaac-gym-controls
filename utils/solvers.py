from .quintic_polynomial import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class QuinticSolver:
    def __init__(self):
        pass

    def pre_grasp_pose(self, grasp_pose, r, grasp_axis_local):
        # Extract position and orientation
        ([x, y, z], [qx, qy, qz, qw]) = grasp_pose

        # Convert quaternion to rotation matrix
        self.rotation = R.from_quat([qx, qy, qz, qw])
        self.rotation_matrix = self.rotation.as_matrix()

        # Convert the grasp axis from local to world frame
        self.grasp_axis_world = self.rotation_matrix @ grasp_axis_local

        # Compute the offset position
        self.offset_position = np.array([x, y, z]) - r * self.grasp_axis_world

        # Create the pre-grasp pose
        self.pre_grasp_pose = ([
            self.offset_position[0],
            self.offset_position[1],
            self.offset_position[2]],
            [qx, qy, qz, qw])

        return self.pre_grasp_pose
    
    def mujoco2normal(self,q):
        return np.array([q[1], q[2], q[3], q[0]])

    def normal2mujoco(self,q):
        return np.array([q[3], q[0], q[1], q[2]])

    def genSlerp(self,x_init_quat, x_final_quat, final_n, n_steps):
        self.X_init_quat = self.mujoco2normal(x_init_quat)
        self.X_final_quat = self.mujoco2normal(x_final_quat)
        
        self.rot_times = np.array([0, final_n])
        self.rots = R.from_quat([self.X_init_quat, self.X_final_quat])
    
        
        self.slerpObj = Slerp(self.rot_times, self.rots)
        self.times = np.linspace(0, final_n, n_steps)
        
        # Perform the interpolation and convert the result back to MuJoCo format
        self.interpolated_rots = self.slerpObj(self.times)
        self.interpolated_quats = self.interpolated_rots.as_quat()
        
        # Convert the interpolated quaternions back to MuJoCo format
        self.mujoco_quats = np.array([self.normal2mujoco(q) for q in self.interpolated_quats])
        
        return self.mujoco_quats
    
    def create_trajectory(self, init_pose, final_pose, steps, waypoint=None):
        self.traj = []
        
        self.init_quat = np.array([init_pose[1][0],init_pose[1][1],init_pose[1][2],init_pose[1][3]])  #wxyz mujoco - mocap is wxyz
        self.final_quat = np.array([final_pose[1][0],final_pose[1][1],final_pose[1][2],final_pose[1][3]])
    
        self.x_poly = QuinticPolynomial(init_pose[0][0], 0, 0, final_pose[0][0], 0, 0, steps, waypoint[0] if waypoint is not None else None)
        self.y_poly = QuinticPolynomial(init_pose[0][1], 0, 0, final_pose[0][1], 0, 0, steps, waypoint[1] if waypoint is not None else None)
        self.z_poly = QuinticPolynomial(init_pose[0][2], 0, 0, final_pose[0][2], 0, 0, steps, waypoint[2] if waypoint is not None else None)
        self.q = self.genSlerp(self.init_quat ,self.final_quat, steps,steps)

        
        for i in range(steps):
            self.pos = [self.x_poly.calc_point(i), self.y_poly.calc_point(i), self.z_poly.calc_point(i)]
            self.quat = self.q[i].tolist()
        
            self.traj.append([self.pos, self.quat])

        return self.traj
    
    def generate_end_effector_trajectories(self,object_trajectory, current_obj_pose, current_pose_L, current_pose_R):
        self.AtrajL = []
        self.AtrajR = []

        # Extract current object pose
        self.obj_pos = np.array(current_obj_pose[0])
        self.obj_quat = np.array(current_obj_pose[1])

        # Extract current end effector poses
        self.pos_L = np.array(current_pose_L[0])
        self.quat_L = np.array(current_pose_L[1])
        self.pos_R = np.array(current_pose_R[0])
        self.quat_R = np.array(current_pose_R[1])

        # Calculate inverse object rotation matrix
        self.inv_obj_R = self.quaternion_to_rotation_matrix(self.quat_inverse(self.obj_quat))

        # Calculate transformations from the object to the end effectors
        self.obj_to_ee_L_pos = self.inv_obj_R @ (self.pos_L - self.obj_pos)
        self.obj_to_ee_L_quat = self.quaternion_multiply(self.quat_inverse(self.obj_quat), self.quat_L)

        self.obj_to_ee_R_pos = self.inv_obj_R @ (self.pos_R - self.obj_pos)
        self.obj_to_ee_R_quat = self.quaternion_multiply(self.quat_inverse(self.obj_quat), self.quat_R)

        self.obj_to_ee_L = np.concatenate([self.obj_to_ee_L_pos, self.obj_to_ee_L_quat])
        self.obj_to_ee_R = np.concatenate([self.obj_to_ee_R_pos, self.obj_to_ee_R_quat])

        for obj_pose in object_trajectory:
            self.pos = np.array(obj_pose[0])
            self.quat = np.array(obj_pose[1])
            
            # Convert quaternion to rotation matrix
            self.R = self.quaternion_to_rotation_matrix(self.quat)

            # Left end effector pose
            self.ee_L_pos = self.pos + self.R @ self.obj_to_ee_L[:3]
            self.ee_L_quat = self.quaternion_multiply(self.quat, self.obj_to_ee_L[3:])
            self.AtrajL.append([self.ee_L_pos.tolist(),self.ee_L_quat.tolist()])

            # Right end effector pose
            self.ee_R_pos = self.pos + self.R @ self.obj_to_ee_R[:3]
            self.ee_R_quat = self.quaternion_multiply(self.quat, self.obj_to_ee_R[3:])
            self.AtrajR.append([self.ee_R_pos.tolist(),self.ee_R_quat.tolist()])

        return self.AtrajL, self.AtrajR


    def quaternion_to_rotation_matrix(self,q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def quat_inverse(self,q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    def quaternion_multiply(self,q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])