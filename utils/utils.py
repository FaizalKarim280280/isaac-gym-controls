from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco
import roma 
import torch

def quat2rot(quat, informat, outformat, degrees):
    if informat == "xyzw":
        r = R.from_quat(quat)
        return (r.as_euler(outformat, degrees))
    elif informat == "wxyz":
        temp_quat = quat[1:]
        temp_quat = np.append(temp_quat, quat[0])
        r = R.from_quat(temp_quat)
        return (r.as_euler(outformat, degrees))
    else:
        raise (ValueError)

def eef_pose(data, eefname):
    eef_pos = data.site(eefname).xpos
    eef_quat = np.zeros(4)
    mujoco.mju_mat2Quat(eef_quat, data.site(eefname).xmat)
    eef_pose = np.concatenate((eef_pos, eef_quat))
    return eef_pose

def torch_tmat2pose(mat, scale, obj_str_pos, obj_str_ori):
    # if not mat.shape == (4, 4):
    #     raise ValueError("matrix must be of shape 4x4 ")
    
    pos = mat[:, :3, 3] * scale # pos is in object frame  
    pos = pos + obj_str_pos # pos is in world frame 
    rot = mat[:, :3, :3] # (batch, 3, 3)
    rot_tf = roma.unitquat_to_rotmat(obj_str_ori) # (batch, 3, 3)
    rot = rot @ rot_tf 
    quat = roma.rotmat_to_unitquat(rot) # (batch, 4)
    
    return torch.cat([pos, quat], dim=-1) # (batch, 7)

def tmat2pose(mat, scale, obj_str_pos, obj_str_ori):
    if (not mat.shape == (4, 4)):
        raise ValueError("matrix must be of shape 4x4 ")

    pos = (mat[:3, 3] * scale)  # `mat[:3, 3]` already gives a (3,) array
    pos = pos + obj_str_pos # Ensure obj_str_pos is a compatible shape (3,)
    rot = mat[:3, :3]
    rot_tf = R.from_euler(
        "xyz", (obj_str_ori[0], obj_str_ori[1], obj_str_ori[2]), degrees=True)
    rot_tform = rot_tf.as_matrix()
    rot = rot @ rot_tform
    quat = np.zeros(4)
    quat = R.from_matrix(rot).as_quat() 
    # mujoco.mju_mat2Quat(quat, rot.flatten())

    return [pos.tolist(), quat.tolist()]

def torch_gen_pre_grasp(grasp, offset):
    # x, y, z = grasp[:, :3]
    rmat = roma.unitquat_to_rotmat(grasp[:, 3:])
    grasp_axis = rmat[:, :, 1]
    offset_pos = grasp[:, :3] - offset * grasp_axis
    pre_grasp_pose = torch.cat([offset_pos, grasp[:, 3:]], dim=-1)
    return pre_grasp_pose
    

def gen_pre_grasp(grasp, offset):
    x, y, z = grasp[0]
    qw, qx, qy, qz = grasp[1]

    rot = R.from_quat([qx, qy, qz, qw])
    rmat = rot.as_matrix()

    grasp_axis = rmat[:, 1]

    offset_pos = np.array([x, y, z]) + offset * grasp_axis

    pre_grasp_pose = [offset_pos.tolist(), [qw, qx, qy, qz]]
    return pre_grasp_pose

def safe_matrix_sqrt(matrix):
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(matrix)

    # Take square root of the absolute values of the eigenvalues
    sqrt_eigvals = np.sqrt(np.abs(eigvals))

    # Reconstruct the square root of the matrix
    sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

    return sqrt_matrix

def compute_damping_matrices(mx_l, mx_r, k):
    # Compute the square roots using the safe square root function
    sqrt_mx_l = safe_matrix_sqrt(mx_l)
    sqrt_mx_r = safe_matrix_sqrt(mx_r)
    sqrt_k = np.sqrt(k)  # K has no negative eigenvalues

    # Compute the damping matrices
    d_l = sqrt_mx_l @ sqrt_k + sqrt_k @ sqrt_mx_l
    d_r = sqrt_mx_r @ sqrt_k + sqrt_k @ sqrt_mx_r

    return d_l, d_r
