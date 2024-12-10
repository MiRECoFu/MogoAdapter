import os
import torch
import numpy as np
import sys
from os.path import join as pjoin
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.join(current_dir, '..'))
from Motion import BVH
from Motion.transforms import quat2repr6d
# from Motion.MotionRepresentation import *
from common.skeleton import Skeleton
from common.quaternion import *
from utils.paramUtil import *
from os.path import join as pjoin

from tqdm import tqdm
import pickle
import smplx
from utils import bvh, quat
from human_body_prior.body_model.body_model import BodyModel
from scipy.spatial.transform import Rotation as R
from bvh import Bvh
import math

def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


joint_names = [
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftFoot",
    "RightFoot",
    "Spine2",
    "LeftToe",
    "RightToe",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    # "LeftThumb",
    # "RightThumb",
]


if __name__ == "__main__":
    ex_fps = 20
    # trans_matrix = np.array([[1.0, 0.0, 0.0],
    #                         [0.0, 0.0, 1.0],
    #                         [0.0, 1.0, 0.0]])
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
    bvh_path = '/root/MogoAdapter/FQwW.bvh'
    model_path = '/root/MogoAdapter/body_models'
    model_type = 'smpl'
    gender = 'MALE'
    male_bm_path = './body_models/smplh/male/model.npz'
    male_dmpl_path = './body_models/dmpls/male/model.npz'

    female_bm_path = './body_models/smplh/female/model.npz'
    female_dmpl_path = './body_models/dmpls/female/model.npz'

    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_betas = 10
    # Load SMPL model
    model = smplx.create(model_path=model_path, 
                         model_type=model_type,
                         gender=gender, 
                         batch_size=1)
    
    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(device)
    # parents = model.parents.detach().cpu().numpy()
    bvh_data = bvh.load(bvh_path)
    # gender = 'male'  # 性别
    mocap_framerate = 20  # 帧率
    num_betas = 10  # 形状参数的数量
    betas = np.zeros(num_betas)  # 形状参数，默认全 0
    # dmpls = None  # 动形参数，不使用时设为 None
    
    frame_length = len(bvh_data["rotations"])  # 总帧数
    rotation_correction = R.from_euler('XYZ', [0, 0, 0], degrees=True)
    
    with open(bvh_path, 'r') as f:
        mocap = Bvh(f.read())
    print(mocap)
    num_frames = len(mocap.frames)
    dmpls = np.zeros((frame_length, 8))
    # if n_frames is None:
    #     num_frames = len(mocap.frames)
    # else:
    #     num_frames = min(n_frames, len(mocap.frames))
    # num_frames_downsampled = math.ceil(num_frames)

    poses = np.zeros((frame_length, 156))
    trans = np.zeros((frame_length, 3))
    for i in range(num_frames):
        for joint_index, joint_name in enumerate(joint_names):
            # rotation = R.from_euler('ZXY', mocap.frame_joint_channels(i, joint_name, ['Zrotation', 'Xrotation', 'Yrotation']), degrees=True)
            rotation = R.from_euler('ZYX', mocap.frame_joint_channels(i, joint_name, ['Zrotation', 'Yrotation', 'Xrotation']), degrees=True)
            if joint_name == 'Hips':
                # rotation = rotation * rotation_correction
                rotation = rotation_correction * rotation

            poses[i, 3 * joint_index:3 * (joint_index + 1)] = rotation.as_rotvec()

            if joint_name == 'Hips':
                x, y, z = mocap.frame_joint_channels(i, joint_name, ['Xposition', 'Yposition', 'Zposition'])
                # trans[i] = np.array([x, -z, y])
                trans[i] = np.array([x, y, z])



    np.savez(
                'bvh_pose_data.npz',
                trans=trans,
                gender=gender,
                mocap_framerate=mocap_framerate,
                betas=betas,
                dmpls=dmpls,
                poses=poses
            )
    # # assert len(rotations[0]) == model.NUM_JOINTS, "Joint count mismatch between BVH and SMPL model."
    bdata = np.load('bvh_pose_data.npz', allow_pickle=True)
    frame_number = bdata['trans'].shape[0]
    # fId = 0 # frame id of the mocap sequence
    # pose_seq = []
    
    bm = male_bm
    down_sample = 1
    
    bdata_poses = bdata['poses'][::down_sample,...]
    bdata_trans = bdata['trans'][::down_sample,...]
    body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(device),
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(device),
            'trans': torch.Tensor(bdata_trans).to(device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(device),
        }
    
    with torch.no_grad():
        body = bm(**body_parms)
    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    np.save('bvh_pose_data_1.npy', pose_seq_np_n)
    
    

    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip = 2, 1
    joints_num = 22
    
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    example_id = "000021"
    data_dir = '/root/autodl-tmp/joints/'
    frame_num = 0
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))

    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    
    source_data = np.load('bvh_pose_data_1.npy')[:, :joints_num]
    data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
    rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
    np.save("bvh_data_new_vecs.npy", data)
    print(f"data: {data}")
    frame_num += data.shape[0]
    