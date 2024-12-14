import os
import torch
import numpy as np
import sys

from torch.utils.data import DataLoader
from os.path import join as pjoin
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.join(current_dir, '..'))
from models.trainer import Trainer
from mogo_clip_models.mogo_clip import MogoClip
from mogo_models.transformers.transformotion import Transformotion
from models.mogo_adapter import MogoAdapter
from motion_vae.model import RVQVAE

from options.train_opt import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data_process.motion_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.util import *

def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
        

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt


def load_mogo_clip():
    mogo_clip = MogoClip(
        embed_dim=opt.mogo_clip_embed_dim,
        layers=opt.mogo_clip_layers,
        heads=opt.mogo_clip_heads,
        width=opt.mogo_clip_width,
        codebook_size=vq_opt.nb_code,
        max_motion_length=opt.max_motion_length,
        clip_version='ViT-L/14',
        device=opt.device
    )
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.mogo_clip_name, 'model', 'best_eval_cosine.tar'),
                            map_location=opt.device)
    model_key = 'mogo_clip'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = mogo_clip.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading mogo_clip {model_opt.mogo_clip_name} from epoch {ckpt["ep"]}!')
    return mogo_clip

def load_trans_model(model_opt, which_model, vq_model, opt):
    clip_version = 'ViT-B/32'
    transformotion = Transformotion(code_dim=model_opt.code_dim, 
                                    vq_model=vq_model, 
                                    clip_dim=512,
                                    clip_version=clip_version,
                                    opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.mogo_name, 'model', which_model),
                      map_location=opt.device)
    model_key = 'transformotion'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = transformotion.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading transformotion Transformer {opt.mogo_name} from epoch {ckpt["ep"]}!')
    return transformotion



if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    
    opt.data_root = '/root/autodl-tmp/HumanML3D'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 22
    opt.max_motion_len = 55
    dim_pose = 263
    radius = 4
    fps = 20
    kinematic_chain = t2m_kinematic_chain
    dataset_opt_path = '/root/autodl-tmp/checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model()
    opt.num_tokens = vq_opt.nb_code
    mogo_adapter = MogoAdapter(
        max_motion_length=opt.max_motion_length,
        layers=opt.layers,
        heads=opt.heads,
        width=opt.width,
        mogo_clip_embed_dim=opt.mogo_clip_embed_dim,
        mogo_dim=1024,
        mogo_q_layers=6,
        scale=1,
        num_tokens=vq_opt.nb_code,
        device=opt.device
    )
    mogo_adapter.to(opt.device)

    vq_model.to(opt.device)
    vq_model.eval()
    for param in vq_model.parameters():
        param.requires_grad = False
    
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim
    t2m_transformer = load_trans_model(model_opt, 'latest.tar', vq_model=vq_model, opt=opt)
    t2m_transformer.to(opt.device)
    t2m_transformer.eval()
    for param in t2m_transformer.parameters():
        param.requires_grad = False
    
    mogo_clip = load_mogo_clip()
    mogo_clip.to(opt.device)
    mogo_clip.eval()
    for param in mogo_clip.parameters():
        param.requires_grad = False

    all_params = 0
    pc_transformer = sum(param.numel() for param in mogo_adapter.parameters())

    # print(t2m_transformer)
    # print("Total parameters of t2m_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))
    
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)
    print(f"================================================={len(train_dataset)}")
    
    seed1 = 42
    seed2 = 623
    set_new_seed(seed1) 

    train_loader1 = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    eval_val_loader1, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)
    set_new_seed(seed2) 
    
    train_loader2 = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    eval_val_loader2, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    trainer = Trainer(opt, t2m_transformer, vq_model, mogo_clip, mogo_adapter)
    
    trainer.train(train_loader1, train_loader2, val_loader, eval_val_loader1, eval_val_loader2, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)