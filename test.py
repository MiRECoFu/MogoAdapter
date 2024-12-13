import clip
import os
import torch
import numpy as np
import sys
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.join(current_dir, '..'))
from motion_vae.model import RVQVAE
from options.train_opt import TrainT2MOptions
from utils.get_opt import get_opt
from utils.paramUtil import t2m_kinematic_chain
from utils.util import *
from data_process.motion_dataset import Text2MotionDataset
from mogo_clip_models.mogo_clip import MogoClip
from mogo_models.transformers.transformotion import Transformotion
from models.mogo_adapter import MogoAdapter


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, "cuda")
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
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'model', 'net_best_fid.tar'),
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
    opt.data_root = '/root/autodl-tmp/HumanML3D'
    print(opt)
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.joints_num = 22
    opt.max_motion_len = 55
    opt.text_dir = pjoin(opt.data_root, 'texts')
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
    print(vq_opt)
    
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
    # print(vq_model)
    
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))
    
    train_file = pjoin(opt.data_root, 'test.txt')
    dataset = Text2MotionDataset(opt, mean, std, train_file)
    seed1 = 42
    seed2 = 623
    set_new_seed(seed1) 
    loader1 = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True)
    set_new_seed(seed2)
    loader2 = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True)
    for i, (batch_data, batch_data2) in enumerate(zip(loader1, loader2)):
        captions, motions, m_lens = batch_data
        motions = motions.detach().float().to(opt.device)
        code_idx, _motion_emb = vq_model.encode(motions)
        motion_code = code_idx[:, :, 0]
        batch_size = motions.shape[0]
        positive_mean, negative_mean, separation = mogo_clip.mean_cosine_similarity(motion_code, captions)
        motion_code_feature = mogo_clip.encode_motion_code(motion_code)
        ce_loss, acc, pred_id, output, logits, all_attends_out = t2m_transformer(captions, code_idx, m_lens, code_idx.clone(), has_adapter=True)
        
        print(f"positive_mean1: {positive_mean} negative_mean1:{negative_mean}")
        # print(f"motion_code_feature: {motion_code_feature} motion_code_feature shape: {motion_code_feature.shape}")
        # print(f"mogo output: {all_attends_out.shape}")
        
        captions2, motions2, m_lens2 = batch_data2
        motions2 = motions2.detach().float().to(opt.device)
        code_idx2, _motion_emb2 = vq_model.encode(motions2)
        motion_code2 = code_idx2[:, :, 0]
        # batch_size = motions.shape[0]
        positive_mean2, negative_mean2, separation2 = mogo_clip.mean_cosine_similarity(motion_code2, captions2)
        input_motion_logits = all_attends_out.to(opt.device)
        motion_code_feature2 = mogo_clip.encode_motion_code(motion_code2).to(opt.device)
        text_feature2 = mogo_clip.encode_text(captions2).to(opt.device)
        mogo_adapter(all_attends_out, text_feature2, motion_code_feature2)
        print(f"positive_mean2: {positive_mean2} negative_mean1:{negative_mean2}")
        if i == 5:
            break