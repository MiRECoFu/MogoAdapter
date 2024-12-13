from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from tools import *

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, device, attn_mask: torch.Tensor = None):
        super().__init__()

        # Cross-attention: Query comes from target (x), key and value come from source (y)
        self.attn = nn.MultiheadAttention(d_model, n_head).to(device)
        self.ln_1 = nn.LayerNorm(d_model).to(device)
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout", nn.Dropout(p=0.1))
        ])).to(device)
        self.ln_2 = nn.LayerNorm(d_model).to(device)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        """x is the query (target input), y is the key and value (source input)."""
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: target input tensor (query for attention)
        y: refer motion input tensor (key and value for attention)
        """
        # Apply cross-attention: x (target) attends to y (source)
        x = x + self.attention(self.ln_1(x), y)  # Cross-attention between x (target) and y (source)
        
        # Apply MLP
        x = x + self.mlp(self.ln_2(x))
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, device, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = [ResidualCrossAttentionBlock(width, heads, device, attn_mask) for _ in range(layers)]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        for resblock in self.resblocks:
            x = resblock(x, y) 
        return x
    
    
class MogoAdapter(nn.Module):
    def __init__(self,
                 max_motion_length: int,
                 layers: int,
                 heads: int,
                 width: int,
                 mogo_clip_embed_dim: int,
                 mogo_dim: int,
                 mogo_q_layers: int,
                 scale: int,
                 vq_model,
                 device
                ):
        super().__init__()
        
        self.max_motion_length = max_motion_length
        self.device = device
        self.mogo_clip_embed_dim = mogo_clip_embed_dim
        self.mogo_dim = mogo_dim
        self.mogo_q_layers = mogo_q_layers
        self.width = width
        
        self.layers = layers
        self.heads = heads
        
        self.scale = scale
        self.vq_model = vq_model
        self.vq_model.eval()
        for param in self.vq_model.parameters():
            param.requires_grad = False
        
        self.trms = nn.ModuleList([])
        
        for i in range(mogo_q_layers):
            cur_heads = self.heads // (i + 1)
            cur_layers = self.layers // (i + 1)
            trm = Transformer(
                width=width,
                layers=cur_layers,
                heads=cur_heads,
                device=device,
                attn_mask=self.build_attention_mask(),
            ).to(self.device)
            self.trms.append(trm)
            
        
        
        self.positional_embedding = nn.Parameter(torch.empty(self.max_motion_length, width))
        self.ln_final = LayerNorm(width)
        self.cond_emb = nn.Linear(self.mogo_clip_embed_dim, self.mogo_dim)
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        nn.init.normal_(self.cond_emb.weight, std=(2 * self.mogo_dim) ** -0.5)
        
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for transformer in self.trms:
            for block in transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)



            
        
    
    def build_attention_mask(self):
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.max_motion_length, self.max_motion_length).to(self.device)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def forward(self, input_motion_logits, refer_text_features, refer_motion_clip_features, refer_motion_lens, is_generate=False):
        bs, m_len, motion_dim, q = input_motion_logits.shape
        refer_motion_features = self.cond_emb(refer_motion_clip_features)
        refer_motion_features = refer_motion_features.unsqueeze(1).unsqueeze(-1)
        refer_motion_features = refer_motion_features.repeat(1, m_len, 1, q).type(input_motion_logits.dtype).to(self.device)
        # print(f"refer_motion_features: {refer_motion_features} {refer_motion_features.shape} input_motion_logits: {input_motion_logits.shape}")
        res_atts = []
        for ind, transformer in enumerate(self.trms):
            input_motion_layer_logits = input_motion_logits[:, :, :, ind].permute(1, 0, 2)
            refer_motion_layer_features = refer_motion_features[:, :, :, ind].permute(1, 0, 2)
            att = transformer(input_motion_layer_logits, refer_motion_layer_features)
            print(f"cross att res: {att}, {att.shape}")
            att = att.permute(1, 0, 2)
            res_atts.append(att)
        res_features = torch.stack(res_atts, dim=-1)
        print(f"res_features res: {res_features}, {res_features.shape}")
