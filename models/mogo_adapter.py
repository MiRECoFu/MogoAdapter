from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mogo_models.transformers.tools import *
# from tools import *

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
    def __init__(self, d_model: int, n_head: int, device):
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
        # self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        """x is the query (target input), y is the key and value (source input)."""
        attn_mask = self.build_attention_mask(x).to(dtype=x.dtype, device=x.device)
        return self.attn(x, y, y, need_weights=False, attn_mask=attn_mask)[0]
    
    def build_attention_mask(self, x: torch.Tensor):
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(x.shape[0], x.shape[0]).to(x.device)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

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
    def __init__(self, width: int, layers: int, heads: int, device):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = [ResidualCrossAttentionBlock(width, heads, device) for _ in range(layers)]

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
                 num_tokens: int,
                 device
                ):
        super().__init__()
        
        self.max_motion_length = max_motion_length
        self.device = device
        self.mogo_clip_embed_dim = mogo_clip_embed_dim
        self.mogo_dim = mogo_dim
        self.mogo_q_layers = mogo_q_layers
        self.width = width
        self.num_tokens = num_tokens
        
        
        self.layers = layers
        self.heads = heads
        
        self.scale = scale

        self.trms = nn.ModuleList([])
        
        for i in range(mogo_q_layers):
            cur_heads = self.heads // (i + 1)
            cur_layers = self.layers // (i + 1)
            trm = Transformer(
                width=width,
                layers=cur_layers,
                heads=cur_heads,
                device=device,
                # attn_mask=self.build_attention_mask(),
            ).to(self.device)
            self.trms.append(trm)
            
        
        
        self.positional_embedding = nn.Parameter(torch.empty(self.max_motion_length, width))
        self.ln_final = LayerNorm(width)
        self.cond_emb = nn.Linear(self.mogo_clip_embed_dim, self.mogo_dim)
        self.head = nn.Linear(self.mogo_dim, self.num_tokens, bias=False) 
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        nn.init.normal_(self.cond_emb.weight, std=(2 * self.mogo_dim) ** -0.5)
        nn.init.normal_(self.head.weight, std=(2 * self.mogo_dim) ** -0.5)
        
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for transformer in self.trms:
            for block in transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                
    def parameters(self):
        return [p for name, p in self.named_parameters()]

    
    # def build_attention_mask(self):
    #     # pytorch uses additive attention mask; fill with -inf
    #     mask = torch.empty(self.max_motion_length, self.max_motion_length).to(self.device)
    #     mask.fill_(float("-inf"))
    #     mask.triu_(1)  # zero out the lower diagonal
    #     return mask
    
    def forward(self, input_motion_logits, refer_motion_clip_features, scale=1):
        bs, m_len, motion_dim, q = input_motion_logits.shape
        refer_motion_features = self.cond_emb(refer_motion_clip_features)
        refer_motion_features = refer_motion_features.unsqueeze(1).unsqueeze(-1)
        refer_motion_features = refer_motion_features.repeat(1, m_len, 1, q).type(input_motion_logits.dtype).to(self.device)
        # print(f"refer_motion_features: {refer_motion_features} {refer_motion_features.shape} input_motion_logits: {input_motion_logits.shape}")
        res_atts = []
        all_out = []
        for ind, transformer in enumerate(self.trms):
            input_motion_layer_logits = input_motion_logits[:, :, :, ind].permute(1, 0, 2)
            refer_motion_layer_features = refer_motion_features[:, :, :, ind].permute(1, 0, 2)
            att = transformer(input_motion_layer_logits, refer_motion_layer_features)
            res_feat = scale * att + input_motion_layer_logits
            att = att.permute(1, 0, 2)
            res_feat = res_feat.permute(1, 0, 2)
            out = self.head(res_feat)
            res_atts.append(res_feat)
            all_out.append(out)
        res_features = torch.stack(res_atts, dim=-1)
        res_all_out = torch.stack(all_out, dim=-1)
        # print(f"res_features res: {res_features}, {res_features.shape}, res_all_out: {res_all_out}, {res_all_out.shape}")
        return res_features, res_all_out
    
    @torch.no_grad()
    @eval_decorator
    def generate(self, prompt_texts, refer_motion_clip_features, m_lens, transformotion, vq_model, labels=None, scale=1):

        self.eval()
        for ind, trm in enumerate(self.trms):
            trm.eval()
        seq_len = max(m_lens).to(self.device)
        batch_size = len(m_lens)
        if labels is None:
            labels = torch.zeros(batch_size, seq_len, 6)
            
        res_seq_ids = []
        
        generated = torch.empty(batch_size, 0, self.mogo_q_layers, dtype=torch.long).to(self.device)
        for k in range(seq_len):
            out, all_attends_out = transformotion(prompt_texts, generated, m_lens, labels=labels[:, k:k+1], mems=None, is_generating=True, has_adapter=True)
            res_features, res_all_out = self.forward(all_attends_out, refer_motion_clip_features, scale=scale)
            logits = res_all_out.permute(0, 1, 3, 2)
            probs = F.softmax(logits[:,-1,:, :], dim=-1)  # (b, seqlen, ntoken)
            dist = Categorical(probs)
            pred_ids = dist.sample()
            pred_ids = pred_ids.unsqueeze(1)
            res_seq_ids.append(pred_ids)
            generated = torch.cat(res_seq_ids, dim=1)
        motion_ids = torch.cat(res_seq_ids, dim=1).to(self.device)
        print(f"motion motion_ids ========================+> {motion_ids}\n labels====> {labels}")
        # gathered_ids = repeat(motion_ids.unsqueeze(-1), 'b n -> b n d', d=6)
        pred_motions = vq_model.forward_decoder(motion_ids)
        # print(f"motion pred_motions ========================+> {pred_motions.shape}\n labels========================+> {labels.shape}")
        for ind, trm in enumerate(self.trms):
            trm.train()
        # self.seqTransDecoderXL.train()
    
        return pred_motions
        

