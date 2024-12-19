from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn
import clip
from mogo_models.transformers.tools import *
from models.qna import *
from models.tools import *


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)
    

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

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
    
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        padding_mode='zeros',
        padding=1
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        elif down:
            self.h_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    
# use qna attention block
class AttentionBlock(nn.Module):
    def __init__(
            self,
            channels,
            num_heads=1,
            use_new_attention_order=False,
            use_qna=True,
            kernel_size=3,
            padding=1,
            padding_mode='zeros',
            device=torch.device("cuda")
            ):
        super().__init__()
        self.channels = channels
        self.use_qna = use_qna
        self.num_heads = num_heads
        self.norm = normalization(channels)

        # self.norm_2 = normalization(channels)
        # TODO not use qna
        if use_qna:
            self.attention = FusedQnA1d(
                in_features=self.channels,
                timesteps_features=None,
                hidden_features=self.channels,
                heads=self.num_heads,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )
    
    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        # q = q.reshape(b, c, -1)
        h = self.norm(x).reshape(b, c, 1, -1)
        h = self.attention(h)
        h = h.reshape(b, c, -1)
        # print(f"h============{h.shape}")
        # h = self.mlp(self.norm_2(h).permute(0, 2, 1)).permute(0, 2, 1)
        return  (x + h).reshape(b, c, *spatial)
    

    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, device):
        super().__init__()
        self.width = width
        self.layers = layers
        self.inputblocks = nn.ModuleList([ResBlock(
                            width,
                            width,
                            dropout=0.1,
                            out_channels=width,
                        ) for _ in range(layers // 2)])
        self.qnablocks = nn.ModuleList([AttentionBlock(width, num_heads=heads, device=device) for _ in range(layers)])
        self.outputblocks = nn.ModuleList([ResBlock(
                            width,
                            width,
                            dropout=0.1,
                            out_channels=width,
                        ) for _ in range(layers // 2)])

    def forward(self, q: torch.Tensor, x: torch.Tensor, scale=1):
        for resblock in self.inputblocks:
            x = resblock(x, q)
        for qnablock in self.qnablocks:
            x = qnablock(x) 
        for resblock in self.outputblocks:
            x = resblock(x, q)
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
        
        self.transformer = Transformer(
            width=width,
            layers=self.layers,
            heads = self.heads,
            device = device
        ).to(self.device)
            
        
        
        self.positional_embedding = nn.Parameter(torch.empty(self.max_motion_length, width))
        self.ln_final = LayerNorm(width)
        self.cond_emb = nn.Linear(self.mogo_clip_embed_dim, self.mogo_dim)
        self.input_emb = nn.Linear(self.mogo_dim, self.mogo_dim)
        self.head = nn.Linear(self.mogo_dim, self.num_tokens, bias=False) 
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        nn.init.normal_(self.cond_emb.weight, std=(2 * self.mogo_dim) ** -0.5)
        nn.init.normal_(self.input_emb.weight, std=(2 * self.mogo_dim) ** -0.5)
        nn.init.normal_(self.head.weight, std=(2 * self.mogo_dim) ** -0.5)
        

                
    def parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('mogo.')]

    
    def forward(self, input_motion_logits, refer_motion_clip_features, scale=1):
        # print(f"input_motion_logits.shape==={input_motion_logits.shape}")
        bs, m_len, q, motion_dim = input_motion_logits.shape
        refer_motion_features = self.cond_emb(refer_motion_clip_features)
        input_motion_logits = self.input_emb(input_motion_logits).permute(0, 1, 3, 2)
        input_motion_logits = input_motion_logits.permute(0, 2, 1, 3)
        att = self.transformer(refer_motion_features, input_motion_logits, scale=scale)
        att = att.permute(0, 2, 1, 3)
        all_out = []
        for ind in range(self.mogo_q_layers):
            res_feat = att[:, :, :, ind]
            out = self.head(res_feat)
            all_out.append(out)
        res_all_out = torch.stack(all_out, dim=-1)

        return att, res_all_out
    
    @torch.no_grad()
    @eval_decorator
    def generate(self, prompt_texts, refer_motion_clip_features, m_lens, transformotion, vq_model, labels=None, scale=1):

        self.eval()
        
        seq_len = max(m_lens).to(self.device)
        batch_size = len(m_lens)
        if labels is None:
            labels = torch.zeros(batch_size, seq_len, 6)
            
        res_seq_ids = []
        
        generated = torch.empty(batch_size, 0, self.mogo_q_layers, dtype=torch.long).to(self.device)
        for k in range(seq_len):
            logits, all_attends_out = transformotion(prompt_texts, generated, m_lens, labels=labels[:, k:k+1], mems=None, is_generating=True, has_adapter=True)
            logits = logits.permute(0, 1, 3, 2)
            
            probs = F.softmax(logits[:,-1,:, :], dim=-1)
            dist = Categorical(probs)
            pred_ids = dist.sample()
            pred_ids = pred_ids.unsqueeze(1)
            res_seq_ids.append(pred_ids)
            generated = torch.cat(res_seq_ids, dim=1)
            
            
            # all_attends_out = all_attends_out.permute(0, 1, 3, 2)
            # res_features, res_all_out = self.forward(all_attends_out, refer_motion_clip_features, scale=scale)
            # logits = res_all_out.permute(0, 1, 3, 2)
            # probs = F.softmax(logits[:,-1,:, :], dim=-1)  # (b, seqlen, ntoken)
            # dist = Categorical(probs)
            # pred_ids = dist.sample()
            # pred_ids = pred_ids.unsqueeze(1)
            # res_seq_ids.append(pred_ids)
            # generated = torch.cat(res_seq_ids, dim=1)
        motion_ids = torch.cat(res_seq_ids, dim=1).to(self.device)
        # print(f"motion motion_ids ========================+> {motion_ids}\n labels====> {labels}")
        input_motion_logits = transformotion.tok_emb(motion_ids)
        res_features, res_all_out = self.forward(input_motion_logits, refer_motion_clip_features, scale=scale)
        out = res_all_out.permute(0, 1, 3, 2)
        probs = F.softmax(out, dim=-1)  # (b, seqlen, ntoken)
        dist = Categorical(probs)
        pred_ids = dist.sample()
        print(f"motion motion_ids ========================+> {pred_ids}\n labels====> {labels}")
        # gathered_ids = repeat(motion_ids.unsqueeze(-1), 'b n -> b n d', d=6)
        pred_motions = vq_model.forward_decoder(pred_ids)
        # print(f"motion pred_motions ========================+> {pred_motions.shape}\n labels========================+> {labels.shape}")
        # self.seqTransDecoderXL.train()
    
        return pred_motions
        

