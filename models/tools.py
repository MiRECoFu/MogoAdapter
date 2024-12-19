import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from einops import rearrange
from torch.distributions import Categorical

import math

import torch as th
import torch.nn as nn

# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids==special_id)
    return mask

# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# classifier free guidance functions

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

# noise schedules

# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low

def cal_performance(out, labels, m_tokens_len, ignore_index=None, smoothing=0., tk=1):
    # loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    bs = labels.shape[0]
    loss_cls = 0.0
    right_num = 0
    sum = m_tokens_len.sum().item() * 6
    for i in range(bs):
        preds = out[i][:m_tokens_len[i]]
        tgts = labels[i][:m_tokens_len[i]]
        loss_cls += F.cross_entropy(preds, tgts, ignore_index=ignore_index) / bs
        pred_id_k = torch.topk(preds, k=tk, dim=1).indices
        pred_id = pred_id_k[:, 0]
        right_num += (pred_id.flatten(0) == tgts.flatten(0)).sum().item()
    acc = right_num / sum

    return loss_cls, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss


def cal_perfor(cls_pred, target, m_tokens_len, ignore_index):
    # print(f"logitslogits====>logits size{cls_pred.shape}")
    # cls_pred = cls_pred[:, 1:, :]
    cls_pred = cls_pred.contiguous()
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    bs = target.shape[0]
    loss_cls = 0.0
    right_num = 0
    # loss = F.cross_entropy(cls_pred.permute(0, 2, 1), target, ignore_index=ignore_index)
    sum = m_tokens_len.sum().item()
    sum = sum * 6 # 6 rvq layers
    for i in range(bs):
        # loss function     (26), (26, 513)
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i]], target[i][:m_tokens_len[i]]) / bs

        # Accuracy
        probs = torch.softmax(cls_pred[i][:m_tokens_len[i]], dim=-1)

        dist = Categorical(probs)
        cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i]].flatten(0)).sum().item()
    acc = right_num / sum
    # acc, pred = calculate_accuracy(output, labels)
    # output = output.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    # labels = labels.view(-1)              # (batch_size * seq_len,) 
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(output, labels)
    # print(f"================cls_pred_index================\n{cls_pred_index}\n================target================\n {target}")
    return loss_cls, cls_pred_index, acc

def calculate_accuracy(output, labels):
    batch_size, seq_len, _ = output.shape
    pred = torch.argmax(output, dim=-1)  # (batch_size, seq_len)

    # 计算正确预测的数量
    correct = (pred == labels).sum().item()

    # 计算准确率
    accuracy = correct / (batch_size * seq_len)
    
    return accuracy, pred


# This code is based on https://github.com/openai/guided-diffusion
"""
Various utilities for neural networks.
"""


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


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


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_length = length
        ctx.save_for_backward(*args)
        with th.no_grad():
            output_tensors = ctx.run_function(*args[:length])
        return output_tensors

    @staticmethod
    @th.cuda.amp.custom_bwd
    def backward(ctx, *output_grads):
        args = list(ctx.saved_tensors)

        # Filter for inputs that require grad. If none, exit early.
        input_indices = [i for (i, x) in enumerate(args) if x.requires_grad]
        if not input_indices:
            return (None, None) + tuple(None for _ in args)

        with th.enable_grad():
            for i in input_indices:
                if i < ctx.input_length:
                    # Not sure why the OAI code does this little
                    # dance. It might not be necessary.
                    args[i] = args[i].detach().requires_grad_()
                    args[i] = args[i].view_as(args[i])
            output_tensors = ctx.run_function(*args[:ctx.input_length])

        if isinstance(output_tensors, th.Tensor):
            output_tensors = [output_tensors]

        # Filter for outputs that require grad. If none, exit early.
        out_and_grads = [(o, g) for (o, g) in zip(output_tensors, output_grads) if o.requires_grad]
        if not out_and_grads:
            return (None, None) + tuple(None for _ in args)

        # Compute gradients on the filtered tensors.
        computed_grads = th.autograd.grad(
            [o for (o, g) in out_and_grads],
            [args[i] for i in input_indices],
            [g for (o, g) in out_and_grads]
        )

        # Reassemble the complete gradient tuple.
        input_grads = [None for _ in args]
        for (i, g) in zip(input_indices, computed_grads):
            input_grads[i] = g
        return (None, None) + tuple(input_grads)
