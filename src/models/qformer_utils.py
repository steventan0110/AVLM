import math


import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from collections import namedtuple
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


mlist = nn.ModuleList

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# tensor helpers

def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t

    if t.shape[-1] > length:
        return t[..., :length]

    return F.pad(t, (0, length - t.shape[-1]))

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def generate_mask_from_repeats(repeats):
    repeats = repeats.int()
    device = repeats.device

    lengths = repeats.sum(dim = -1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim = -1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value = 0.)

    seq = torch.arange(max_length, device = device)
    seq = repeat(seq, '... j -> ... i j', i = repeats.shape[-1])

    cumsum = rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = rearrange(cumsum_exclusive, '... i -> ... i 1')

    lengths = rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask



Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)


    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask


    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask = mask)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale
        # key padding mask
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash = False,
        cross_attn_include_queries = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal = causal, dropout = dropout, use_flash = use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim = -2)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # mask: (B, N)

        out = self.attend(q, k, v, mask = mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    


class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim = -1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta
    

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_context = None,
        max_latents = 128,  # Changed from num_latents to max_latents
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        use_flash_attn = False,
        use_pos_embed = True
    ):
        super().__init__()
        dim_context = dim_context or dim

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        # Initialize with maximum possible number of latents
        self.latents = nn.Parameter(torch.randn(max_latents, dim))
        nn.init.normal_(self.latents, std = 0.02)
        self.max_latents = max_latents
        
        # Create fixed sinusoidal positional embeddings
        if use_pos_embed:
            self.register_buffer('pos_emb', self._create_sinusoidal_embeddings(max_latents, dim))
        else:
            self.pos_emb = None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    use_flash = use_flash_attn,
                    cross_attn_include_queries = False,
                    causal=False
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)
    
    def _create_sinusoidal_embeddings(self, length, dim):
        """
        Create sinusoidal positional embeddings
        """
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pos_emb = torch.zeros(length, dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        
        # Handle the case where dim is odd
        if dim % 2 == 0:
            pos_emb[:, 1::2] = torch.cos(position * div_term)
        else:
            pos_emb[:, 1::2] = torch.cos(position * div_term[:dim//2])
            
        return pos_emb

    def forward(self, x, num_latents, mask = None):
        """
        Args:
            x: input tensor of shape [batch, seq_len, dim]
            mask: attention mask for x
            num_latents: number of latent queries to use (must be <= max_latents)
        """
        batch = x.shape[0]
        
        # Use all latents if num_latents not specified
        num_latents = min(num_latents, self.max_latents)

        x = self.proj_context(x)

        # Only use the specified number of latents
        latents = self.latents[:num_latents]
        # Add sinusoidal positional embeddings to latents
        if self.pos_emb is not None:
            latents = latents + self.pos_emb[:num_latents]
        
        latents = repeat(latents, 'n d -> b n d', b = batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask = mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)
