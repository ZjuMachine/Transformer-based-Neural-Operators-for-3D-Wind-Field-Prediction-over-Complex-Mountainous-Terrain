# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling use_cross_unet or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )


        # >>> ADD:
        self._enable_log = False
        self._last_slice_weights = None   # (B,H,N,G) on CPU
        self._last_slice_attn = None      # (B,H,G,G) on CPU (optional)

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D


        # 
        # if self._enable_log:
        if getattr(self, '_enable_log', False):
            self._last_slice_weights = slice_weights.detach().cpu()

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim


        self.blocks_new = nn.ModuleList([Patchsolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))


        self.patcher=PointVoxelPatcher()

        # >>> ADD:
        self.log_attn = False    
        self.attn_ctx = None     

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):

        cfd_data, geom_data,eplison = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :] # 1, N, 3+1+3

        
        if self.unified_pos: # False
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None: # None
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            # B N C 
            fx = self.preprocess(x)
            # B N E
            fx = fx + self.placeholder[None, None, :]

        # ------- divid patch------
        pos=x[:,:,:3] # x y z
        patch_id = self.patcher(pos)     # (1,N)
        # ------- point order -------
        perm, inv_perm, sections = patch_sort(patch_id)
        fx_sorted  = fx.gather(1, perm[..., None].expand(-1, -1, fx.size(-1)))  # (1,N,C)

        last_idx = len(self.blocks_new) - 1
        for li, block in enumerate(self.blocks_new):

            if hasattr(block, "Attn") and hasattr(block.Attn, "log_attn"):
                block.Attn.log_attn = (self.log_attn and li == last_idx)
            fx_sorted = block(fx_sorted, sections)

        # >>> ADD
        # if self.log_attn:
        if getattr(self, 'log_attn', False):  
            try:
                last_attn = self.blocks_new[-1].Attn

                local_patch_attn = None
                slice_weights = None
                alpha = None
                if hasattr(last_attn, "_last_alpha"):
                    alpha = float(last_attn._last_alpha)
                elif hasattr(last_attn, "gate"):

                    alpha = float(torch.sigmoid(last_attn.gate).item())


                if hasattr(last_attn, "_last_debug") and last_attn._last_debug is not None:
                    local_patch_attn = last_attn._last_debug.get("local_patch_attn", None)
                    slice_weights     = last_attn._last_debug.get("slice_weights", None)
                else:

                    if hasattr(last_attn, "local_attn") and hasattr(last_attn.local_attn, "_last_patch_attn"):
                        local_patch_attn = last_attn.local_attn._last_patch_attn
                    if hasattr(last_attn, "global_attn") and hasattr(last_attn.global_attn, "_last_slice_weights"):
                        slice_weights = last_attn.global_attn._last_slice_weights

                self.attn_ctx = {
                    "pos":            pos[0].detach().cpu(),     # (N,3)
                    "perm":           perm[0].detach().cpu(),    # orig -> sorted
                    "inv_perm":       inv_perm[0].detach().cpu(),# sorted -> orig
                    "sections":       sections,                  # list[int]
                    "alpha":          alpha,                     # gate σ(g)
                    "local_patch_attn": local_patch_attn,        # list[(n,n)]
                    "slice_weights":    slice_weights,           # (1,H,N,G)

                }
            except Exception as e:

                self.attn_ctx = {"error": str(e)}

        # ------- 4. re-order -------
        fx_out = fx_sorted.gather(1, inv_perm[..., None].expand(-1, -1, fx_sorted.size(-1)))  # (1,N,C)

        return fx_out [0]








class PointVoxelPatcher(nn.Module):

    def __init__(self, patch_shape=(10, 10, 10)):   # 3×2×2=12
        super().__init__()
        self.patch_shape = torch.tensor(patch_shape, dtype=torch.long)

    def forward(self, pos: torch.Tensor):
        B, N, _ = pos.shape
        # min max
        p_min = pos.min(dim=1, keepdim=True)[0]
        p_max = pos.max(dim=1, keepdim=True)[0]
        unit = (p_max - p_min + 1e-6) / self.patch_shape.to(pos.device)  # 体素边长

        # point corordi (ix,iy,iz)
        idx = ((pos - p_min) / unit).floor()
        
        # 分clamp
        patch_shape_device = self.patch_shape.to(pos.device)
        for i in range(3):
            idx[..., i] = idx[..., i].clamp(0, patch_shape_device[i] - 1)
        
        idx = idx.long()
        ix, iy, iz = idx.unbind(-1) # belong to which patch

        nx, ny, nz = self.patch_shape.tolist() # patch number 
        patch_id = ix + nx * iy + nx * ny * iz        # (B,N) # patch ID
        return patch_id # , int(nx*ny*nz)



def patch_sort(patch_id):

    B, N = patch_id.shape
    perm, inv_perm, sections = [], [], []
    for b in range(B):
        order = torch.argsort(patch_id[b])             # (N,) # patchid 
        perm.append(order)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(N, device=order.device)
        inv_perm.append(inv) 
        uniq, counts = torch.unique(patch_id[b, order], return_counts=True)
        sections.append(counts.tolist())
    return torch.stack(perm), torch.stack(inv_perm), sections




class DualAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        self.dim = dim
        
        # local
        self.local_attn =StandardAttentionWithSections(dim, heads, dim_head, dropout)
        
        # global
        self.global_attn = Physics_Attention_Irregular_Mesh(
            dim, heads, dim_head, dropout, slice_num)
        

        
        # fusion type
        # self.fusion_type = "add"  # "add" or  "concat"
        self.fusion_type ='gate_norm_add'
        
        if self.fusion_type == "concat":
            self.fusion_proj = nn.Linear(dim * 2, dim)
        elif self.fusion_type == "add":
            self.local_weight = nn.Parameter(torch.ones(1))
            self.global_weight = nn.Parameter(torch.ones(1))
        elif self.fusion_type =='gate_norm_add':
            self.gate      = nn.Parameter(torch.zeros(1))
            self.ln_local  = nn.LayerNorm(dim)
            self.ln_global = nn.LayerNorm(dim)

        # >>> ADD:
        self.log_attn = False   
        self._last_alpha = None
        self._last_debug = None  # {"local_patch_attn": list[(n,n)], "slice_weights": (B,H,N,G), ...}

    

    def forward(self, x, patch_mask):

        local_out = self.local_attn(x,patch_mask)
        
        global_out = self.global_attn(x) 
        
        if self.fusion_type == "concat":
            fused = torch.cat([local_out, global_out], dim=-1)  # (B, N, 2*dim)
            output = self.fusion_proj(fused)  # (B, N, dim)
        elif self.fusion_type=='gate_norm_add':
            local_n  = self.ln_local(local_out)
            global_n = self.ln_global(global_out)
            alpha  = torch.sigmoid(self.gate)
            output   = alpha * local_n + (1 - alpha) * global_n
        else:  # add
            output = self.local_weight * local_out + self.global_weight * global_out

        # if self.log_attn:
        if getattr(self, 'log_attn', False): 
            self._last_alpha = float(alpha.item())
            self._last_debug = {
                "local_patch_attn": getattr(self.local_attn,  "_last_patch_attn", None),
                "slice_weights":    getattr(self.global_attn, "_last_slice_weights", None),
            }
            
        return output

    

class Patchsolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = DualAttention(
            hidden_dim, 
            heads=num_heads, 
            dim_head=hidden_dim // num_heads,
            dropout=dropout, 
            slice_num=slice_num
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, mask=None):
        fx = self.Attn(self.ln_1(fx),mask) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class StandardAttentionWithSections(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # >>> ADD:
        self._enable_log = False
        self._last_patch_attn = None  # list[Tensor(n,n)] for the last call

    @torch.no_grad()
    def _avg_heads(self, attn):  # attn: (1, H, n, n)
        return attn.mean(dim=1).squeeze(0).detach().cpu()  # (n, n)
    def forward(self, x, sections):
        B, N, C = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        batch_outputs = []
        
        for b in range(B):
            batch_q = q[b:b+1]  # 1 H N D
            batch_k = k[b:b+1]  # 1 H N D  
            batch_v = v[b:b+1]  # 1 H N D
            
            patch_outputs = []
            start = 0
            
            for n in sections[b]:  # 遍历每个patch

                patch_q = batch_q[:, :, start:start+n, :]  # 1 H n D
                patch_k = batch_k[:, :, start:start+n, :]  # 1 H n D
                patch_v = batch_v[:, :, start:start+n, :]  # 1 H n D
                
                dots = torch.matmul(patch_q, patch_k.transpose(-1, -2)) * self.scale
                attn = self.softmax(dots)
                attn = self.dropout(attn)
                
                patch_out = torch.matmul(attn, patch_v)  # 1 H n D
                patch_outputs.append(patch_out)
                start += n

                if getattr(self, '_enable_log', False) and b == 0:
                    if not hasattr(self, '_last_patch_attn'):
                        self._last_patch_attn = []
                    self._last_patch_attn.append(self._avg_heads(attn))
            
            batch_out = torch.cat(patch_outputs, dim=2)  # 1 H N D
            batch_outputs.append(batch_out)
        
        out = torch.cat(batch_outputs, dim=0)  # B H N D
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Standard_Attention_With_Mask(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # mask
        if mask is not None:
            # mask shape: (B, N, N) 或 (B, 1, N, N)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, N, N)

            dots = dots.masked_fill(mask, float('-inf'))
        
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    


class Physics_Attention_Irregular_Mesh_With_Mask(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.slice_num = slice_num

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, sections):
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[:, :, :, None] + 1e-5)

        out_slice_tokens = []
        for b in range(B):
            start = 0
            batch_out = []
            for n in sections[b]: 
                patch_slice = slice_token[b:b+1, :, start:start+n, :]  # B=1, H, n, D
                q = self.to_q(patch_slice)
                k = self.to_k(patch_slice)
                v = self.to_v(patch_slice)
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                attn = self.softmax(dots)
                attn = self.dropout(attn)
                out = torch.matmul(attn, v)  # B=1, H, n, D
                batch_out.append(out)
                start += n
            batch_out = torch.cat(batch_out, dim=2) 
            out_slice_tokens.append(batch_out)
        out_slice_token = torch.cat(out_slice_tokens, dim=0)  # B H N D

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)





def build_block_diag_mask(sections, device):

    N = sum(sections)
    mask = torch.ones(N, N, dtype=torch.bool, device=device)
    start = 0
    for n in sections:
        mask[start:start+n, start:start+n] = False  
        start += n
    return mask


