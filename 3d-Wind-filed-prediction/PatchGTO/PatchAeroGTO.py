import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from einops import rearrange

class MLP(nn.Module): 
    def __init__(self, 
                input_size=128,  #in
                output_size=128,  # out
                layer_norm=True, 
                n_hidden=2, 
                hidden_size=256, 
                act = 'PReLU',
                ):
        super(MLP, self).__init__()
        if act == 'GELU':
            self.act = nn.GELU()
        elif act == 'SiLU':
            self.act = nn.SiLU()
        elif act == 'PReLU':
            self.act = nn.PReLU()
            
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))  # norm

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x) 
    
class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm, act = 'GELU', output_size=edge_size)
        
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm, act = 'GELU', output_size=output_size)

    def forward(self, V, E, edges):
        # V[2,100000,128] E[2,300000,128], edges[2,300000,128]
        edges = edges.long()
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        # print(senders.shape, receivers.shape) #torch.Size([2, 300000, 128]) torch.Size([2, 300000, 128])
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt) #[2, 300000, 128]

        col = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        # print(col.shape) # torch.Size([2, 300000, 128])
        edge_sum = scatter_mean(edge_embeddings, col, dim=-2, dim_size=V.shape[1])
        # print(edge_sum.shape) #torch.Size([2, 100000, 128])
        node_inpt = torch.cat([V, edge_sum], dim=-1)

        node_embeddings = self.f_node(node_inpt)
        # print(node_embeddings.shape)

        return node_embeddings, edge_embeddings #[2, 100000, 128]  [2, 300000, 128]
    
class Encoder(nn.Module): 
    def __init__(self, 
                 state_embedding_dim = 128,
                 ):
        super(Encoder, self).__init__()

        self.state_embedding_dim = state_embedding_dim # 128
        self.enc_s_dim = 128 # 128
        
        self.enc_s = MLP(input_size = 114, output_size = self.enc_s_dim, act = 'SiLU',layer_norm = False) # in=114 out=128

        self.fv = MLP(input_size = self.enc_s_dim, output_size=state_embedding_dim, layer_norm = False) # in=128 out=128

    def FourierEmbedding(self, pos, pos_start, pos_length): # pos [2,100000,6] pos_start=-4, pos_length=9
        # F(x) = [cos(2^i * pi * x), sin(2^i * pi * x)], i = -4, -3, -2, -1, 0, 1, 2, 3, 4
        original_shape = pos.shape
        new_pos = pos.reshape(-1, original_shape[-1]) # [200000, 6]

        index = torch.arange(pos_start, pos_start + pos_length, device=pos.device)
        index = index.float() #[-4, 4]
        freq = 2 ** index * torch.pi # [2^(-4) * pi, 2^(-3) * pi, ..., 2^(4) * pi]

        cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1)) # [1, 1, 9] * [200000, 6] -> [200000, 6, 9]
        sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1)) # [1, 1, 9] * [200000, 6] -> [200000, 6, 9]
        embedding = torch.cat([cos_feat, sin_feat], dim=-1) # [200000, 6, 18]
        embedding = embedding.view(*original_shape[:-1], -1) # [2, 100000, 6 * 18]
        all_embeddings = torch.cat([embedding, pos], dim=-1) # [2, 100000, 108+6] = [2, 100000, 114] [B, N, D*(2L+1)]
        
        return all_embeddings

    def forward(self, node_pos):
        #  1.
        pos_enc = self.FourierEmbedding(node_pos, -4, 9) # 2 * 9 * 2 + 2 = 38 pos_enc.shape torch.Size([2, 100000, 114])
        # print('pos_enc.shape', pos_enc.shape)
        s_enc = self.enc_s(pos_enc) # [2, 100000, 128]
        # print('s_enc.shape', s_enc.shape)
        # 3.         
        V_in = torch.cat([s_enc], dim=-1)
        V_in = self.fv(V_in) # (2， 100000， 128）  
        # print('V_in.shape', V_in.shape)
        return V_in, s_enc
    
class AttentionBlock(nn.Module): 
    def __init__(self, 
                n_token = 64,
                w_size = 128, 
                n_heads = 4,
                ): # 64/128/4
        super(AttentionBlock, self).__init__()
        
        self.channel_dim = w_size
        self.n_token = n_token
        
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.channel_dim ** -0.5
        self.n_heads = n_heads
        
        # 1. 
        self.Q = nn.Parameter(torch.randn(self.n_token, self.channel_dim), requires_grad=True)
        
        self.to_q_1 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_k_1 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_v_1 = nn.Linear(self.channel_dim, self.channel_dim)
        
        # 2. 
        self.attention2 = nn.MultiheadAttention(embed_dim=w_size, num_heads=n_heads, batch_first=True)
        
        # 3.
        self.to_q_2 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_k_2 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_v_2 = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, W_0):
        
        # 1. transform decoder
        B = W_0.shape[0]
        learned_Q = self.Q.unsqueeze(0).expand(B, -1, -1)
        
        Q_1 = self.to_q_1(learned_Q)
        K_1 = self.to_k_1(W_0)
        V_1 = self.to_v_1(W_0)

        attn1 = self.softmax(torch.einsum('bmc, bnc -> bmn', Q_1, K_1) * self.scale)
        W_1 = torch.matmul(attn1, V_1)  
    
        # 2. self-attention
        W_2, _ = self.attention2(W_1, W_1, W_1)
        
        # 3. transform decoder
        Q_2 = self.to_q_2(W_0)
        K_2 = self.to_k_2(W_2)
        V_2 = self.to_v_2(W_2)
                
        attn2 = self.softmax(torch.einsum('bnc, bmc -> bnm', Q_2, K_2) * self.scale)
        W_3 = torch.matmul(attn2, V_2)  

        return W_3  
    
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
            
            for n in sections[b]: 
         
                patch_q = batch_q[:, :, start:start+n, :]  # 1 H n D
                patch_k = batch_k[:, :, start:start+n, :]  # 1 H n D
                patch_v = batch_v[:, :, start:start+n, :]  # 1 H n D
                
          
                dots = torch.matmul(patch_q, patch_k.transpose(-1, -2)) * self.scale
                attn = self.softmax(dots)
                attn = self.dropout(attn)
                
                patch_out = torch.matmul(attn, patch_v)  # 1 H n D
                patch_outputs.append(patch_out)
                start += n
            

            batch_out = torch.cat(patch_outputs, dim=2)  # 1 H N D
            batch_outputs.append(batch_out)
        
  
        out = torch.cat(batch_outputs, dim=0)  # B H N D
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class PointVoxelPatcher(nn.Module):
    def __init__(self, patch_shape=(10, 10, 10)):   # 3×2×2=12
        super().__init__()
        self.patch_shape = torch.tensor(patch_shape, dtype=torch.long)

    def forward(self, pos: torch.Tensor):
        B, N, _ = pos.shape

        p_min = pos.min(dim=1, keepdim=True)[0]
        p_max = pos.max(dim=1, keepdim=True)[0]
        unit = (p_max - p_min + 1e-6) / self.patch_shape.to(pos.device) 


        idx = ((pos - p_min) / unit).floor()
        

        patch_shape_device = self.patch_shape.to(pos.device)
        for i in range(3):
            idx[..., i] = idx[..., i].clamp(0, patch_shape_device[i] - 1)
        
        idx = idx.long()
        ix, iy, iz = idx.unbind(-1) 

        nx, ny, nz = self.patch_shape.tolist() 
        patch_id = ix + nx * iy + nx * ny * iz        
        return patch_id # , int(nx*ny*nz)
    

def patch_sort(patch_id):

    B, N = patch_id.shape
    perm, inv_perm, sections = [], [], []
    for b in range(B):
        order = torch.argsort(patch_id[b])           
        perm.append(order) 
        inv = torch.empty_like(order)
        inv[order] = torch.arange(N, device=order.device)
        inv_perm.append(inv) 
       
        uniq, counts = torch.unique(patch_id[b, order], return_counts=True)
        sections.append(counts.tolist())
    return torch.stack(perm), torch.stack(inv_perm), sections

class MixerBlock(nn.Module):
    def __init__(self, state_embedding_dim, att_embedding_dim, n_head, n_token, enc_s_dim=0, idx = 0, n_blocks = 4):
        super(MixerBlock, self).__init__()
        self.enc_s_dim = enc_s_dim
        node_size = state_embedding_dim if enc_s_dim == 0 else state_embedding_dim + enc_s_dim
        self.gnn = GNN(node_size=node_size, edge_size=state_embedding_dim, output_size=state_embedding_dim, layer_norm=True)

        self.ln1 = nn.LayerNorm(att_embedding_dim)
        self.ln2 = nn.LayerNorm(att_embedding_dim)
        self.linear = nn.Linear(att_embedding_dim, att_embedding_dim)
        self.MHA = AttentionBlock(n_token=n_token, w_size=att_embedding_dim, n_heads=n_head) # 64/128/4

        self.idx = idx
        self.n_blocks = n_blocks
        self.act = nn.SiLU()

        # Patch attention
        self.patch_attn = StandardAttentionWithSections(dim=att_embedding_dim, heads=n_head, dim_head=att_embedding_dim // n_head)
        self.patch_size = 1000

        self.gate = nn.Parameter(torch.zeros(1))
        self.ln_local = nn.LayerNorm(att_embedding_dim)
        self.ln_global = nn.LayerNorm(att_embedding_dim)
            

    def forward(self, V, E, edges, s_enc, node_pos):
        # V [2,100_000,128]， E [2,300000,128]
        if self.enc_s_dim > 0:
            V_in = torch.cat([V, s_enc], dim=-1)
        else:
            V_in = V
        # print(V_in.shape, E.shape, edges.shape) #torch.Size([2, 100000, 128]) torch.Size([2, 300000, 128]) torch.Size([2, 300000, 2])
        v, e = self.gnn(V_in, E, edges) # v[2, 100000, 128]  e[2, 300000, 128]

        V = V + v
        
        if self.idx < self.n_blocks:    
            E = E + e
        else:
            E = E + e

        V_ln = self.ln1(V)

        
        patch_id = PointVoxelPatcher(patch_shape=(V.shape[1] // self.patch_size, 1, 1))(node_pos[..., :3]) 
        perm, inv_perm, sections = patch_sort(patch_id)
        V_sorted = V_ln.gather(1, perm[..., None].expand(-1, -1, V_ln.size(-1)))  # (B,N,C)
        V_patch_attn = self.patch_attn(V_sorted, sections)
        V_patch_attn = V_patch_attn.gather(1, inv_perm[..., None].expand(-1, -1, V_patch_attn.size(-1))) 

        # print(V_patch_attn.shape) [2, 100000, 128]
     
        V_global_attn = self.MHA(V_ln)
        # print(V_global_attn.shape) [2, 100000, 128]

        local_n = self.ln_local(V_patch_attn)
        global_n = self.ln_global(V_global_attn)
        alpha = torch.sigmoid(self.gate)
        W_1 = alpha * local_n + (1 - alpha) * global_n


        
        W_2 = V + W_1
        
        W_3 = W_2 + self.linear(self.ln2(W_2))

        return W_3, E # [2,100000,128] [2,300000,128]

class Mixer(nn.Module):
    def __init__(self, N, state_embedding_dim, att_embedding_dim, n_head, n_token):
        super(Mixer, self).__init__()

        self.fe = MLP(input_size=3, output_size=state_embedding_dim, n_hidden=1, act = 'SiLU', layer_norm=False)
        
        self.blocks = nn.ModuleList([
            MixerBlock(state_embedding_dim=state_embedding_dim, att_embedding_dim=att_embedding_dim, 
                       n_head=n_head, n_token=n_token, enc_s_dim=0 if i > 0 else 128, idx = int(i), n_blocks = N-1)
            for i in range(N)
        ])


    def forward(self, V, edges, node_pos, s_enc):
        edges = edges.long()
        # print(node_pos.shape, edges.shape) # torch.Size([2, 100000, 6]) torch.Size([2, 300000, 2])
        # print( "edegs",(edges[..., 0].unsqueeze(-1).repeat(1, 1, 2)).shape)
        senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2)) # [2,300000,2]->[2,300000]->[2,300000,1]->[2,300000,1*2] dim=-2 表示在倒数第二个维度（即 100000 这个节点维度）上做索引。(错误)
        receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2)) #  [2, 300000, 2, 6]
        # print("receivers",receivers.shape) # torch.Size([2, 300000, 2])
        # print(edges[0][1])
        # print(receivers[0][1])
        distance = receivers - senders # [2, 300000, 2]
        # print("distance.shape", distance.shape) #[2,300000, 2]
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True)) # [2, 300000, 1]
        # print("norm", norm.shape)
        E = torch.cat([distance, norm], dim=-1) #【2,300000，2+1】
        # print("E", E.shape)
        E = self.fe(E)
        # print(E.shape) [2,300000,128]
        # print(edges[0][0], E[0][0].shape)

        for block in self.blocks:
            V, E = block(V, E, edges, s_enc, node_pos) # [2,100000,128] [2,300000,128]

        return V  # [2,100000,128]
    
class Decoder(nn.Module):
    def __init__(self,
                 state_embedding_dim = 128, 
                 state_size = 4,
                 ):
        super(Decoder, self).__init__()
        
        self.final_mlp_node = nn.Sequential(
            nn.Linear(state_embedding_dim, state_embedding_dim), nn.PReLU(),
            nn.Linear(state_embedding_dim, state_embedding_dim), nn.PReLU(),
            nn.Linear(state_embedding_dim, state_size)
        )
        
    def forward(self, V):
        
        V_in = torch.cat([V],dim=-1)
        final_state_node = self.final_mlp_node(V_in) #
        
        return final_state_node  #torch.Size([2, 100000, 3])
 
class time_stepping(nn.Module):
    def __init__(self, 
                N_block =4,
                state_size = 3, 
                state_embedding_dim = 128, 
                att_embedding_dim = 128,
                n_head = 4,
                n_token = 64,
                ):


        super(time_stepping, self).__init__()
    
        self.encoder = Encoder(
            state_embedding_dim = state_embedding_dim # 128
            )

        self.mixer = Mixer(
            N= N_block,  # 4
            state_embedding_dim = state_embedding_dim, # 128
            att_embedding_dim = att_embedding_dim, #128
            n_head = n_head, # 4
            n_token = n_token # 64
        )
        
        self.decoder = Decoder(
            state_embedding_dim = state_embedding_dim, #128
            state_size = state_size # 3
            )
        
    def forward(self, node_pos, edges):
        
        # 1. Encoder + time_aggregator
        V, s_enc = self.encoder(node_pos) 
        
        # 2. mixer(processer)
        V = self.mixer(V, edges, node_pos, s_enc) # V [2,100000,128]
        
        # 3. decoder 
        final_state_node = self.decoder(V) # torch.Size([2, 100000, 3])
        
        return final_state_node  #torch.Size([2, 100000, 3])

class AeroGTO(nn.Module):
    def __init__(self, 
                N_block = 4, 
                state_size = 3,  
                state_embedding_dim = 128, 
                att_embedding_dim = 128,
                n_head = 4,
                n_token = 64,
                ):
        super(AeroGTO, self).__init__()
        
        self.time_stepping = time_stepping(
            N_block = N_block, # 4
            state_size = state_size, # 3
            state_embedding_dim = state_embedding_dim, # 128
            att_embedding_dim = att_embedding_dim, # 128
            n_head = n_head, # 4
            n_token = n_token, # 64
        )
        

    def forward(self, node_pos, edges):# (2, 100000, 6) (2, 300000, 2)
        
        f_t = self.time_stepping(node_pos, edges) # (2, 100000, 6) (2, 300000, 2)
        next_state = f_t

        return next_state

if __name__ == "__main__":
    pass