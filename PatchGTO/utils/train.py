import torch
from tqdm import tqdm
from torch.nn.functional import one_hot
import random
import torch.nn as nn
import numpy as np
import os

import h5py

from torch.cuda.amp import autocast

import torch
import torch.nn.functional as F


import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from utils.loss import TestLoss
from utils.shapenet_velocity_all import Car_Dataset_v



# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.reshape((num_examples, -1)) - y.reshape((num_examples, -1)), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        diff_norms = torch.norm(x-y, 2)
        y_norms = torch.norm(y, self.p)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def get_val_loss(output_v_hat, v, if_rescale, info):
    device = output_v_hat.device
    v_mean = torch.tensor(info['v_mean']).to(device)
    v_std = torch.tensor(info['v_std']).to(device)
    
    #################################
    loss_fn = TestLoss(d = len(v_std))
    v_target = v.to(device)
    
    losses = {}
    criterion = nn.MSELoss()
    
    losses['L2_v_norm'] = loss_fn(output_v_hat, (v_target - v_mean) / v_std).item()
    losses["MSE_loss_norm"] = criterion(output_v_hat, (v_target - v_mean) / v_std)
    ################
    if if_rescale:
        v_hat = output_v_hat * v_std + v_mean
    else:
        v_hat = output_v_hat

    zero_mask = torch.all(torch.isclose(v_target, torch.zeros_like(v_target), atol=1e-7), dim=-1)  # shape: (batch_size, N)
    zero_mask = zero_mask.unsqueeze(-1)  # shape: (batch_size, N, 1)
    v_hat = v_hat * (~zero_mask)  
        
    losses['L2_v'] = loss_fn(v_hat, v_target).item()
    losses["MSE_loss"] = criterion(v_hat, v_target).item()

    # R2loss
    ss_res = torch.sum((v_hat - v_target) ** 2)
    ss_tot = torch.sum((v_target - torch.mean(v_target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    losses["R2_v"] = r2.item()
    
    
    # losses['L2_v'] = get_l2_loss(v_hat, v_target).item()
    
    return losses


def get_train_loss(output_v_hat, v, loss_flag, if_rescale, info):
    device = output_v_hat.device
    v_mean = torch.tensor(info['v_mean']).to(device)
    v_std = torch.tensor(info['v_std']).to(device)

    #################################
    loss_fn = TestLoss(d = len(v_std))
    v_target = v.to(device)
    
    losses = {}
    criterion = nn.MSELoss()
        
    if loss_flag == 'L2_loss_norm':
        losses['loss'] = loss_fn(output_v_hat, (v_target - v_mean) / v_std)
    elif loss_flag == 'MSE_loss_norm':
        losses['loss'] = criterion(output_v_hat, (v_target - v_mean) / v_std)

    # 有效
    losses['L2_v_norm'] = loss_fn(output_v_hat, (v_target - v_mean) / v_std).item()
    losses["MSE_loss_norm"] = criterion(output_v_hat, (v_target - v_mean) / v_std).item()
    
    if if_rescale:
        v_hat = output_v_hat * v_std + v_mean
    else:
        v_hat = output_v_hat

    zero_mask = torch.all(torch.isclose(v_target, torch.zeros_like(v_target), atol=1e-7), dim=-1)  # shape: (batch_size, N)
    zero_mask = zero_mask.unsqueeze(-1)  # shape: (batch_size, N, 1)
    v_hat = v_hat * (~zero_mask)  
    
    if loss_flag == 'L2_loss':
        # losses['loss'] = 10*loss_fn(v_hat, v_target)+smooth_loss/10000
        losses['loss'] = 10*loss_fn(v_hat, v_target)
    elif loss_flag == 'MSE_loss':
        # losses['loss'] = 10*criterion(v_hat, v_target)+smooth_loss/10000
        losses['loss'] = 10*criterion(v_hat, v_target)
        

    # 有效
    losses['L2_v'] = loss_fn(v_hat, v_target).item()
    losses["MSE_loss"] = criterion(v_hat, v_target).item()
    

    return losses


def train(args, model, train_dataloader, optim, device):
        
    model.train()    
    loss = 0
    L2_v = 0
    L2_v_norm = 0
    MSE_loss = 0
    MSE_loss_norm = 0
    
    num = 0
    for i, [input, t] in enumerate(tqdm(train_dataloader, desc="Training")):
        # forward
        optim.zero_grad()
        
        v = input['velocity']        
        node_pos = input['node_pos']
        edges = input['edges']
        
 
        output_v_hat = model(node_pos.to(device), edges.to(device)) 
        
        costs = get_train_loss(
            output_v_hat,
            v, 
            args.train["loss_flag"], 
            args.train["if_rescale"], 
            args.train["info"],
            )
        
        costs['loss'].backward()
        optim.step()
        
        loss = loss + costs['loss'].item()
        L2_v = L2_v + costs['L2_v']
        L2_v_norm = L2_v_norm + costs['L2_v_norm']
        MSE_loss =  MSE_loss + costs['MSE_loss']
        MSE_loss_norm =  MSE_loss_norm + costs['MSE_loss_norm']
        #########################################
        num = num + 1
        
        # break
        
    batch_error = {}
    batch_error['loss'] = loss / num
    batch_error['L2_v'] = L2_v / num
    batch_error['L2_v_norm'] = L2_v_norm / num
    batch_error['MSE_loss'] = MSE_loss / num
    batch_error['MSE_loss_norm'] = MSE_loss_norm / num
    
    return batch_error

def validate(args, model, val_dataloader, device):
    model.eval()
    
    L2_v = 0
    L2_v_norm = 0
    MSE_loss = 0
    MSE_loss_norm = 0
    
    num = 0
    with torch.no_grad():
          
        for i, [input, t] in enumerate(tqdm(val_dataloader, desc="Validation")):
        # for i, [input,_] in enumerate(val_dataloader):    
            
            v = input['velocity']        
            node_pos = input['node_pos']
            edges = input['edges']

            output_v_hat = model(node_pos.to(device), edges.to(device)) 
                                
            costs = get_val_loss(
                output_v_hat,
                v, 
                args.train["if_rescale"], 
                args.train["info"]
                )

            L2_v = L2_v + costs['L2_v']
            L2_v_norm = L2_v_norm + costs['L2_v_norm']
            MSE_loss =  MSE_loss + costs['MSE_loss']
            MSE_loss_norm =  MSE_loss_norm + costs['MSE_loss_norm']
            #########################################
            num = num + 1

    batch_error = {}
    
    batch_error['L2_v'] = L2_v / num
    batch_error['L2_v_norm'] = L2_v_norm / num
    batch_error['MSE_loss'] = MSE_loss / num
    batch_error['MSE_loss_norm'] = MSE_loss_norm / num
    
    return batch_error


def infer(args, model, test_dataloader, device):
    v_mean = torch.tensor(args.train["info"]['v_mean']).to(device)
    v_std = torch.tensor(args.train["info"]['v_std']).to(device)
    
    model.to(device)
    model.eval()

    L2_v = 0
    L2_v_norm = 0
    MSE_loss = 0
    MSE_loss_norm = 0
    R2_v = 0
    
    num = 0
    with torch.no_grad():
        for i, [input, name] in enumerate(tqdm(test_dataloader, desc="testing")):
            node_pos = input['node_pos']
            edges = input['edges']
            v = input['velocity']
            # print("node_pos.shape:",node_pos.shape) # node_pos.shape: torch.Size([1, 392073, 6])
            # print("v.shape:",v.shape) # v.shape: torch.Size([1, 392073, 3])

            # batch_size = args.dataset["test"]["N_target"]//2 
            batch_size = args.dataset["test"]["N_target"]
            # v_hat_all = infer_large_sample(model, node_pos, edges, batch_size, device).to(device)
            v_hat_all = infer_large_sample3(model, node_pos, edges, batch_size, device).to(device)
            # v_hat_all = model(node_pos.to(device), edges.to(device))
            print("v_hat_all.shape:",v_hat_all.shape, "v.shape:",v.shape) # v_hat_all.shape: torch.Size([1, 392073, 3]) v.shape: torch.Size([1, 392073, 3])

            costs = get_val_loss(
                v_hat_all,
                v, 
                args.train["if_rescale"], 
                args.train["info"]
                )

            L2_v = L2_v + costs['L2_v']
            L2_v_norm = L2_v_norm + costs['L2_v_norm']
            MSE_loss =  MSE_loss + costs['MSE_loss']
            MSE_loss_norm =  MSE_loss_norm + costs['MSE_loss_norm']
            R2_v = R2_v + costs['R2_v']
            ##################
            num = num + 1
            ################
            if args.train["if_rescale"]:
                v_hat = v_hat_all[...,:3] * v_std + v_mean
            else:
                v_hat = v_hat_all
            
            v = v.to(device)
            zero_mask = torch.all(torch.isclose(v, torch.zeros_like(v), atol=1e-7), dim=-1)  # shape: (batch_size, N)
            zero_mask = zero_mask.unsqueeze(-1)  # shape: (batch_size, N, 1)
            v_hat = v_hat * (~zero_mask) 

            dataset = Car_Dataset_v(
                data_path = args.dataset["data_path"],
                mode = "test"
            )
            noscale_node_pos = dataset.unscale_pos(node_pos)


        
            filename = name[0].split('/')[-3] + '-' + name[0].split('/')[-1]
            # filename = name[0]
            output_dir = "./result/output_vtk"
            os.makedirs(output_dir, exist_ok=True) 

            print(f"the test is {filename}")

            
            print(costs)
            
            xyz = noscale_node_pos[..., :3].to(device)  # shape: [1, 305738, 3]

            output = torch.cat([xyz, v, v_hat], dim=-1)  # shape: [1, 305738, 9]


            output = output.squeeze(0)  # shape: [305738, 9]

            output_np = output.cpu().numpy()

            np.save(f"{output_dir}/{filename}.npy", output_np)

            print("output_combined.npy，shape:", output_np.shape)
    
    batch_error = {}
    
    batch_error['L2_v'] = L2_v / num
    batch_error['L2_v_norm'] = L2_v_norm / num
    batch_error['MSE_loss'] = MSE_loss / num
    batch_error['MSE_loss_norm'] = MSE_loss_norm / num
    batch_error['R2_v'] = R2_v / num
    print(batch_error)
    print("Finished!")

def infer_large_sample3(model, node_pos, edges, batch_size, device, extra_rounds=2):
    N = node_pos.shape[1]
    v_hat_sum = torch.zeros((1, N, 3), device=device)
    vote_count = torch.zeros((1, N, 1), device=device)

    node_pos = node_pos.to(device)
    edges = edges.to(device)

    full_indices = torch.randperm(N, device=device)
    groups = torch.split(full_indices, batch_size)

    for indices in groups:
        indices_sorted, _ = torch.sort(indices)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[indices_sorted] = True

        edges = edges.squeeze(0)  # [E, 2]
        mask_edges = mask[edges[:, 0]] & mask[edges[:, 1]]
        sampled_edges = edges[mask_edges]

        idx_map = {old.item(): new for new, old in enumerate(indices_sorted)}
        remapped_edges = torch.stack([
            torch.tensor([idx_map[i.item()] for i in sampled_edges[:, 0]], device=device),
            torch.tensor([idx_map[i.item()] for i in sampled_edges[:, 1]], device=device)
        ], dim=1)
        edges = edges.unsqueeze(0)  # [E, 2]

        sampled_node_pos = node_pos[:, indices_sorted, :]

        with torch.no_grad():
            v_hat_chunk = model(sampled_node_pos, remapped_edges)

        v_hat_sum[:, indices_sorted, :] += v_hat_chunk
        vote_count[:, indices_sorted, :] += 1

    for _ in range(extra_rounds):
        indices = torch.randperm(N, device=device)[:batch_size]
        indices_sorted, _ = torch.sort(indices)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[indices_sorted] = True

        edges = edges.squeeze(0)  # [E, 2]
        mask_edges = mask[edges[:, 0]] & mask[edges[:, 1]]
        sampled_edges = edges[mask_edges]

        idx_map = {old.item(): new for new, old in enumerate(indices_sorted)}
        remapped_edges = torch.stack([
            torch.tensor([idx_map[i.item()] for i in sampled_edges[:, 0]], device=device),
            torch.tensor([idx_map[i.item()] for i in sampled_edges[:, 1]], device=device)
        ], dim=1)
        edges = edges.unsqueeze(0)  # [E, 2]

        sampled_node_pos = node_pos[:, indices_sorted, :]

        with torch.no_grad():
            v_hat_chunk = model(sampled_node_pos, remapped_edges)

        v_hat_sum[:, indices_sorted, :] += v_hat_chunk
        vote_count[:, indices_sorted, :] += 1


    vote_count = torch.clamp(vote_count, min=1.0)
    v_hat_avg = v_hat_sum / vote_count

    return v_hat_avg


def filter_edges(edges, start, end):

    edges = edges.squeeze(0) 
    mask = (edges[:, 0] >= start) & (edges[:, 0] < end) & \
           (edges[:, 1] >= start) & (edges[:, 1] < end)
    filtered = edges[mask]

    filtered -= start
    return filtered.unsqueeze(0)