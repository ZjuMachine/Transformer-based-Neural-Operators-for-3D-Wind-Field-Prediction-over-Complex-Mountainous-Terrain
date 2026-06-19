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

import train_windey
import os
import torch
import argparse

from dataset.load_dataset import load_train_val_fold_windey
from dataset.dataset import GraphDataset
from models.Transolver import Model
from models.Patch_solver import Model as Patch_Model



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--slice_num', default=32, type=int, help='number of slices')
parser.add_argument('--use_sparse', action='store_true', help='use sparse representation (default: False)')
parser.add_argument('--sparse_ratio',default=100,type=int,help='sparse points in the plane')
parser.add_argument('--tesy_mode', default=1, type=int, help='1 normal 2 zeroshotdataset')

args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs, 'model_name': args.model_name}

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
print('Current device:',device)

train_data, val_data, coef_norm = load_train_val_fold_windey(args, preprocessed=args.preprocessed)
# create  dataloader
train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

if args.use_sparse:
    channle_size=7+3
else:
    channle_size=7

if args.cfd_model == 'Transolver':
    model = Model(n_hidden=256, n_layers=8, space_dim=channle_size,
                  fun_dim=0,
                  n_head=8,
                  mlp_ratio=2, out_dim=3,
                  slice_num=int(args.slice_num),
                  unified_pos=0).cuda()  
    print(model)
elif args.cfd_model == 'Patch_solver':
    model = Patch_Model(n_hidden=164, n_layers=4, space_dim=channle_size,
                  fun_dim=0,
                  n_head=8,
                  mlp_ratio=2, out_dim=3,
                  slice_num=int(args.slice_num),
                  unified_pos=0).cuda()  
    print(model)

if args.use_sparse:
    path = f'use_sparse/metrics/{args.cfd_model}/{args.model_name}/{args.fold_id}/{args.nb_epochs}_{args.weight}/{args.slice_num}'
else:
    path = f'metrics/{args.cfd_model}/{args.model_name}/{args.fold_id}/{args.nb_epochs}_{args.weight}/{args.slice_num}'
if not os.path.exists(path):
    os.makedirs(path)


model = train_windey.main(device, train_ds, val_ds, model, hparams, path, val_iter=args.val_iter, reg=args.weight,
                   coef_norm=coef_norm)
