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


import os
import torch
import argparse
import yaml
import numpy as np
import time
from torch import nn
from torch_geometric.loader import DataLoader
from utils.tools_yj import extract_name_and_angle,get_save_name
from utils.drag_coefficient import cal_coefficient
from dataset.load_dataset import load_train_val_fold_file_windey
from dataset.dataset import GraphDataset, load_unstructured_grid_data
import scipy as sc
import vtk
from tqdm import tqdm
from vtk.util import numpy_support
from utils.plot import visual_y_height_yv,visual_y_height_yv_scatter,visual_y_z_yv,visual_y_z_yv_scatter,visual_3d_plane_velocity
from models.Transolver import Model
from models.Patch_solver import Model as Patch_Model


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--slice_num', default=32, type=int, help='number of slices')
parser.add_argument('--tesy_mode', default=1, type=int, help='1 normal 2 zeroshotdataset')
parser.add_argument('--test_task_name', default='normal')
parser.add_argument('--use_sparse', action='store_true', help='use sparse representation (default: False)')
parser.add_argument('--sparse_ratio',default=100,type=int,help='sparse points in the plane')
args = parser.parse_args()
print(args)


n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

train_data, val_data, coef_norm, vallst = load_train_val_fold_file_windey(args, preprocessed=True)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
print(f"test data: {val_data}")


all_names = []
for i, sublist in enumerate(vallst):
    name ,angle  = extract_name_and_angle(sublist)
    extracted_names=get_save_name(name ,angle,False)
    all_names.append(extracted_names) 

if args.use_sparse:
    path = f'use_sparse/metrics/{args.cfd_model}/{args.model_name}/{args.fold_id}/{args.nb_epochs}_{args.weight}/{args.slice_num}'
else:
    path = f'metrics/{args.cfd_model}/{args.model_name}/{args.fold_id}/{args.nb_epochs}_{args.weight}/{args.slice_num}'

model = torch.load(os.path.join(path, f'model_{args.model_name}_{args.nb_epochs}.pth'),weights_only=False).to(device)




test_loader = DataLoader(val_ds, batch_size=1)

if args.use_sparse:
    results_dir_path='./sparse_results/results/'
else:
    results_dir_path='./results/'

if not os.path.exists(results_dir_path + args.cfd_model + '/'):
    os.makedirs(results_dir_path + args.cfd_model + '/')

with torch.no_grad():
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')

    l2errs_velo, mses_velo_var, times = [], [], []
    print('Starting testing...')

    for index, (cfd_data, geom) in enumerate(test_loader):
        # ------------------------------------------------------------------
        # 1) read caseVTK
        # ------------------------------------------------------------------
        if args.tesy_mode==2:
            rootnew='/mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_3'
            case_dir = os.path.join(rootnew, vallst[index])  # e.g. /.../VTK
        else:
            case_dir = os.path.join(args.data_dir, vallst[index])  # e.g. /.../VTK
        vtk_files = [f for f in os.listdir(case_dir) if f.endswith('.vtk')]
        assert vtk_files, f"No VTK file found in {case_dir}"
        vtk_file = os.path.join(case_dir, vtk_files[0])
        vtk_data = load_unstructured_grid_data(vtk_file)  # vtk.vtkUnstructuredGrid

        # ------------------------------------------------------------------
        # 2) forward
        # ------------------------------------------------------------------
        cfd_data, geom = cfd_data.to(device), geom.to(device)
        tic = time.time()
        out = model((cfd_data, geom, None))
        toc = time.time()
        targets = cfd_data.y

        # ------------------------------------------------------------------
        # 3) denorm
        # ------------------------------------------------------------------
        if coef_norm is not None:
            mean = torch.tensor(coef_norm[2], device=device)
            std = torch.tensor(coef_norm[3], device=device)
            out_denorm = out * std + mean
            y_denorm = targets * std + mean
        else:
            out_denorm, y_denorm = out, targets


    
        ############## find speed =0  ##############
        # print(out_denorm.shape) #(N,3)
        zero_mask = torch.all(torch.isclose(y_denorm, torch.zeros_like(y_denorm), atol=1e-7), dim=-1)
        indices = torch.nonzero(zero_mask)
        # print("Indices where zero_mask is True:")
        # print(indices)
        zero_mask = zero_mask.unsqueeze(-1)
        out_denorm = out_denorm * (~zero_mask)
        # print(zero_mask)
        ############## find speed =0  ##############




        results_path = os.path.join(
            results_dir_path, args.cfd_model, args.model_name,
            str(args.fold_id), f"{args.nb_epochs}_{args.weight}", str(args.slice_num),args.test_task_name)
        os.makedirs(results_path, exist_ok=True)

        # Save raw numpy for offline analysis
        np.save(os.path.join(results_path, f"{all_names[index]}_pred.npy"),
                out_denorm.cpu().numpy())
        np.save(os.path.join(results_path, f"{all_names[index]}_gt.npy"),
                y_denorm.cpu().numpy())
        


        ################## section ########################
        # denorm x
        if coef_norm is not None:
            mean_x = torch.tensor(coef_norm[0], device=device)
            std_x = torch.tensor(coef_norm[1], device=device)
            out_x= cfd_data.x * std_x + mean_x

        # print(cfd_data.x.shape)
        # print(out_x.shape)
        # print(out_x[:,0].max())
        # print(out_x[:,0].min())
        plotted=False
        if plotted ==True:
            heightarr=[10,50,100,200,300,400,500]
            for h in tqdm(heightarr):
                img_path=os.path.join(results_path, f"Case_{all_names[index]}_")
                img_name=f"Case_{all_names[index]}"
                
                # y-v
                visual_y_height_yv(out_x[:,0:3].cpu().numpy(), y_denorm.squeeze(0).cpu().numpy(), out_denorm.cpu().numpy(), h,img_name, args, img_path)
                visual_y_height_yv_scatter(out_x[:,0:3].cpu().numpy(), y_denorm.squeeze(0).cpu().numpy(), out_denorm.squeeze(0).cpu().numpy(), h,img_name, args, img_path)
                visual_3d_plane_velocity(out_x[:,0:3].cpu().numpy(), y_denorm.squeeze(0).cpu().numpy(), out_denorm.squeeze(0).cpu().numpy(), h,img_name, args, img_path,'y')        
            ##################  section ########################



        # ------------------------------------------------------------------
        # 4) generate VTK arrays
        # ------------------------------------------------------------------
        pred_np = out_denorm.cpu().numpy().astype(np.float32).reshape(-1)
        gt_np   = y_denorm.cpu().numpy().astype(np.float32).reshape(-1)
        diff_np = abs((pred_np - gt_np)/(gt_np+1e-10))*100.0 

        def make_array(data_1d: np.ndarray, name: str) -> vtk.vtkFloatArray:
            arr = numpy_support.numpy_to_vtk(data_1d, deep=True, array_type=vtk.VTK_FLOAT)
            arr.SetNumberOfComponents(3)
            arr.SetName(name)
            return arr

        vtk_pred = make_array(pred_np, "U_pred")
        vtk_diff = make_array(diff_np, "U_diff")

        # ------------------------------------------------------------------
        # 5) 
        # ------------------------------------------------------------------


        writer = vtk.vtkUnstructuredGridWriter()

        # 5‑1) velocity
        pred_grid = vtk.vtkUnstructuredGrid(); pred_grid.DeepCopy(vtk_data)
        pred_grid.GetPointData().AddArray(vtk_pred)
        pred_grid.GetPointData().SetActiveVectors("U_pred")
        pred_grid.Modified()
        writer.SetFileName(os.path.join(results_path, f"{all_names[index]}_pred_velo_new.vtk"))
        writer.SetInputData(pred_grid)
        writer.Write()

        # 5‑2) error
        diff_grid = vtk.vtkUnstructuredGrid(); diff_grid.DeepCopy(vtk_data)
        diff_grid.GetPointData().AddArray(vtk_diff)
        diff_grid.GetPointData().SetActiveVectors("U_diff")
        diff_grid.Modified()
        writer.SetFileName(os.path.join(results_path, f"{all_names[index]}_pred_velo_diff_new.vtk"))
        writer.SetInputData(diff_grid)
        writer.Write()

        # 5‑3) real
        gt_grid = vtk.vtkUnstructuredGrid(); gt_grid.DeepCopy(vtk_data)
        gt_grid.Modified()
        writer.SetFileName(os.path.join(results_path, f"{all_names[index]}_gt_velo_new.vtk"))
        writer.SetInputData(gt_grid)
        writer.Write()

        # ------------------------------------------------------------------
        # 6) statics
        # ------------------------------------------------------------------
        l2err_velo = torch.norm(out_denorm - y_denorm) / torch.norm(y_denorm)
        mse_velo_var = criterion_func(out, targets).mean(dim=0)

        l2errs_velo.append(l2err_velo.cpu().item())
        mses_velo_var.append(mse_velo_var.cpu().numpy())
        times.append(toc - tic)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    avg_l2err_velo = float(np.mean(l2errs_velo))
    rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
    if coef_norm is not None:
        rmse_velo_var *= coef_norm[3]

    print("relative l2 error velo:", avg_l2err_velo)
    print("velo RMSE per component:", rmse_velo_var,
          "overall:", float(np.sqrt(np.mean(rmse_velo_var ** 2))))
    print("avg inference time [s]:", float(np.mean(times)))