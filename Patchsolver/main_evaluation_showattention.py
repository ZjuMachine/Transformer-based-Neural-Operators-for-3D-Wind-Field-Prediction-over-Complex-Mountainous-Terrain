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
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr, spearmanr
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# >>>
from models.Patch_solver import patch_sort  # 
import matplotlib
matplotlib.use('Agg')  #
import matplotlib.pyplot as plt
from einops import rearrange
import warnings

from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d as bs2d
from scipy.interpolate import griddata


import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
import warnings

import xdem



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



parser.add_argument('--attn_vis', action='store_true',
                    help='visualize local/global attention around anchors')
parser.add_argument('--attn_heights', default='10,150,300', type=str,
                    help='anchor heights in meters, comma-separated')
parser.add_argument('--attn_per_height', default=1, type=int,
                    help='anchors per height plane')
parser.add_argument('--attn_tol', default=5.0, type=float,
                    help='height tolerance when picking anchors (m)')
parser.add_argument('--attn_topk_tokens', default=8, type=int,
                    help='top-k slice tokens to display')


args = parser.parse_args()
print(args)




def plot_roughness_vs_attention(
    ctx,
    pos3,
    perm,
    inv_perm,
    case_dir,
    save_path,
    height_slice=None,
    grid_size=(500, 500),
    roughness_type='tri'  # 'tri', 'roughness', 'slope', 'curvature', 'tpi'
):
 
    from scipy.stats import pearsonr, spearmanr
    from scipy.interpolate import RegularGridInterpolator
    from rasterio.transform import from_origin
    import xdem
    
    
    bottom_dir = os.path.join(case_dir, 'bottom')
    if not os.path.exists(bottom_dir):
        print(f"[WARN] No bottom directory in {case_dir}")
        return
    
    bottom_files = [f for f in os.listdir(bottom_dir) if f.endswith('.vtk')]
    if not bottom_files:
        print(f"[WARN] No VTK file in {bottom_dir}")
        return
    
    bottom_vtk = os.path.join(bottom_dir, bottom_files[0])
    bottom_data = load_poly_data(bottom_vtk)
    terrain_points = vtk_to_numpy(bottom_data.GetPoints().GetData())
    # print(terrain_points)
    

    dem_array, (x_grid, y_grid) = points_to_dem(terrain_points, grid_size=grid_size)
    resolution = (x_grid[1] - x_grid[0] + y_grid[1] - y_grid[0]) / 2
    

    dem_array_flipped = np.flipud(dem_array)
    
    transform = from_origin(
        x_grid[0],     
        y_grid[-1],    
        resolution, resolution
    )
    dem = xdem.DEM.from_array(
        dem_array_flipped,
        transform=transform,
        crs="EPSG:32633",
        nodata=-9999
    )
    

    attributes = compute_base_attributes(dem)
    

    if roughness_type not in attributes:
        print(f"[WARN] {roughness_type} not in attributes, using 'tri'")
        roughness_type = 'tri'
    
    roughness_raster = attributes[roughness_type]
    

    if hasattr(roughness_raster, 'data'):
        roughness_grid = np.asarray(roughness_raster.data)
    else:
        roughness_grid = np.asarray(roughness_raster)
    

    roughness_grid = np.flipud(roughness_grid)
    

    from scipy.ndimage import generic_filter
    def fill_nan(arr):
        mask = np.isnan(arr)
        if not mask.any():
            return arr
        arr_filled = arr.copy()

        from scipy.ndimage import distance_transform_edt
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        arr_filled = arr[tuple(ind)]
        return arr_filled
    
    roughness_grid_filled = fill_nan(roughness_grid)

    interp = RegularGridInterpolator(
        (y_grid, x_grid),  
        roughness_grid_filled,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    

    query_points = np.column_stack([pos3[:, 1], pos3[:, 0]])  # (y, x) 顺序
    roughness_at_nodes = interp(query_points)
    

    sections = list(ctx["sections"])
    bounds = np.cumsum([0] + sections)
    N = len(perm)
    eps = 1e-12
    
    # H_global
    sw_sorted = ctx["slice_weights"].mean(1)[0].numpy()
    H_global_sorted = -(sw_sorted * np.log(sw_sorted + eps)).sum(axis=1)
    H_global = H_global_sorted[inv_perm]
    
    # H_local
    H_local_sorted = np.zeros(N, dtype=np.float32)
    for pid, A in enumerate(ctx["local_patch_attn"]):
        A_np = np.clip(A.numpy(), 0.0, 1.0)
        row_sum = A_np.sum(axis=1, keepdims=True) + eps
        P = A_np / row_sum
        Hrow = -(P * np.log(P + eps)).sum(axis=1)
        st, ed = bounds[pid], bounds[pid + 1]
        H_local_sorted[st:ed] = Hrow
    H_local = H_local_sorted[inv_perm]
    

    n_sorted = np.concatenate([np.full(s, s, dtype=np.int32) for s in sections])
    n_per_node = n_sorted[inv_perm]
    H_local = H_local / np.log(n_per_node + eps)
    G = ctx["slice_weights"].shape[-1]
    H_global = H_global / np.log(G + eps)
    

    z = pos3[:, 2]
    valid_mask = (np.isfinite(roughness_at_nodes) & 
                  np.isfinite(H_local) & np.isfinite(H_global))
    
    if height_slice is not None:
        h, tol = float(height_slice[0]), float(height_slice[1])
        valid_mask &= (np.abs(z - h) < tol)
        title_suffix = f" (z≈{h:.0f}m)"
    else:
        title_suffix = ""
    
    R = roughness_at_nodes[valid_mask]
    Hl = H_local[valid_mask]
    Hg = H_global[valid_mask]
    
    if R.size < 50:
        print(f"[WARN] Only {R.size} valid points, skipping plot")
        return

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['figure.dpi'] = 1200
    
    width_mm, height_mm = 180, 45
    fig, axes = plt.subplots(1, 3, figsize=(width_mm/25.4, height_mm/25.4))
    
    roughness_label = roughness_type.upper()
    
    # : R vs H_local (hexbin)
    ax = axes[0]
    hb = ax.hexbin(R, Hl, gridsize=40, cmap='viridis', mincnt=1)
    plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    try:
        z_fit = np.polyfit(R, Hl, 1)
        x_line = np.linspace(R.min(), R.max(), 100)
        ax.plot(x_line, np.poly1d(z_fit)(x_line), 'r--', lw=1)
        r_p, _ = pearsonr(R, Hl)
        ax.text(0.05, 0.95, f'r={r_p:.3f}', transform=ax.transAxes, 
               fontsize=6, va='top', bbox=dict(facecolor='white', alpha=0.7))
    except:
        pass
    ax.set_xlabel(roughness_label)
    ax.set_ylabel(r'$H_{local}$')
    ax.set_title(f'{roughness_label} vs Local Entropy' + title_suffix, fontsize=7)
    
    # : R vs H_global (hexbin)
    ax = axes[1]
    hb = ax.hexbin(R, Hg, gridsize=40, cmap='viridis', mincnt=1)
    plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    try:
        z_fit = np.polyfit(R, Hg, 1)
        ax.plot(x_line, np.poly1d(z_fit)(x_line), 'r--', lw=1)
        r_p, _ = pearsonr(R, Hg)
        ax.text(0.05, 0.95, f'r={r_p:.3f}', transform=ax.transAxes,
               fontsize=6, va='top', bbox=dict(facecolor='white', alpha=0.7))
    except:
        pass
    ax.set_xlabel(roughness_label)
    ax.set_ylabel(r'$H_{global}$')
    ax.set_title(f'{roughness_label} vs Global Entropy' + title_suffix, fontsize=7)
    

    ax = axes[2]
    r_bins = np.linspace(np.nanpercentile(R, 2), np.nanpercentile(R, 98), 12)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    hl_m, hl_s, hg_m, hg_s = [], [], [], []
    for i in range(len(r_bins) - 1):
        mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
        if mask.sum() > 5:
            hl_m.append(np.mean(Hl[mask]))
            hl_s.append(np.std(Hl[mask]))
            hg_m.append(np.mean(Hg[mask]))
            hg_s.append(np.std(Hg[mask]))
        else:
            hl_m.append(np.nan)
            hl_s.append(np.nan)
            hg_m.append(np.nan)
            hg_s.append(np.nan)
    
    hl_m, hl_s = np.array(hl_m), np.array(hl_s)
    hg_m, hg_s = np.array(hg_m), np.array(hg_s)
    
    ax.errorbar(r_centers, hl_m, yerr=hl_s, fmt='o-', capsize=2, 
               label=r'$H_{local}$', markersize=3, lw=1)
    ax.errorbar(r_centers, hg_m, yerr=hg_s, fmt='s-', capsize=2,
               label=r'$H_{global}$', markersize=3, lw=1)
    ax.set_xlabel(roughness_label)
    ax.set_ylabel('Entropy')
    ax.set_title('Binned Statistics', fontsize=7)
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=1200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"[INFO] Saved {save_path}")


def load_poly_data(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()
    return poly_data


def points_to_dem(points, grid_size=(1000, 1000)):

    x_min, y_min = points[:, :2].min(axis=0)
    x_max, y_max = points[:, :2].max(axis=0)
    

    x_grid = np.linspace(x_min, x_max, grid_size[1])
    y_grid = np.linspace(y_min, y_max, grid_size[0])
    xx, yy = np.meshgrid(x_grid, y_grid)
    

    dem = griddata(
        points[:, :2], 
        points[:, 2],  
        (xx, yy),
        method='linear',
        fill_value=np.nan
    )
    

    mask = np.isnan(dem)
    if mask.any():
        dem[mask] = np.nanmean(dem)
    
    return dem, (x_grid, y_grid)

def compute_base_attributes(dem):

    attributes = {}
    

    attributes['slope'] = xdem.terrain.slope(dem)
    

    attributes['aspect'] = xdem.terrain.aspect(dem)

    attributes['curvature'] = xdem.terrain.curvature(dem)
    

    attributes['planform_curvature'] = xdem.terrain.planform_curvature(dem)
    attributes['profile_curvature'] = xdem.terrain.profile_curvature(dem)
    

    attributes['tri'] = xdem.terrain.terrain_ruggedness_index(dem)
    

    attributes['tpi'] = xdem.terrain.topographic_position_index(dem)
    

    attributes['roughness'] = xdem.terrain.roughness(dem)
    

    if hasattr(xdem.terrain, 'rugosity'):
        attributes['rugosity'] = xdem.terrain.rugosity(dem)
        
    return attributes
