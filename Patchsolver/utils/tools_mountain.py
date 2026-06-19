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

"""
Uses your existing xdem functions to compute terrain attributes
"""

import os
import torch
import argparse
import yaml
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d as bs2d
from scipy.interpolate import griddata

# ========= Triple plot (left H_local / middle H_global / right profile or histogram), based on AGL =========
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
import warnings

import xdem
from rasterio.transform import from_origin

# ============== Add this plotting function to your script ==============

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
    """
    Plot scatter plot of terrain roughness vs attention entropy
    
    Args:
        ctx: model._attn_hook_ctx
        pos3: (N,3) denormalized coordinates
        perm, inv_perm: sorting mapping
        case_dir: case path containing bottom subdirectory
        save_path: save path
        height_slice: (height, tol) height filtering
        roughness_type: which roughness metric to use
    """
    from scipy.stats import pearsonr, spearmanr
    
    # ---------- 1) Load terrain and compute roughness ----------
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
    points = vtk_to_numpy(bottom_data.GetPoints().GetData())
    
    # Convert to DEM
    dem_array, (x_grid, y_grid) = points_to_dem(points, grid_size=grid_size)
    resolution = (x_grid[1] - x_grid[0] + y_grid[1] - y_grid[0]) / 2
    
    # Create xdem.DEM object
    transform = from_origin(
        points[:, 0].min(),
        points[:, 1].max(),
        resolution, resolution
    )
    dem = xdem.DEM.from_array(
        dem_array,
        transform=transform,
        crs="EPSG:32633",
        nodata=-9999
    )
    
    # Compute terrain attributes
    attributes = compute_base_attributes(dem)
    
    # Select roughness type
    if roughness_type not in attributes:
        print(f"[WARN] {roughness_type} not in attributes, using 'tri'")
        roughness_type = 'tri'
    
    roughness_grid = attributes[roughness_type]
    # Handle masked array
    if hasattr(roughness_grid, 'data'):
        roughness_grid = np.asarray(roughness_grid.data)
    else:
        roughness_grid = np.asarray(roughness_grid)
    
    # ---------- 2) Interpolate roughness to flow field points ----------
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    grid_vals = roughness_grid.ravel()
    valid = np.isfinite(grid_vals)
    
    roughness_at_nodes = griddata(
        grid_pts[valid], grid_vals[valid],
        pos3[:, :2],
        method='linear',
        fill_value=np.nan
    )
    
    # ---------- 3) Compute H_local and H_global ----------
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
    
    # Normalization
    n_sorted = np.concatenate([np.full(s, s, dtype=np.int32) for s in sections])
    n_per_node = n_sorted[inv_perm]
    H_local = H_local / np.log(n_per_node + eps)
    G = ctx["slice_weights"].shape[-1]
    H_global = H_global / np.log(G + eps)
    
    # ---------- 4) Data filtering ----------
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
    
    # ---------- 5) Plotting ----------
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['figure.dpi'] = 300
    
    width_mm, height_mm = 160, 45
    fig, axes = plt.subplots(1, 3, figsize=(width_mm/25.4, height_mm/25.4))
    
    roughness_label = roughness_type.upper()
    
    # Left: R vs H_local (hexbin)
    ax = axes[0]
    hb = ax.hexbin(R, Hl, gridsize=40, cmap='viridis', mincnt=1)
    plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    # Trend line
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
    
    # Middle: R vs H_global (hexbin)
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
    
    # Right: Binned statistics
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
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"[INFO] Saved {save_path}")