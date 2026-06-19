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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from matplotlib.colors import TwoSlopeNorm


def visual_y_z_yv(node_pos, v_true, v_pred, name, args):

    # Calculate the midpoint of x coordinate
    x_center = (node_pos[:, 0].min() + node_pos[:, 0].max()) / 2

    # Find all unique x values
    unique_x_values = np.unique(node_pos[:, 0])

    # Select the x value closest to x_center
    x_mid = unique_x_values[np.argmin(np.abs(unique_x_values - x_center))]

    # Keep only the points where x equals x_mid
    idx_mid_range = node_pos[:, 0] == x_mid
    
    # Extract filtered data
    node_pos_filtered = node_pos[idx_mid_range]
    v_true_filtered = v_true[idx_mid_range]
    v_pred_filtered = v_pred[idx_mid_range]
    
    # Get filtered y and z coordinates
    y_coords = node_pos_filtered[:, 1]
    z_coords = node_pos_filtered[:, 2]

    # Calculate triangular mesh
    triang = tri.Triangulation(y_coords, z_coords)


    # **Find terrain region**
    terrain_mask = np.isclose(v_true_filtered[:, 1], 0, atol=1e-7)  
    terrain_y = y_coords[terrain_mask]  # y coordinates of terrain
    terrain_z = z_coords[terrain_mask]  # z coordinates of terrain

    # **Calculate center coordinates of each triangle**
    triangles = triang.triangles  # Get vertex indices of triangles
    y_tri_center = np.mean(y_coords[triangles], axis=1)  # Calculate y coordinate of triangle center
    z_tri_center = np.mean(z_coords[triangles], axis=1)  # Calculate z coordinate of triangle center
    

    # **Create mask**
    mask = np.zeros(len(triangles), dtype=bool)  # Initialize mask

    # **Iterate through each triangle to check if it's below terrain**
    for i in range(len(triangles)):
        y_c, z_c = y_tri_center[i], z_tri_center[i]
        
        # **Find z_terrain on terrain curve closest to y_c**
        idx_nearest = np.argmin(np.abs(terrain_y - y_c))  # Find nearest terrain y
        z_terrain = terrain_z[idx_nearest]  # Get corresponding terrain z
        
        # **If z_c is below terrain, mask this triangle**
        if z_c < z_terrain:
            mask[i] = True

    # **Apply mask**
    triang.set_mask(mask)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    vmin = min(v_true_filtered[:, 1].min(), v_pred_filtered[:, 1].min())
    vmax = max(v_true_filtered[:, 1].max(), v_pred_filtered[:, 1].max())
    levels = np.linspace(vmin, vmax, 100)

    # True and predicted values
    c1 = axs[0].tricontourf(triang, v_true_filtered[:, 1], levels=levels, cmap="Spectral_r", extend="both")
    axs[0].set_title(f'{name} - True Velocity (Y Direction)')
    axs[0].set_xlabel("Y")
    axs[0].set_ylabel("Z")
    
    c2 = axs[1].tricontourf(triang, v_pred_filtered[:, 1], levels=levels, cmap="Spectral_r", extend="both")
    axs[1].set_title(f'{name} - Predicted Velocity (Y Direction)')
    axs[1].set_xlabel("Y")

    # Add shared colorbar
    cbar = fig.colorbar(c1, ax=[axs[0], axs[1]], orientation='vertical')

    # Error heatmap
    error = v_pred_filtered[:, 1] - v_true_filtered[:, 1]
    max_err = np.max(np.abs(error))

    if max_err == 0:
        max_err = 1e-8
    
    norm = TwoSlopeNorm(vmin=-max_err, vcenter=0.0, vmax=max_err)
    c0 = axs[2].tricontourf(triang, error, levels = 100, cmap="RdBu_r", norm=norm)
    axs[2].set_title(f'{name} - Velocity Error (Y Direction)')
    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")
    fig.colorbar(c0, ax=axs[2])

    # Adjust layout to avoid overlap
    # plt.tight_layout()
    plt.show()
    
    # Save image
    png_save_path = os.path.join(args.save_path, "result")
    os.makedirs(png_save_path, exist_ok=True)
    plt.savefig(f"{png_save_path}/yv_prediction_comparison_{name}.png", dpi=1200)
    plt.close()

def visual_y_z_yv_land(node_pos, v_true, v_pred, name, args):
    """
    Plot true velocity, predicted velocity and error on y-z plane,
    selecting values within a small range near the x coordinate midpoint
    
    Parameters:
    - node_pos: Shape (N, 3), node position array, each row is (x, y, z) of a node
    - v_true: Shape (N, 3), true velocity array, each row is (vx, vy, vz) of a node
    - v_pred: Shape (N, 3), predicted velocity array, each row is (vx, vy, vz) of a node
    - name: Name used in figure title
    - x_range: Size of x coordinate range for filtering nodes near the midpoint
    """
    
    # Calculate the midpoint of x coordinate
    x_center = (node_pos[:, 0].min() + node_pos[:, 0].max()) / 2

    # Find all unique x values
    unique_x_values = np.unique(node_pos[:, 0])

    # Select the x value closest to x_center
    x_mid = unique_x_values[np.argmin(np.abs(unique_x_values - x_center))]

    # Keep only the points where x equals x_mid
    idx_mid_range = node_pos[:, 0] == x_mid
    
    # Extract filtered data
    node_pos_filtered = node_pos[idx_mid_range]
    v_true_filtered = v_true[idx_mid_range]
    v_pred_filtered = v_pred[idx_mid_range]
    
    # Get filtered y and z coordinates
    y_coords = node_pos_filtered[:, 1]
    z_coords = node_pos_filtered[:, 2]

    # Calculate triangular mesh
    triang = tri.Triangulation(y_coords, z_coords)


    # **Find terrain region** Regions with zero velocity are terrain
    terrain_mask = np.isclose(v_true_filtered[:, 1], 0, atol=1e-7)  
    terrain_y = y_coords[terrain_mask]  # y coordinates of terrain
    terrain_z = z_coords[terrain_mask]  # z coordinates of terrain

    # **Calculate center coordinates of each triangle**
    triangles = triang.triangles  # Get vertex indices of triangles
    y_tri_center = np.mean(y_coords[triangles], axis=1)  # Calculate y coordinate of triangle center
    z_tri_center = np.mean(z_coords[triangles], axis=1)  # Calculate z coordinate of triangle center
    

    # **Create mask**
    mask = np.zeros(len(triangles), dtype=bool)  # Initialize mask

    # **Iterate through each triangle to check if it's below terrain**
    for i in range(len(triangles)):
        y_c, z_c = y_tri_center[i], z_tri_center[i]
        
        # **Find z_terrain on terrain curve closest to y_c**
        idx_nearest = np.argmin(np.abs(terrain_y - y_c))  # Find nearest terrain y
        z_terrain = terrain_z[idx_nearest]  # Get corresponding terrain z
        
        # **If z_c is below terrain, mask this triangle**
        if z_c < z_terrain:
            mask[i] = True

    # **Apply mask**
    triang.set_mask(mask)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    vmin = min(v_true_filtered[:, 1].min(), v_pred_filtered[:, 1].min())
    vmax = max(v_true_filtered[:, 1].max(), v_pred_filtered[:, 1].max())
    levels = np.linspace(vmin, vmax, 100)

    # True and predicted values
    c1 = axs[0].tricontourf(triang, v_true_filtered[:, 1], levels=levels, cmap="Spectral_r", extend="both")
    axs[0].set_title(f'{name} - True Velocity (Y Direction)')
    axs[0].set_xlabel("Y")
    axs[0].set_ylabel("Z")
    axs[0].scatter(terrain_y, terrain_z, color='black', s=1)
    
    c2 = axs[1].tricontourf(triang, v_pred_filtered[:, 1], levels=levels, cmap="Spectral_r", extend="both")
    axs[1].set_title(f'{name} - Predicted Velocity (Y Direction)')
    axs[1].set_xlabel("Y")
    axs[1].scatter(terrain_y, terrain_z, color='black', s=1)

    # Add shared colorbar
    cbar = fig.colorbar(c1, ax=[axs[0], axs[1]], orientation='vertical')

    # Error heatmap
    error = v_pred_filtered[:, 1] - v_true_filtered[:, 1]
    max_err = np.max(np.abs(error))

    if max_err == 0:
        max_err = 1e-8
    
    norm = TwoSlopeNorm(vmin=-max_err, vcenter=0.0, vmax=max_err)
    c0 = axs[2].tricontourf(triang, error, levels = 100, cmap="RdBu_r", norm=norm)
    axs[2].set_title(f'{name} - Velocity Error (Y Direction)')
    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")
    axs[2].scatter(terrain_y, terrain_z, color='black', s=1)
    fig.colorbar(c0, ax=axs[2])

    # Adjust layout to avoid overlap
    # plt.tight_layout()
    plt.show()
    
    # Save image
    png_save_path = os.path.join(args.save_path, "result")
    os.makedirs(png_save_path, exist_ok=True)
    plt.savefig(f"{png_save_path}/yv_prediction_comparison_land_{name}.png", dpi=1200)
    plt.close()



def visual_y_z_yv_scatter(node_pos, v_true, v_pred, name, args):
    """
    Plot scatter plots of true velocity, predicted velocity and error on y-z plane
    (Y direction velocity component), using one layer of data closest to the center cross-section.
    
    Parameters:
    - node_pos: (N, 3) node coordinate array (x, y, z)
    - v_true:   (N, 3) true velocity array
    - v_pred:   (N, 3) predicted velocity array
    - name: Additional name in figure title
    """
    # Get center value in x direction
    x_center = (node_pos[:, 0].min() + node_pos[:, 0].max()) / 2
    unique_x_values = np.unique(node_pos[:, 0])
    x_mid = unique_x_values[np.argmin(np.abs(unique_x_values - x_center))]

    # Filter cross-section points
    idx_mid = node_pos[:, 0] == x_mid
    y_coords = node_pos[idx_mid, 1]
    z_coords = node_pos[idx_mid, 2]

    vy_true = v_true[idx_mid, 1]
    vy_pred = v_pred[idx_mid, 1]
    error = vy_pred - vy_true

    # Colormap range
    vmin = min(vy_true.min(), vy_pred.min())
    vmax = max(vy_true.max(), vy_pred.max())
    levels = np.linspace(vmin, vmax, 100)

    max_err = np.max(np.abs(error))
    if max_err == 0:
        max_err = 1e-8
    norm_err = TwoSlopeNorm(vmin=-max_err, vcenter=0.0, vmax=max_err)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    # True velocity scatter plot
    sc1 = axs[0].scatter(y_coords, z_coords, c=vy_true, cmap='Spectral_r', s=2, vmin=vmin, vmax=vmax)
    axs[0].set_title(f'{name} - True Velocity (Y Direction)')
    axs[0].set_xlabel("Y")
    axs[0].set_ylabel("Z")
    fig.colorbar(sc1, ax=axs[0])

    # Predicted velocity scatter plot
    sc2 = axs[1].scatter(y_coords, z_coords, c=vy_pred, cmap='Spectral_r', s=2, vmin=vmin, vmax=vmax)
    axs[1].set_title(f'{name} - Predicted Velocity (Y Direction)')
    axs[1].set_xlabel("Y")
    axs[1].set_ylabel("Z")
    fig.colorbar(sc2, ax=axs[1])

    # Error scatter plot
    sc3 = axs[2].scatter(y_coords, z_coords, c=error, cmap='RdBu_r', s=2, norm=norm_err)
    axs[2].set_title(f'{name} - Velocity Error (Y Direction)')
    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")
    fig.colorbar(sc3, ax=axs[2])

    plt.show()
    png_save_path = os.path.join(args.save_path, "result")
    os.makedirs(png_save_path, exist_ok=True)
    plt.savefig(f"{png_save_path}/yv_prediction_comparison_scatter_{name}.png", dpi=1200)
    plt.close()


def visual_y_height_yv(node_pos, v_true, v_pred, height, name, args, imgpath):
    """
    Plot comparison curves of true and predicted velocity in y direction
    at x midpoint, at a position with specified height above terrain.

    Parameters:
    - node_pos: (N, 3), each node's (x, y, z)
    - v_true: (N, 3), true velocity (vx, vy, vz)
    - v_pred: (N, 3), predicted velocity (vx, vy, vz)
    - height: float, height (meters)
    - name: string used for file naming
    """

    # Calculate x midpoint
    x_center = (node_pos[:, 0].min() + node_pos[:, 0].max()) / 2
    unique_x_values = np.unique(node_pos[:, 0])  # Sorted x coordinates
    x_mid = unique_x_values[np.argmin(np.abs(unique_x_values - x_center))]  # np.argmin returns index of minimum

    # Filter points where x = x_mid
    idx_mid_range = node_pos[:, 0] == x_mid
    node_pos_filtered = node_pos[idx_mid_range]  # Midpoint position
    v_true_filtered = v_true[idx_mid_range]  # Midpoint velocity
    v_pred_filtered = v_pred[idx_mid_range]  # Midpoint velocity

    y_coords = node_pos_filtered[:, 1]
    z_coords = node_pos_filtered[:, 2]

    # Find terrain (points with zero velocity)
    terrain_mask = np.isclose(v_true_filtered[:, 1], 0, atol=1e-7)  
    terrain_y = y_coords[terrain_mask]  # y coordinates of terrain
    terrain_z = z_coords[terrain_mask]  # z coordinates of terrain

    # Calculate target z values at specified height
    target_z = terrain_z + height  # Terrain + height

    # For each y, find the point closest to target_z
    selected_v_true = []
    selected_v_pred = []
    selected_y = []

    y_epsilon = 50.0  # Precision for y matching
    z_tolerance = 50.0  # Maximum allowed deviation in z direction (10 meters)

    for y_val, z_target in zip(terrain_y, target_z):
        # 1. Find all points with y value close to y_val
        mask_y = np.abs(y_coords - y_val) < y_epsilon
        z_candidates = z_coords[mask_y]

        # 2. If no candidates, skip
        if len(z_candidates) == 0:
            continue

        # 3. Find point with z closest to target_z
        z_diffs = np.abs(z_candidates - z_target)
        idx_closest = np.argmin(z_diffs)
        # 4. Check if closest z is within 10m
        if z_diffs[idx_closest] > z_tolerance:
            continue  # Exceeds 10m, skip

        # 4. Get global index from mask_y
        global_indices = np.where(mask_y)[0]  # This is the index in original array
        selected_idx = global_indices[idx_closest]

        # 5. Record vy and y
        selected_v_true.append(v_true_filtered[selected_idx, 1])  # vy
        selected_v_pred.append(v_pred_filtered[selected_idx, 1])  # vy
        selected_y.append(y_coords[selected_idx])
    # Convert to numpy arrays
    selected_y = np.array(selected_y)
    selected_v_true = np.array(selected_v_true)
    selected_v_pred = np.array(selected_v_pred)

    # Sort (by y for better plotting)
    sort_idx = np.argsort(selected_y)
    selected_y = selected_y[sort_idx]
    selected_v_true = selected_v_true[sort_idx]
    selected_v_pred = selected_v_pred[sort_idx]

    # Calculate error
    abs_error = np.abs(selected_v_true - selected_v_pred)
    rel_error = abs_error / (np.abs(selected_v_true) + 1e-8)

    # Plotting
    # Manually specified min/max velocity (Vy direction)
    vmin = -24.7909  # The 2nd value of Velocity Min you provided
    vmax = 10.8108   # The 2nd value of Velocity Max you provided

    cmap = plt.get_cmap('Spectral_r')
    color_blue = cmap(0.15)
    color_red = cmap(0.85)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Original curves
    axs[0].plot(selected_y, selected_v_true, label="True Vy", color=color_blue, linewidth=2)
    axs[0].plot(selected_y, selected_v_pred, label="Predicted Vy", color=color_red, linewidth=2)
    axs[0].set_xlabel("Y")
    axs[0].set_ylabel("Vy")
    axs[0].set_title(f"Velocity Comparison at Height {height}m - {name}")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)
    # axs[0].set_ylim(vmin, vmax)

    # ---------------- Error plot (absolute + relative) ----------------
    ax1 = axs[1]                      # Primary axis: absolute error
    ax2 = ax1.twinx()                 # Secondary axis: relative error, shared x axis

    # Absolute error curve (left y axis)
    p1 = ax1.plot(selected_y, abs_error, color = color_blue, linewidth=2, label="Absolute Error")
    ax1.set_ylabel("Absolute Error |True - Pred|", color=color_blue)
    ax1.tick_params(axis='y', labelcolor=color_blue)

    # Relative error curve (right y axis)
    p2 = ax2.plot(selected_y, rel_error, color = color_red, linewidth=2, linestyle='--', label="Relative Error")
    ax2.set_ylabel("Relative Error |True - Pred| / (|True| + ε)", color=color_red)
    ax2.tick_params(axis='y', labelcolor= color_red)

    # Title and grid
    ax1.set_xlabel("Y")
    ax1.set_title(f"Error at Height {height}m - {name}")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Merge legends
    lines = p1 + p2
    labels = [line.get_label() for line in lines]
    axs[1].legend(lines, labels, loc='upper right')
    


    # Save image
    png_save_path = imgpath
    os.makedirs(png_save_path, exist_ok=True)
    plt.savefig(f"{png_save_path}/yv_height_{height}m_{name}.png", dpi=1200)
    plt.close()



def visual_3d_plane_velocity(node_pos, v_true, v_pred, height, name='', args=None, imgpath='./', component='y'):
    """
    Plot velocity on a 3D plane at specified height (3D surface plot + contour plot)
    
    Parameters:
    - node_pos: (N, 3), each node's (x, y, z)
    - v_true: (N, 3), true velocity (vx, vy, vz)
    - v_pred: (N, 3), predicted velocity (vx, vy, vz)
    - height: float, height (meters)
    - component: string, velocity component ('x', 'y', 'z')
    - name: string used for file naming
    - args: parameter object (optional)
    - imgpath: image save path
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import os

    # Velocity component index mapping
    component_idx = {'x': 0, 'y': 1, 'z': 2}
    if component not in component_idx:
        raise ValueError("component must be 'x', 'y', or 'z'")
    
    comp_idx = component_idx[component]
    
    # Find terrain points (points with zero velocity)
    terrain_mask = (np.abs(v_true[:, 0]) < 1e-7) & (np.abs(v_true[:, 1]) < 1e-7) & (np.abs(v_true[:, 2]) < 1e-7)
    
    if not np.any(terrain_mask):
        print("Warning: No terrain points found, using points with minimum z coordinate as terrain reference")
        # If no points with zero velocity found, use lowest points as terrain
        min_z_indices = np.where(node_pos[:, 2] == np.min(node_pos[:, 2]))[0]
        terrain_points = node_pos[min_z_indices]
    else:
        terrain_points = node_pos[terrain_mask]
    
    print(f"Found {len(terrain_points)} terrain points")
    
    # Create target height positions for each terrain point
    target_positions = []
    found_v_true = []
    found_v_pred = []
    found_positions = []
    
    search_tolerance = 20.0  # Search tolerance
    
    for terrain_point in terrain_points:
        # Calculate target position: ground point + specified height
        target_x, target_y = terrain_point[0], terrain_point[1]
        target_z = terrain_point[2] + height
        
        # Find closest point to target position in original data
        distances = np.sqrt((node_pos[:, 0] - target_x)**2 + 
                           (node_pos[:, 1] - target_y)**2 + 
                           (node_pos[:, 2] - target_z)**2)
        
        # Find nearest point
        min_dist_idx = np.argmin(distances)
        min_distance = distances[min_dist_idx]
        
        # If distance is within tolerance, use this point
        if min_distance < search_tolerance:
            # Use ground point's x,y coordinates and target height z coordinate
            found_positions.append([target_x, target_y, target_z])
            found_v_true.append(v_true[min_dist_idx, comp_idx])
            found_v_pred.append(v_pred[min_dist_idx, comp_idx])
    
    if len(found_positions) < 4:
        print(f"Warning: Too few data points found at height {height}m ({len(found_positions)}), cannot generate image")
        return
    
    # Convert to numpy arrays
    plot_positions = np.array(found_positions)
    plot_v_true = np.array(found_v_true)
    plot_v_pred = np.array(found_v_pred)
    plot_error = np.abs(plot_v_true - plot_v_pred)
    
    print(f"Found {len(plot_positions)} valid data points at height {height}m")
    
    # Create interpolation grid for 3D surface plot
    x_surf = np.linspace(plot_positions[:, 0].min(), plot_positions[:, 0].max(), 200)
    y_surf = np.linspace(plot_positions[:, 1].min(), plot_positions[:, 1].max(), 200)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
    
    # Interpolate terrain height to regular grid
    try:
        terrain_z = griddata((terrain_points[:, 0], terrain_points[:, 1]), terrain_points[:, 2],
                            (X_surf, Y_surf), method='cubic', fill_value=np.nan)
    except:
        terrain_z = griddata((terrain_points[:, 0], terrain_points[:, 1]), terrain_points[:, 2],
                            (X_surf, Y_surf), method='linear', fill_value=np.nan)
    
    # Create Z surface for 3D plot (terrain shape + specified height)
    Z_surf = terrain_z + height
    
    # Interpolate velocity data to regular grid (for 3D surface and contour plots)
    try:
        V_true_surf = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_true, 
                              (X_surf, Y_surf), method='cubic', fill_value=np.nan)
        V_pred_surf = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_pred, 
                              (X_surf, Y_surf), method='cubic', fill_value=np.nan)
    except:
        V_true_surf = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_true, 
                              (X_surf, Y_surf), method='linear', fill_value=np.nan)
        V_pred_surf = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_pred, 
                              (X_surf, Y_surf), method='linear', fill_value=np.nan)
    
    V_error_surf = np.abs(V_true_surf - V_pred_surf)
    
    # Create higher resolution grid for contour plots
    x_grid = np.linspace(plot_positions[:, 0].min(), plot_positions[:, 0].max(), 200)
    y_grid = np.linspace(plot_positions[:, 1].min(), plot_positions[:, 1].max(), 200)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate velocity data (for contour plots)
    try:
        Z_true = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_true, 
                         (X_grid, Y_grid), method='cubic', fill_value=np.nan)
        Z_pred = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_pred, 
                         (X_grid, Y_grid), method='cubic', fill_value=np.nan)
    except:
        Z_true = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_true, 
                         (X_grid, Y_grid), method='linear', fill_value=np.nan)
        Z_pred = griddata((plot_positions[:, 0], plot_positions[:, 1]), plot_v_pred, 
                         (X_grid, Y_grid), method='linear', fill_value=np.nan)
    
    Z_error = np.abs(Z_true - Z_pred)

    # # 3D surface plot: use true velocity range
    # v_min_surf = np.nanmin(V_true_surf)
    # v_max_surf = np.nanmax(V_true_surf)
    
    # # Contour plot: use true velocity range
    # v_min_contour = np.nanmin(Z_true)
    # v_max_contour = np.nanmax(Z_true)


    v_min_surf = np.nanmin([np.nanmin(V_true_surf), np.nanmin(V_pred_surf)])
    v_max_surf = np.nanmax([np.nanmax(V_true_surf), np.nanmax(V_pred_surf)])
    
    # Contour plot: use common range of true and predicted values
    v_min_contour = np.nanmin([np.nanmin(Z_true), np.nanmin(Z_pred)])
    v_max_contour = np.nanmax([np.nanmax(Z_true), np.nanmax(Z_pred)])
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. True velocity 3D surface plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # Normalize velocity data for color mapping
    v_true_norm = (V_true_surf - np.nanmin(V_true_surf)) / (np.nanmax(V_true_surf) - np.nanmin(V_true_surf))
    surface1 = ax1.plot_surface(X_surf, Y_surf, Z_surf, facecolors=plt.cm.Spectral_r(v_true_norm),
                                alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'True V{component} Surface at Height {height}m - {name}')
    # Add colorbar
    m1 = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r)
    m1.set_array(V_true_surf)
    m1.set_clim(v_min_surf, v_max_surf)
    plt.colorbar(m1, ax=ax1, shrink=0.5, label=f'V{component} True')
    
    # 2. Predicted velocity 3D surface plot
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    v_pred_norm = (V_pred_surf - np.nanmin(V_pred_surf)) / (np.nanmax(V_pred_surf) - np.nanmin(V_pred_surf))
    surface2 = ax2.plot_surface(X_surf, Y_surf, Z_surf, facecolors=plt.cm.Spectral_r(v_pred_norm),
                                alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Predicted V{component} Surface at Height {height}m - {name}')
    # Add colorbar
    m2 = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r)
    m2.set_array(V_pred_surf)
    m2.set_clim(v_min_surf, v_max_surf)
    plt.colorbar(m2, ax=ax2, shrink=0.5, label=f'V{component} Pred')
    
    # 3. Error 3D surface plot
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    v_error_norm = (V_error_surf - np.nanmin(V_error_surf)) / (np.nanmax(V_error_surf) - np.nanmin(V_error_surf))
    surface3 = ax3.plot_surface(X_surf, Y_surf, Z_surf, facecolors=plt.cm.Reds(v_error_norm),
                                alpha=0.8, linewidth=0, antialiased=True)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Absolute Error Surface at Height {height}m - {name}')
    # Add colorbar
    m3 = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
    m3.set_array(V_error_surf)
    plt.colorbar(m3, ax=ax3, shrink=0.5, label='Absolute Error')



    contour_levels = np.linspace(v_min_contour, v_max_contour, 20)
    # 4. True velocity contour plot
    ax4 = fig.add_subplot(2, 3, 4)
    contour1 = ax4.contourf(X_grid, Y_grid, Z_true, levels=contour_levels, cmap='Spectral_r', 
                           vmin=v_min_contour, vmax=v_max_contour)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title(f'True V{component} Contour - {name}')
    plt.colorbar(contour1, ax=ax4)
    
    # 5. Predicted velocity contour plot
    ax5 = fig.add_subplot(2, 3, 5)
    contour2 = ax5.contourf(X_grid, Y_grid, Z_pred, levels=contour_levels, cmap='Spectral_r',
                           vmin=v_min_contour, vmax=v_max_contour)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title(f'Predicted V{component} Contour - {name}')
    plt.colorbar(contour2, ax=ax5)
    
    # 6. Error contour plot
    ax6 = fig.add_subplot(2, 3, 6)
    contour3 = ax6.contourf(X_grid, Y_grid, Z_error, levels=20, cmap='Reds')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title(f'Error Contour - {name}')
    plt.colorbar(contour3, ax=ax6)
    
    plt.tight_layout()
    
    # Save image
    os.makedirs(imgpath, exist_ok=True)
    plt.savefig(f"{imgpath}/3d_plane_v{component}_height_{height}m_{name}.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print(f"3D plane visualization completed:")
    print(f"  Height: {height}m")
    print(f"  Velocity component: V{component}")
    print(f"  Number of data points: {len(plot_positions)}")
    print(f"  True velocity range: [{np.min(plot_v_true):.4f}, {np.max(plot_v_true):.4f}]")
    print(f"  Predicted velocity range: [{np.min(plot_v_pred):.4f}, {np.max(plot_v_pred):.4f}]")
    print(f"  Mean absolute error: {np.mean(plot_error):.4f}")
    print(f"  Max absolute error: {np.max(plot_error):.4f}")
    print(f"  Z coordinate range: [{np.min(plot_positions[:, 2]):.4f}, {np.max(plot_positions[:, 2]):.4f}]")
    
def visual_y_height_yv_scatter(node_pos, v_true, v_pred, height, name, args, imgpath):
    """
    Plot scatter comparison of true and predicted velocity in y direction
    at x midpoint, at a position with specified height above terrain.

    Parameters:
    - node_pos: (N, 3), each node's (x, y, z)
    - v_true: (N, 3), true velocity (vx, vy, vz)
    - v_pred: (N, 3), predicted velocity (vx, vy, vz)
    - height: float, height (meters)
    - name: string used for file naming
    """

    # Calculate x midpoint
    x_center = (node_pos[:, 0].min() + node_pos[:, 0].max()) / 2
    unique_x_values = np.unique(node_pos[:, 0])
    x_mid = unique_x_values[np.argmin(np.abs(unique_x_values - x_center))]

    # Filter points where x = x_mid
    idx_mid_range = node_pos[:, 0] == x_mid
    node_pos_filtered = node_pos[idx_mid_range]
    v_true_filtered = v_true[idx_mid_range]
    v_pred_filtered = v_pred[idx_mid_range]

    y_coords = node_pos_filtered[:, 1]
    z_coords = node_pos_filtered[:, 2]

    # Find terrain (points with zero velocity)
    terrain_mask = np.isclose(v_true_filtered[:, 1], 0, atol=1e-7)
    terrain_y = y_coords[terrain_mask]
    terrain_z = z_coords[terrain_mask]

    # Calculate target z values at specified height
    target_z = terrain_z + height

    selected_v_true = []
    selected_v_pred = []
    selected_y = []

    y_epsilon = 50.0  # y matching precision
    z_tolerance = 50.0  # Maximum allowed z deviation (10m)

    for y_val, z_target in zip(terrain_y, target_z):
        mask_y = np.abs(y_coords - y_val) < y_epsilon
        z_candidates = z_coords[mask_y]

        if len(z_candidates) == 0:
            continue

        z_diffs = np.abs(z_candidates - z_target)
        idx_closest = np.argmin(z_diffs)

        if z_diffs[idx_closest] > z_tolerance:
            continue

        global_indices = np.where(mask_y)[0]
        selected_idx = global_indices[idx_closest]

        selected_v_true.append(v_true_filtered[selected_idx, 1])  # vy
        selected_v_pred.append(v_pred_filtered[selected_idx, 1])  # vy
        selected_y.append(y_coords[selected_idx])

    selected_y = np.array(selected_y)
    selected_v_true = np.array(selected_v_true)
    selected_v_pred = np.array(selected_v_pred)

    sort_idx = np.argsort(selected_y)
    selected_y = selected_y[sort_idx]
    selected_v_true = selected_v_true[sort_idx]
    selected_v_pred = selected_v_pred[sort_idx]

    # ============================
    # Plot scatter chart
    # ============================
    # Manually specified min/max velocity (Vy direction)
    vmin = -24.7909  # The 2nd value of Velocity Min you provided
    vmax = 10.8108   # The 2nd value of Velocity Max you provided
    cmap = plt.get_cmap('Spectral_r')
    color_blue = cmap(0.15)
    color_red = cmap(0.85)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(selected_y, selected_v_true, label="True Vy", color=color_blue, s=10, alpha=0.8)
    plt.scatter(selected_y, selected_v_pred, label="Predicted Vy", color=color_red, s=10, alpha=0.8, marker='x')
    plt.xlabel("Y")
    plt.ylabel("Vy")
    plt.title(f"Velocity Scatter at Height {height}m - {name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.ylim(vmin, vmax)

    # Save image
    png_save_path = imgpath
    os.makedirs(png_save_path, exist_ok=True)
    os.makedirs(png_save_path, exist_ok=True)
    plt.savefig(f"{png_save_path}/yv_height_scatter_{height}m_{name}.png", dpi=1200)
    plt.close()

if __name__ == '__main__':
    from shapenet_velocity import Car_Dataset_v
    test_dataset = Car_Dataset_v( 
        data_path = "../../data_process_result/",
        mode="train",
        sample = False
        )
    
    for i in range(len(test_dataset)):
        input_data, name = test_dataset[i]
        filename = name.split('/')[-1]
        print(name, filename)
        node_pos = input_data['node_pos']  # (N, 6) -> (x, y, z, centroid_x, centroid_y, centroid_z)
        velocity = input_data['velocity']
        
        unscale_node_pos = test_dataset.unscale_pos(node_pos)

        velocity = velocity.detach().cpu().numpy()
        unscale_node_pos = unscale_node_pos.detach().cpu().numpy()
        # visual_y_z_yv(node_pos, velocity, velocity, filename)
        # visual_y_z_yv_scatter(node_pos, velocity, velocity, filename)
        # visual_y_z_yv_land(node_pos, velocity, velocity, filename)

        visual_y_height_yv(unscale_node_pos, velocity, velocity-1, 500, filename)
        visual_y_height_yv_scatter(unscale_node_pos, velocity, velocity-1, 500, filename)
        if i == 5:
            break