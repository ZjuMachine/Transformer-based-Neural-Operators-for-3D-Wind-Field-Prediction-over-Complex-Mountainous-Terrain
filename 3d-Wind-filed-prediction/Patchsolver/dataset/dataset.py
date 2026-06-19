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
import vtk
import os
import itertools
import random
import numpy as np
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy

from scipy.spatial import cKDTree as KDTree

def get_sparse_data(init, wind_uvws, bottom_points,heights=(50,100,200,300), n_xy=100,terrain_knn=32,terrain_quantile=5):

    pos=init[:,:3] #  （N，7）-->(N,3)

    pos = np.asarray(pos, dtype=np.float32)
    wind_uvws = np.asarray(wind_uvws, dtype=np.float32)
    bottom_points = np.asarray(bottom_points, dtype=np.float32)

    # xy range
    xmin, ymin = np.min(pos[:, :2], axis=0)
    xmax, ymax = np.max(pos[:, :2], axis=0)

    # sparse wind   
    xy_grid = uniform_xy_grid(xmin, xmax, ymin, ymax, n_xy=n_xy)  # (100,2)

    # z ground
    z_ground = estimate_terrain_z_from_bottom(
        xy_grid, bottom_points, knn=terrain_knn, q_percentile=terrain_quantile
    )  # (n_xy,)

    # 4) z = z_ground + h
    planes = []
    for h in heights:
        z = (z_ground + float(h)).astype(np.float32)[:, None]  # (n_xy,1)
        planes.append(np.concatenate([xy_grid, z], axis=1))
    known_pts = np.concatenate(planes, axis=0)                  # (L*n_xy,3)

    # 5)smaple 
    kd_all = _build_kdtree(pos)
    _, nn_idx_for_known = kd_all.query(known_pts, k=1)
    known_winds = wind_uvws[nn_idx_for_known]                   # (L*n_xy,3)

    # 6) pos
    kd_known = _build_kdtree(known_pts)
    _, which_known = kd_known.query(pos, k=1)
    sparse_uvws = known_winds[which_known]                      # (N,3)

    return_init = np.hstack((init, sparse_uvws))                # 水平堆叠
    return return_init

def uniform_xy_grid(xmin, xmax, ymin, ymax, n_xy=100):

    n_side = int(np.sqrt(n_xy))
    if n_side * n_side != n_xy:

        n_x = int(np.ceil(np.sqrt(n_xy)))
        n_y = int(np.ceil(n_xy / n_x))
    else:
        n_x = n_y = n_side
    xs = np.linspace(xmin, xmax, n_x, dtype=np.float32)
    ys = np.linspace(ymin, ymax, n_y, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    P = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return P[:n_xy, :]  

def _build_kdtree(arr):

    return KDTree(np.asarray(arr, dtype=np.float32))


def estimate_terrain_z_from_bottom(xy_pts, bottom_points, knn=32, q_percentile=5):

    xy_pts = np.asarray(xy_pts, dtype=np.float32)
    bp = np.asarray(bottom_points, dtype=np.float32)
    if bp.ndim != 2 or bp.shape[1] < 3:
        raise ValueError(f"bottom_points shape should be(M,3), get {bp.shape}")

    tree = _build_kdtree(bp[:, :2])
    k = min(int(knn), len(bp))
    if k <= 1:
        _, idx = tree.query(xy_pts, k=1)
        z = bp[idx, 2]
        return z.astype(np.float32)

    _, idx = tree.query(xy_pts, k=k)  # idx: (P,k)
    z_neighbors = bp[idx, 2]          # (P,k)
    z = np.percentile(z_neighbors, q=q_percentile, axis=1)
    return z.astype(np.float32)

def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output

def load_poly_data(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()
    return poly_data

def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    # dist： (N, 1)

    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    # visualize_poly_data(poly_data, surface_filter)
    # poly_data.GetPointData().SetScalars(None)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    # normal_filter.SetSplitting(0)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    '''
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    '''

    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    # c2p.ProcessAllArraysOn()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    # return unstructured_grid_data
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    # print(np.max(np.max(np.abs(normal), axis=1)), np.min(np.max(np.abs(normal), axis=1)))
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
    # print(normal)
    return normal


def visualize_poly_data(poly_data, surface_filter, normal_filter=None):
    if normal_filter is not None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData(normal_filter.GetOutput())
        # mask.RandomModeOn()
        mask.Update()
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(0.1)
        glyph.Update()
        norm_mapper = vtk.vtkPolyDataMapper()
        norm_mapper.SetInputData(normal_filter.GetOutput())
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputData(glyph.GetOutput())
        norm_actor = vtk.vtkActor()
        norm_actor.SetMapper(norm_mapper)
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(1, 0, 0)
        norm_render = vtk.vtkRenderer()
        norm_render.AddActor(norm_actor)
        norm_render.SetBackground(0, 1, 0)
        glyph_render = vtk.vtkRenderer()
        glyph_render.AddActor(glyph_actor)
        glyph_render.AddActor(norm_actor)
        glyph_render.SetBackground(0, 0, 1)

    scalar_range = poly_data.GetScalarRange()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(surface_filter.GetOutputPort())
    mapper.SetScalarRange(scalar_range)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)
    if normal_filter is not None:
        renderer_window.AddRenderer(norm_render)
        renderer_window.AddRenderer(glyph_render)
    renderer_window.Render()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()


def get_datalist(root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(samples):
        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, 'x.npy'))
            target = np.load(os.path.join(save_path, 'y.npy'))
            pos = np.load(os.path.join(save_path, 'pos.npy'))
            surf = np.load(os.path.join(save_path, 'surf.npy'))
            edge_index = np.load(os.path.join(save_path, 'edge_index.npy'))
        else:
            file_name_press = os.path.join(root, os.path.join(s, 'quadpress_smpl.vtk'))
            file_name_velo = os.path.join(root, os.path.join(s, 'hexvelo_smpl.vtk'))

            if not os.path.exists(file_name_press) or not os.path.exists(file_name_velo):
                continue

            unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
            unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

            velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
            press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
            points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
            points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())

            # edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
            # edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

            sdf_velo, normal_velo = get_sdf(points_velo, points_press) #sdf is signed distance function, 有符号距离函数：一种在计算机图形学中用于表示形状的数学函数，它为每个点分配一个实数，表示该点到形状表面的距离，正值表示在形状外部，负值表示在形状内部。
            sdf_press = np.zeros(points_press.shape[0])
            normal_press = get_normal(unstructured_grid_data_press)

            surface = {tuple(p) for p in points_press}
            exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
            velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

            pos_ext = points_velo[exterior_indices]
            pos_surf = points_press
            sdf_ext = sdf_velo[exterior_indices]
            sdf_surf = sdf_press
            normal_ext = normal_velo[exterior_indices]
            normal_surf = normal_press
            velo_ext = velo[exterior_indices]
            velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3) for p in pos_surf])
            press_ext = np.zeros([len(exterior_indices), 1])
            press_surf = press

            init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
            init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
            target_ext = np.c_[velo_ext, press_ext]
            target_surf = np.c_[velo_surf, press_surf]

            surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
            pos = np.concatenate([pos_ext, pos_surf])
            init = np.concatenate([init_ext, init_surf])
            target = np.concatenate([target_ext, target_surf])
            # edge_index = get_edge_index(pos, edges_press, edges_velo)

            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, 'x.npy'), init)
                np.save(os.path.join(save_path, 'y.npy'), target)
                np.save(os.path.join(save_path, 'pos.npy'), pos)
                np.save(os.path.join(save_path, 'surf.npy'), surf)
                # np.save(os.path.join(save_path, 'edge_index.npy'), edge_index)

        surf = torch.tensor(surf)
        pos = torch.tensor(pos)
        x = torch.tensor(init)
        y = torch.tensor(target)
        # edge_index = torch.tensor(edge_index)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        # data = Data(pos=pos, x=x, y=y, surf=surf.bool(), edge_index=edge_index)
        data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        dataset.append(data)

    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_in) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)

        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

    return dataset

def get_datalist_windey(args,root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(samples):

        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, 'x.npy'))
            target = np.load(os.path.join(save_path, 'y.npy'))

            if args.use_sparse==True:
                
                # read data
                case_dir = os.path.join(root, s) #case_dir is /dell_mnt/windey_data/newcase10/result/0.0/VTK
                bottom_dir = os.path.join(case_dir, 'bottom')
                bottom_files = [f for f in os.listdir(bottom_dir) if os.path.isfile(os.path.join(bottom_dir, f))]
                bottom_vtk_files = [f for f in bottom_files if f.endswith('.vtk')]
                bottom_vtk_file = os.path.join(bottom_dir, bottom_vtk_files[0])
                bottom_data = load_poly_data(bottom_vtk_file)
                bottom_points = vtk_to_numpy(bottom_data.GetPoints().GetData())
                n_xy_points = args.sparse_ratio
                init=get_sparse_data(init,target,bottom_points,n_xy=1000)
                # print(init.shape)
            # pos = np.load(os.path.join(save_path, 'pos.npy'))
            # surf = np.load(os.path.join(save_path, 'surf.npy'))
            # edge_index = np.load(os.path.join(save_path, 'edge_index.npy'))

        else:

            case_dir = os.path.join(root, s) #case_dir is /dell_mnt/windey_data/newcase10/result/0.0/VTK
            files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
            vtk_files = [f for f in files if f.endswith('.vtk')]
            vtk_file = os.path.join(case_dir, vtk_files[0]) #there is only one vtk file in each case. vtk_file is /dell_mnt/windey_data/newcase10/result/0.0/VTK/0.0_442.vtk
            bottom_dir = os.path.join(case_dir, 'bottom')
            bottom_files = [f for f in os.listdir(bottom_dir) if os.path.isfile(os.path.join(bottom_dir, f))]
            bottom_vtk_files = [f for f in bottom_files if f.endswith('.vtk')]
            bottom_vtk_file = os.path.join(bottom_dir, bottom_vtk_files[0])

            if not os.path.exists(vtk_file):
                continue

            unstructured_grid_data = load_unstructured_grid_data(vtk_file)
            bottom_data = load_poly_data(bottom_vtk_file)
            velo = vtk_to_numpy(unstructured_grid_data.GetPointData().GetArray('U')) # U
            points = vtk_to_numpy(unstructured_grid_data.GetPoints().GetData()) # point ordinates
            bottom_points = vtk_to_numpy(bottom_data.GetPoints().GetData())
            sdf_velo, normal_velo = get_sdf(points, bottom_points)

            
            init = np.c_[points, sdf_velo, normal_velo]
            target = velo 

            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, 'x.npy'), init)
                np.save(os.path.join(save_path, 'y.npy'), target)


        x = torch.tensor(init)
        y = torch.tensor(target)


        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        # data = Data(pos=pos, x=x, y=y, surf=surf.bool(), edge_index=edge_index)
        # data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        data = Data(x=x, y=y)
        dataset.append(data)

    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_in) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)

        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

    return dataset

def get_edges(unstructured_grid_data, points, cell_size=4):
    edge_indeces = set()
    # print(vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).shape)
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(-1, cell_size + 1)
    # cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData())
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=idx, num_hops=num_hops, edge_index=data.edge_index,
                                                  relabel_nodes=True)
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim:gravity_dim + 1] - shape_pc[:, gravity_dim:gravity_dim + 1].min()
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    # data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    # print(f'r = {r}, #edges = {data.edge_index.size(1)}')
    return data


class GraphDataset(Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        # if not use_cfd_mesh:
        #     assert r is not None
        #     for i in range(len(self.datalist)):
        #         self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        # shape = get_shape(data, use_height=self.use_height)
        shape = torch.tensor(3)
        return self.datalist[idx], shape


if __name__ == '__main__':
    import numpy as np

    #file_name = '1a0bc9ab92c915167ae33d942430658c'
    file_name = '100715345ee54d7ae38b52b4ee9d36a3'

    # root = '/data/PDE_data/mlcfd_data/training_data'
    # save_path = '/data/PDE_data/mlcfd_data/preprocessed_data/param0/' + file_name
    root = './mlcfd_data/training_data'
    save_path = './mlcfd_data/preprocessed_data/param0/' + file_name
    file_name_press = 'param0/' + file_name + '/quadpress_smpl.vtk'
    file_name_velo = 'param0/' + file_name + '/hexvelo_smpl.vtk'
    file_name_press = os.path.join(root, file_name_press)
    file_name_velo = os.path.join(root, file_name_velo)
    unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
    unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

    velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
    press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
    points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
    points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())

    edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
    edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

    sdf_velo, normal_velo = get_sdf(points_velo, points_press)
    sdf_press = np.zeros(points_press.shape[0])
    normal_press = get_normal(unstructured_grid_data_press)

    surface = {tuple(p) for p in points_press}
    exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
    velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

    pos_ext = points_velo[exterior_indices]
    pos_surf = points_press
    sdf_ext = sdf_velo[exterior_indices]
    sdf_surf = sdf_press
    normal_ext = normal_velo[exterior_indices]
    normal_surf = normal_press
    velo_ext = velo[exterior_indices]
    velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3) for p in pos_surf])
    press_ext = np.zeros([len(exterior_indices), 1])
    press_surf = press

    init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
    init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
    target_ext = np.c_[velo_ext, press_ext]
    target_surf = np.c_[velo_surf, press_surf]

    surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
    pos = np.concatenate([pos_ext, pos_surf])
    init = np.concatenate([init_ext, init_surf])
    target = np.concatenate([target_ext, target_surf])

    edge_index = get_edge_index(pos, edges_press, edges_velo)

    data = Data(pos=torch.tensor(pos), edge_index=torch.tensor(edge_index))
    data = create_edge_index_radius(data, r=0.2)
    x, y = data.edge_index
    import torch_geometric

    print(max(torch_geometric.utils.degree(x)), max(torch_geometric.utils.degree(y)))

    print(points_velo.shape, points_press.shape)
    print(surf.shape, pos.shape, init.shape, target.shape, edge_index.shape)
