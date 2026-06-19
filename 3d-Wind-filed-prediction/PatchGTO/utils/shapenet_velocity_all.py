import os.path
import re
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from sklearn.neighbors import KDTree  

import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import time
from tqdm import tqdm 

import matplotlib.pyplot as plt


import random
def centoirds(polydata):
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(polydata)
    cell_centers.Update()
    numpy_cell_centers = vtk_to_numpy(cell_centers.GetOutput().GetPoints().GetData()).astype(np.float32)
    return numpy_cell_centers

def read_vtk(file_path):

    reader = vtk.vtkUnstructuredGridReader() 
    reader.SetFileName(file_path)
    reader.Update()

    data = reader.GetOutput()

    
    centroid = torch.from_numpy(centoirds(data))
    
    points = data.GetPoints()
    num_points = points.GetNumberOfPoints()
    point_coords = np.array([points.GetPoint(i) for i in range(num_points)], dtype=np.float32)

    cells = data.GetCells()

    edges = []
    
    cell_types = data.GetCellTypesArray()

    num_cells = cell_types.GetNumberOfTuples()
    hexahedron_type = vtk.VTK_HEXAHEDRON
    
    all_cell = []
    for i in range(num_cells):
        if cell_types.GetValue(i) == hexahedron_type: 
            cell = vtk.vtkIdList()
            cells.GetCellAtId(i, cell)
            cell_ids = [cell.GetId(j) for j in range(cell.GetNumberOfIds())]
            all_cell.append(cell_ids)
            # print(cell_ids)
            edges.extend([(cell_ids[j], cell_ids[j + 1]) for j in range(len(cell_ids) - 1)])

            edges.append((cell_ids[-1], cell_ids[0]))
    all_cell = torch.tensor(all_cell)
    edges = np.array(edges, dtype=np.int32)
    


    velocity = vtk_to_numpy(data.GetPointData().GetArray("U")).astype(np.float32)


    point_centers = torch.zeros((point_coords.shape[0], 3))

    centroid_expanded = centroid.repeat_interleave(8, dim=0)  # (M * 8, 3)

    point_centers.scatter_add_(0, all_cell.flatten().unsqueeze(1).expand(-1, 3), centroid_expanded)

    vertex_counts = torch.bincount(all_cell.flatten(), minlength=point_coords.shape[0]).float().unsqueeze(1)
    point_centers = point_centers / vertex_counts
    
    return point_coords, edges, velocity, point_centers


def get_data(path, mode, item):

    file_path = os.path.join(item, "VTK/cutted.vtk")
    # print(file_path)

    node_pos, edges, velocity, centroid = read_vtk(file_path)

    item_lower = item.lower()
    if "chatou" in item_lower:
        label = 1
    elif "jiepai" in item_lower:
        label = 2
    elif "newcase" in item_lower:
        label = 3
    elif "taojiang" in item_lower:
        label = 4
    else:
        label = 5
    
    match = re.search(r'result/([\d.]+)', item)
    angle = float(match.group(1))

    case_label = np.stack([
        np.full((node_pos.shape[0],), label, dtype=np.float32),
        np.full((node_pos.shape[0],), angle, dtype=np.float32)
    ], axis=1)  # [N, 2]

    return node_pos, edges, velocity, centroid, case_label


class Car_Dataset_v(Dataset):
    def __init__(self, data_path, mode="train", N_target = 5000, E_target = 15000, adj_num=15, sample = True, sample_times = 1):
        super(Car_Dataset_v, self).__init__()
        assert mode in ["train", "test"]
        
        self.dataloc = []
        
        self.mode = mode
        self.fn = data_path

        self.N_target = N_target  
        self.E_target = E_target 
        self.adj_num = adj_num  
        self.sample = sample 
        self.sample_times = sample_times 
        
       
        data_dir = self.fn #os.path.join(self.fn, "Train" if mode == "train" else "Test")

        
        file_list = sorted(os.listdir(data_dir)) 

    
        self.dataloc = []

        for file in file_list:
            result_dir = os.path.join(data_dir, file, "result")
            if not os.path.isdir(result_dir):
                continue

            
            angle_dirs = sorted(os.listdir(result_dir))
            for angle in angle_dirs:
                if mode == 'Train':
                    if angle == '0.0':
                        continue
                elif mode == 'Test':
                    if angle != '0.0':
                        continue
                subdir = os.path.join(result_dir, angle)
                if os.path.isdir(subdir):
                    self.dataloc.append(subdir)
    
       
        self.dataloc = np.array(self.dataloc, dtype=str)

        

    def __len__(self):
        return len(self.dataloc) * self.sample_times
    
    def sample_node(self, nodes, N_target):

        N = nodes.shape[0]
        indices = torch.randperm(N)[:N_target] 
        sampled_nodes = nodes[indices]
        return sampled_nodes, indices
    
    def knn_edges(self, node_pos, adj_num):
 
        node_pos_np = node_pos.numpy()

        tree = KDTree(node_pos_np)
        dists, indices = tree.query(node_pos_np, k=adj_num + 1)  

        N = node_pos_np.shape[0]
        
        i_indices = np.repeat(np.arange(N), adj_num)
        j_indices = indices[:, 1:].reshape(-1) 
        edges = np.stack([i_indices, j_indices], axis=1)

        return torch.tensor(edges, dtype=torch.long)
    

    
    def sample_edges(self, edges, node_indices):
        node_indices = node_indices.cpu().numpy() 
        
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        mask = np.isin(edges[:, 0], node_indices) & np.isin(edges[:, 1], node_indices)
        sampled_edges = edges[mask]
        
        
        sampled_edges = np.vectorize(index_map.get)(sampled_edges)
        
        return torch.tensor(sampled_edges, dtype=torch.int32)  
    
    def sample_edges2(self, edges, E_target):
        sampled_edges = edges[torch.randperm(edges.shape[0])[:E_target]]
        return sampled_edges
        
    def get_singel(self, item):
        
        node_pos, edges, velocity, centroid, case_label = get_data(
            self.fn,
            self.mode,
            self.dataloc[item]
            )
        
        node_pos = torch.from_numpy(node_pos).float()
        velocity = torch.from_numpy(velocity).float()
        edges = torch.from_numpy(edges).long()
        centroid = centroid.float()
        case_label = torch.from_numpy(case_label).float()

        if self.sample:
            sampled_node_pos, node_indices = self.sample_node(node_pos, self.N_target)

            knn_edges = self.knn_edges(node_pos, self.adj_num)

            combined_edges = torch.cat([edges, knn_edges], dim=0)

            edges = self.sample_edges(combined_edges, node_indices)

            sampled_edges = self.sample_edges2(edges, self.E_target)


            sampled_velocity = velocity[node_indices]
            
            sampled_centroid = centroid[node_indices]
            sampled_case_label = case_label[node_indices]

        else:  
            sampled_node_pos = node_pos
            sampled_velocity = velocity
            sampled_centroid = centroid
            sampled_case_label = case_label

            knn_edges = self.knn_edges(sampled_node_pos, self.adj_num)
            sampled_edges = torch.cat([edges, knn_edges], dim=0)

        sampled_node_pos = self.scale_pos(sampled_node_pos)
        sampled_centroid = self.scale_centroid(sampled_centroid)
        all_pos = torch.cat([sampled_node_pos, sampled_centroid], dim=-1)
        
        input = {
            'node_pos': all_pos, # （N, 6）(x,y,z,centroid_1,centroid_2,centroid_3)
            'velocity': sampled_velocity,# (N, 3)(x,y,z)
            'edges': sampled_edges # (E, 2)
        }
        
        return input, self.dataloc[item]
        
    
    def scale_pos(self, pos):
        pos_max = torch.tensor([4164.8599, 4164.8599, 1297.3800]).to(pos.device)
        pos_min = torch.tensor([-4164.8599, -4164.8599,  62.2841]).to(pos.device)
        # pos_min = pos.min(dim=0).values
        # pos_max = pos.max(dim=0).values
        
        node_pos = (pos - pos_min.reshape(-1,3)) / (pos_max.reshape(-1,3) - pos_min.reshape(-1,3))
       
        return node_pos

    def scale_label(self, label):
        pos_max = torch.tensor([4, 337.5]).to(label.device)
        pos_min = torch.tensor([1, 0.0]).to(label.device)

        
        node_pos = (label - pos_min) / ((pos_max) - pos_min)
       
        return node_pos
    
    def scale_centroid(self, centroid):
        pos_max = torch.tensor([4111.8252, 4111.8252, 1248.8230]).to(centroid.device)
        pos_min = torch.tensor([-4111.8252, -4111.8252, 0.0000]).to(centroid.device)
        centroid = torch.nan_to_num(centroid, nan=0.0)  
        # pos_min = centroid.min(dim=0).values
        # pos_max = centroid.max(dim=0).values

        scale_range = pos_max - pos_min
        scale_range[scale_range == 0] = 1.0

        centroid = (centroid - pos_min) / scale_range

        return centroid

    def unscale_pos(self, node_pos):

        pos_max = torch.tensor([4164.8599, 4164.8599, 1297.3800]).to(node_pos.device)
        pos_min = torch.tensor([-4164.8599, -4164.8599, 62.2841]).to(node_pos.device)

        centroid_max = torch.tensor([4111.8252, 4111.8252, 1248.8230]).to(node_pos.device)
        centroid_min = torch.tensor([-4111.8252, -4111.8252, 0.00000]).to(node_pos.device)
        # p_max = torch.tensor([4, 337.5]).to(node_pos.device)
        # p_min = torch.tensor([1, 0.0]).to(node_pos.device)


        pos_part = node_pos[... , :3]
        centroid_part = node_pos[... , 3:]

        pos_part = pos_part * (pos_max - pos_min) + pos_min
        centroid_part = centroid_part * (centroid_max - centroid_min) + centroid_max
        noscale_node_pos = torch.cat([pos_part, centroid_part], dim=-1)

        return noscale_node_pos
    
    def __getitem__(self, item):

        true_idx = item % len(self.dataloc)  
        input, name = self.get_singel(true_idx)  
        return input, name 

    
if __name__ == '__main__':
    pass