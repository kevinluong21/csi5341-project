import os

import fpsample
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from torch.serialization import safe_globals

import torch
from torch.utils.data import Dataset

class PointVessel(Dataset):
    def __init__(self, data_dir, split, cfg):
        super().__init__()
        self.debug = False
        self.cfg = cfg
        self.data_dir = os.path.join(data_dir, split)
        
        self.preload_data = cfg['preload']
        self.scan_prune_dist = cfg['scan_prune_dist']
        self.resolution = cfg['cond_resolution']
        self.node_resolution = cfg['node_resolution']
        self.num_nodes = cfg['num_nodes']
        self.num_condition_pts = cfg['num_points']
        self.prune_min_radius = cfg['prune_min_radius']
        self.prune_length = cfg['prune_length']
        self.node_count_vox_size = cfg['node_count_vox_size']
        self.scan_farthps = cfg['scan_farthps']
        self.node_fps = cfg['node_fps']
        self.varieties = cfg['varieties']
        self.debug_vis = cfg['debug_vis']
        self.dataset_norm = False
        self.multiply_data = cfg['multiply_data']
        self.overfit = cfg['overfit']

        self.split = split
        self.cache_maps = {}
        self.points_datapath = [os.path.join(self.data_dir, p) for p in os.listdir(self.data_dir)]

        # compute normalization stats
        if self.dataset_norm:
            if not os.path.exists(os.path.join(data_dir, 'variety_max_xy_extent.npy')):
                self.variety_max_xy_extent = self.compute_variety_max_xy_extent()
                np.save(os.path.join(data_dir, 'variety_max_xy_extent.npy'), self.variety_max_xy_extent)
            else:
                self.variety_max_xy_extent = np.load(os.path.join(data_dir, 'variety_max_xy_extent.npy'), allow_pickle=True).item()
                print('Loaded data stats')
        
        self.nr_data = len(self.points_datapath)
        print('The size of %s data is %d'%(self.split,self.nr_data))

    def compute_variety_max_xy_extent(self):
        # compute extents for x and y for each variety
        variety_max_xy_extent = {}
        print("Computing variety max xy extent", self.points_datapath[0])
        for tree in tqdm(self.points_datapath):
            raw_data = torch.load(tree, weights_only=False)
            scan_points = raw_data['scan_points']
            scan_points -= np.mean(scan_points, axis=0)
            variety = [v for v in self.varieties if v in tree][0]
            if variety not in variety_max_xy_extent:
                variety_max_xy_extent[variety] = 0
            max_xy_extent = np.max(np.abs(scan_points[:,:2])) 
            if max_xy_extent > variety_max_xy_extent[variety]:
                variety_max_xy_extent[variety] = max_xy_extent
        return variety_max_xy_extent

    def center_and_normalize_data(self, data, filename, normalize=False):
        # move cloud xy around origin, z to 0
        cloud_offset = np.mean(data['scan_points'], axis=0).astype(np.float32)
        cloud_offset[2] = np.min(data['scan_points'][:,2]).astype(np.float32)
        
        if normalize:
            max_val = self.variety_max_xy_extent[[v for v in self.varieties if v in filename][0]]
        else:
            max_val = 1.0
        data['scan_points'] = (data['scan_points'] - cloud_offset) / max_val
        # for branch_id in data['skeleton'].branches.keys():
        #     data['skeleton'].branches[branch_id].xyz = (data['skeleton'].branches[branch_id].xyz - cloud_offset) / max_val
        data['skeleton_vertices'] = (data['skeleton_vertices'] - cloud_offset) / max_val

        return data, (cloud_offset, max_val)

    def __getitem__(self, index):
        if self.overfit:
            index = 0
        else:
            index = index % self.nr_data
        # print(self.points_datapath[index])
        # with safe_globals([np.core.multiarray._reconstruct]):
        #     raw_data = torch.load(self.points_datapath[index], weights_only=True)

        raw_data = torch.load(self.points_datapath[index], weights_only=False)
        raw_data, normalization_factors = self.center_and_normalize_data(raw_data, filename=os.path.basename(self.points_datapath[index]), normalize=self.dataset_norm)
        scan_points = raw_data['scan_points']
        highres_gt_skeleton_nodes = raw_data['skeleton_vertices']
        skeleton_edges = raw_data['skeleton_edges']
        
        if self.split == 'val_sem':
            scan_point_classes = raw_data['pred_sem']
        else:
            scan_point_classes = raw_data['scan_point_classes']
        scan_point_colors = raw_data['scan_point_colors']
        
        if self.num_condition_pts == "auto":
            # voxel downsampling
            # shuffle the points
            rand_perm = torch.randperm(scan_points.shape[0])
            scan_points = scan_points[rand_perm]
            scan_point_classes = scan_point_classes[rand_perm]
            scan_point_colors = scan_point_colors[rand_perm]
            
            _, mapping = ME.utils.sparse_quantize(coordinates=scan_points / self.scan_resolution, return_index=True, device="cpu")
            scan_points = scan_points[mapping]
            scan_point_classes = scan_point_classes[mapping]
            scan_point_colors = scan_point_colors[mapping]
        else:
            if self.scan_farthps:
                if self.num_condition_pts > scan_points.shape[0]:
                    samples_idx = np.random.choice(scan_points.shape[0], self.num_condition_pts, replace=True)
                else:
                    samples_idx = fpsample.bucket_fps_kdline_sampling(scan_points, self.num_condition_pts, h=9)
            else:
                samples_idx = np.random.choice(scan_points.shape[0], self.num_condition_pts, replace=True)
            scan_points = scan_points[samples_idx]
            scan_point_classes = scan_point_classes[samples_idx]
            scan_point_colors = scan_point_colors[samples_idx]
        
        if self.num_nodes == "auto":
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(scan_points)
            # voxel downsample the scan points
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=self.node_count_vox_size)
            num_skel_nodes = len(scan_pcd.points)
        else:
            num_skel_nodes = self.num_nodes

        # only inference, no training
        # if self.split == 'train':
        #     skel_augm_variation = num_skel_nodes*0.15
        #     num_skel_nodes += np.random.randint(-skel_augm_variation, skel_augm_variation)
            
        if num_skel_nodes > len(highres_gt_skeleton_nodes):
            num_skel_nodes = len(highres_gt_skeleton_nodes)

        # the actual skeleton points and, by extension, the parent nodes are only used for training
        # skeleton_points, node_parent_ids, _ = self.downsample_ls(highres_gt_skeleton_nodes, skeleton_edges, num_skel_nodes, fps=self.node_fps)

        # this value is never actually used during inference or training as far as I know, so set to None
        # highres_node_parent_ids = np.concatenate((np.array((-1,)), skeleton_edges[:, 0]))
        highres_node_parent_ids = None

        # only inference, no training (the else was taken out of the if statement and runs no matter what)
        # if self.split == 'train':
        #     p_concat = np.concatenate((skeleton_points, scan_points, highres_gt_skeleton_nodes), axis=0)
        #     p_concat = self.transforms(p_concat)
        #     skeleton_points = torch.tensor(p_concat[:len(skeleton_points)], dtype=torch.float32)
        #     scan_points = torch.tensor(p_concat[len(skeleton_points):len(skeleton_points)+len(scan_points)], dtype=torch.float32)
        #     highres_gt_skeleton_nodes = torch.tensor(p_concat[len(skeleton_points)+len(scan_points):], dtype=torch.float32)
        # else:
        #     skeleton_points = torch.tensor(skeleton_points, dtype=torch.float32)
        #     scan_points = torch.tensor(scan_points, dtype=torch.float32)
        #     highres_gt_skeleton_nodes = torch.tensor(highres_gt_skeleton_nodes, dtype=torch.float32)

        # the actual skeleton points are only used for training, but the inference might require any input of the same size or may fail without it, 
        # so pass a dummy of the same size instead
        # the size is the number of (skeleton nodes/vertices, 3) where 3 is because each point is in 3d
        # skeleton_points = torch.tensor(skeleton_points, dtype=torch.float32)
        skeleton_points = torch.tensor(np.zeros((len(highres_gt_skeleton_nodes), 3)), dtype=torch.float32)
        scan_points = torch.tensor(scan_points, dtype=torch.float32)
        highres_gt_skeleton_nodes = torch.tensor(highres_gt_skeleton_nodes, dtype=torch.float32)

        # similar to the skeleton points, the parent node ids are only used if the skeleton is given, but inference does not require a skeleton
        # however, we pass a dummy of the same shape that it would expect in case it fails for a missing value
        node_parent_ids = torch.tensor(np.zeros(len(highres_gt_skeleton_nodes)), dtype=torch.float32)
        
        # if self.debug_vis:
        #     print("tree", self.points_datapath[index], "skeleton_points", skeleton_points.shape, "scan_points", scan_points.shape, "orig_points", raw_data['scan_points'].shape,"highres_gt_skeleton_nodes", highres_gt_skeleton_nodes.shape)  
        #     self.vis_batch(skeleton_points, highres_gt_skeleton_nodes, scan_points, highres_node_parent_ids, node_parent_ids, scan_point_classes)

        return  (skeleton_points,
            torch.tensor(highres_gt_skeleton_nodes, dtype=torch.float32),
            torch.tensor(node_parent_ids),
            torch.tensor(scan_points, dtype=torch.float32),
            torch.tensor(scan_point_classes),
            torch.tensor(scan_point_colors, dtype=torch.float32),
            self.points_datapath[index],
            highres_node_parent_ids,
            normalization_factors,
            )

    def __len__(self):
        if self.overfit:
            return 20 if self.split == 'train' else 1
        else:
            return self.nr_data * self.multiply_data if self.split == 'train' else self.nr_data