# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
from numba import njit, prange
import scipy.ndimage
import os
import time

from igibson.utils.constants import MAX_CLASS_COUNT, SemanticClass

from moma_llm.utils.constants import OCCUPANCY, CLASS_NAME_TO_CLASS_ID, CLASS_ID_TO_CLASS_NAME
from moma_llm.navigation.frontier import find_frontiers, classify_frontiers
from moma_llm.utils.utils import get_obj_bounding_box


@njit(parallel=False)
def last_nonzero_numba(arr, value):
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[1]):
            for z in range(arr.shape[2], 0, -1):
                if arr[x, y, z] != 0:
                    value[x, y] = arr[x, y, z]
                    break
    return value


class SimpleSlam:
    def __init__(self, grid_size, voxel_size, sensor_range, min_points_for_detection):
        self.sensor_range = sensor_range
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.midpoint = grid_size // 2
        assert self.grid_size % 2 == 0
        self.min_points_for_detection = min_points_for_detection
        self.reset()

    @property
    def seen_instances(self):
        return set(self.instance_viewpoints.keys())
    
    def reset(self):
        self.voxel_map = np.zeros([self.grid_size] * 3, dtype=np.float32)
        self.bev_map_semantic = np.zeros([self.grid_size] * 2, dtype=np.float32)
        self.bev_map_occupancy = np.zeros_like(self.bev_map_semantic)
        # for each instance, keep track of the closest viewpoint and distance to use it for the assignment of objects to the voronoi graph
        # instance_id: (viewpoint_position, distance-to-instance)
        self.instance_viewpoints = dict()

    def world2voxel(self, world_coords):
        voxel_idx = np.round(world_coords / self.voxel_size).astype(int) + self.midpoint
        assert np.logical_and(voxel_idx > 0, voxel_idx < self.grid_size - 1).all(), world_coords
        return voxel_idx
    
    def voxel2world(self, voxel_coords):
        if isinstance(voxel_coords, tuple):
            voxel_coords = np.array(voxel_coords)
        world_coords = (voxel_coords - self.midpoint) * self.voxel_size
        return world_coords

    def _update_closest_viewpoints(self, scene, instance_seg, dist, extrinsic_inv):
        viewpoint_pos_world = extrinsic_inv[:3, 3]
        instance_seg_sorted = instance_seg[instance_seg.argsort()]
        ins_ids, ins_idxs, ins_counts = np.unique(instance_seg_sorted, return_index=True, return_counts=True)
        instance_seg_by_dist = np.split(dist, ins_idxs)
        
        def _get_volume(instance_id):
            bbox = scene.objects_by_id[instance_id].bounding_box
            return np.prod(bbox) if (bbox is not None) else np.inf
        
        for instance_id, dists, count in zip(ins_ids[1:], instance_seg_by_dist[1:], ins_counts[1:]):
            if (count > self.min_points_for_detection) or (_get_volume(instance_id) < 0.01):
                min_dist = np.min(dists)
                if self.instance_viewpoints.get(instance_id, (None, np.inf))[1] > min_dist:
                    self.instance_viewpoints[instance_id] = (viewpoint_pos_world, min_dist)

    @property
    def clipping_range(self):
        rng = int(1 / self.voxel_size)
        return self.midpoint - rng, self.midpoint + rng
    
    def delete_obj_from_voxel_map(self, obj):
        world_frame_vertex_positions = get_obj_bounding_box(obj)
        map_vertex_positions = self.world2voxel(world_frame_vertex_positions[:, :2])
        x12 = sorted(map_vertex_positions[:, 0])
        y12 = sorted(map_vertex_positions[:, 1])
        self.voxel_map[x12[0]-1:x12[1]+1, y12[0]-1:y12[1]+1] = 0        

    @staticmethod
    def _get_outside_window_filter(state):
        window_mask = state["seg"] == CLASS_NAME_TO_CLASS_ID["window"]
        wall_mask = state["seg"] == CLASS_NAME_TO_CLASS_ID["walls"]
        if np.any(window_mask) > 0:
            window_instances = list(np.unique(state["ins_seg"][window_mask]))
            masked_windows = copy.deepcopy(window_mask).squeeze(-1)
            for window_instance in window_instances:
                single_window_mask = state["ins_seg"] == window_instance
                single_window_mask = cv2.dilate(single_window_mask.astype(np.uint8), np.ones((1, 1), np.uint8), iterations = 1)
                # check if window occupies any pixel in row 0
                min_horiz, max_horiz = np.min(single_window_mask.nonzero()[1]), np.max(single_window_mask.nonzero()[1])
                min_vert, max_vert = np.min(single_window_mask.nonzero()[0]), np.max(single_window_mask.nonzero()[0])
                # occupy outside lines
                if np.any(single_window_mask[0, :]) > 0:
                    single_window_mask[0:2, min_horiz:max_horiz] = True
                if np.any(single_window_mask[:, 0]) > 0:    
                    single_window_mask[min_vert:max_vert, 0:2] = True
                if np.any(single_window_mask[:, -1]) > 0:
                    single_window_mask[min_vert:max_vert, -2:-1] = True
                if np.any(single_window_mask[-1, :]) > 0:
                    single_window_mask[-2:-1, min_horiz:max_horiz] = True
                size_param = np.mean([max_horiz - min_horiz, max_vert - min_vert]) # TODO: Tune dilation size based on distance
                # fill holes in the single_window_mask
                instance_filled_sobel = scipy.ndimage.binary_fill_holes(single_window_mask) > 0
                masked_windows[instance_filled_sobel] = True
            masked_windows = cv2.dilate(masked_windows.astype(np.uint8), np.ones((5, 5), np.uint8), iterations = 1)
            outside_window_filter = cv2.erode(masked_windows.astype(np.uint8), np.ones((12, 12), np.uint8), iterations = 1) > 0
        else:
            outside_window_filter = np.zeros_like(state["seg"], dtype=np.uint8).squeeze(-1)
        return outside_window_filter

    def _update_voxel_map(self, state, extrinsic, scene):
        pc = state["pc"].reshape(-1, 3)
        dist = np.linalg.norm(pc, axis=1)
        dist_dense = dist.reshape(state["seg"].shape)

        # FILTERING OBJECTS OUTSIDE OF WINDOWS
        outside_window_filter = self._get_outside_window_filter(state)

        # SENSING DEPTH FILTERING + WINDOW FILTERING
        within_sensing_range = (dist_dense < self.sensor_range).squeeze(-1)
        within_map_bounds = np.logical_and(~outside_window_filter, within_sensing_range)

        # overwrite everything within
        points = pc[within_map_bounds.reshape(-1)]

        rgb = copy.deepcopy(state["rgb"])
        rgb[~within_map_bounds] = 0
        seg = state["seg"][:, :, 0]
        seg[~within_map_bounds] = 0
        ins_seg = state["ins_seg"][:, :, 0]
        ins_seg[~within_map_bounds] = 0

        seg = seg[within_map_bounds].reshape(-1)
        ins_seg = ins_seg[within_map_bounds].reshape(-1)

        within_map_bounds = within_map_bounds.reshape(-1)   
        
        # Add a column of ones for the inverse transform
        points = np.c_[points, np.ones(points.shape[0])]
        
        extrinsic_inv = np.linalg.inv(extrinsic)
        self._update_closest_viewpoints(scene=scene, instance_seg=ins_seg, dist=dist[within_map_bounds], extrinsic_inv=extrinsic_inv)

        # Get map indices
        world_points = extrinsic_inv.dot(points.T).T
        voxel_idx = self.world2voxel(world_points)
        
        # voxel_idx have multiple values that refer to the same voxel. Which of those gets selected is more or less random
        # so first write the free-space values, then all other values to make sure that we don't write a freespace-class on top of an occupied-class
        is_floor = np.isin(seg, [CLASS_NAME_TO_CLASS_ID["floors"], CLASS_NAME_TO_CLASS_ID["carpet"]])
        self.voxel_map[voxel_idx[is_floor, 0], voxel_idx[is_floor, 1], voxel_idx[is_floor, 2]] = CLASS_NAME_TO_CLASS_ID["floors"]
        
        # remaining_idx = np.logical_and(~is_floor, ~np.isin(seg, [CLASS_NAME_TO_CLASS_ID["floors"], CLASS_NAME_TO_CLASS_ID["carpet"]]))
        voxel_idx_remaining = voxel_idx[~is_floor]
        seg_remaining = seg[~is_floor]
        self.voxel_map[voxel_idx_remaining[:, 0], voxel_idx_remaining[:, 1], voxel_idx_remaining[:, 2]] = seg_remaining

    def _update_2d_map(self):
        # Clip out ceiling
        clipped_map = self.voxel_map[:, :, self.clipping_range[0] : self.clipping_range[1]]
        # sum from top to bottom
        value = np.zeros_like(self.bev_map_semantic)
        bev_map = last_nonzero_numba(clipped_map, value)
        self.bev_map_semantic = bev_map
        
        occupancy_map = np.zeros_like(self.bev_map_semantic)
        occupancy_map[bev_map > 0] = OCCUPANCY.OCCUPIED
        occupancy_map[np.isin(bev_map, [CLASS_NAME_TO_CLASS_ID["floors"], CLASS_NAME_TO_CLASS_ID["carpet"]])] = OCCUPANCY.FREE
        self.bev_map_occupancy = occupancy_map
        
    def update(self, state, extrinsic, scene):
        self._update_voxel_map(state, extrinsic, scene=scene)
        self._update_2d_map()
    
    def get_frontiers(self, agent_pos_meter, occupancy_map=None):
        if occupancy_map is None:
            occupancy_map = self.bev_map_occupancy
        assert set(np.unique(occupancy_map)) == set(OCCUPANCY), "occupancy map has to follow the values of the OCCUPANCY enum"
        occupancy_map = occupancy_map.astype(np.uint8)
        
        robot_pos_pixel = self.world2voxel(agent_pos_meter)[:2]
        frontier_centers_pixel, frontier_img = find_frontiers(occupancy_map, 
                                                              robot_pos_pixel,
                                                              smoothing_kernel_size=3)
        frontier_centers_world = self.voxel2world(frontier_centers_pixel)
        
        frontier_classification = classify_frontiers(slam=self, frontier_img=frontier_img, frontier_centers_world=frontier_centers_world)
        
        return frontier_centers_world, frontier_img, frontier_classification
        