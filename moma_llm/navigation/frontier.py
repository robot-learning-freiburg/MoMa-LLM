# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import cv2
import numpy as np
import skimage

from moma_llm.topology.topology import TopologyMapping
from moma_llm.utils.constants import FRONTIER_CLASSIFICATION, OCCUPANCY


def find_frontiers(occupancy_map, agent_pos_pixel, smoothing_kernel_size: int):   
    """
    Find frontier points
    1. dilate the map for smoothing
    2. find all points at the boundary of free space and unexplored space
    3. only take those frontiers that are connected to the agent's position in the dilated map
    4. take the center of each connected frontier as center point
    """ 
    assert occupancy_map.dtype == np.uint8, occupancy_map.dtype
    
    # Dilate the occupancy map to smoothen the map
    kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size), np.uint8)
    occupancy_map_dilated = cv2.dilate(occupancy_map, kernel, iterations=1)
    
    # Prepare components image for labeling
    freespace_img = np.full_like(occupancy_map_dilated, fill_value=OCCUPANCY.UNEXPLORED)
    freespace_img[occupancy_map_dilated == OCCUPANCY.FREE] = OCCUPANCY.FREE
    # Make sure agent is on free space
    agent_radius_pixel  = 4
    freespace_img[agent_pos_pixel[0] - agent_radius_pixel : agent_pos_pixel[0] + agent_radius_pixel,
                  agent_pos_pixel[1] - agent_radius_pixel : agent_pos_pixel[1] + agent_radius_pixel,] = OCCUPANCY.FREE

    # Label the components in the image
    components_labels, num = skimage.morphology.label(freespace_img, connectivity=2, background=OCCUPANCY.UNEXPLORED, return_num=True)
    connected_idx = (components_labels == components_labels[agent_pos_pixel[0], agent_pos_pixel[1]])
    
    # Create a structuring element for morphological operations
    selem = skimage.morphology.disk(1)

    # Find indices for different conditions
    free_idx = (occupancy_map_dilated == OCCUPANCY.FREE)
    neighbor_unexp_idx = (skimage.filters.rank.minimum(occupancy_map_dilated, selem) == OCCUPANCY.UNEXPLORED)
    neighbor_occp_idx = (skimage.filters.rank.maximum(occupancy_map_dilated, selem)== OCCUPANCY.OCCUPIED)

    # Get frontier indices and prepare cluster image
    frontier_idx = free_idx & neighbor_unexp_idx & (~neighbor_occp_idx)
    valid_frontier_idx = frontier_idx & connected_idx
    frontier_img = skimage.measure.label(valid_frontier_idx.astype(np.uint8), connectivity=2, return_num=False)
    # ignore 0, which is the background
    cluster_labels = np.unique(frontier_img)[1:]
    
    frontier_centers_pixel = []
    for frontier_label in cluster_labels:
        final_idx_coords = np.where(frontier_img == frontier_label)
        x_np, y_np = final_idx_coords[0], final_idx_coords[1]
        distances = np.sum((np.subtract.outer(x_np, x_np) ** 2 + np.subtract.outer(y_np, y_np) ** 2)** 0.5, axis=1)
        frontier_center_pixel = np.array([x_np[np.argmin(distances)], y_np[np.argmin(distances)]])   
        frontier_centers_pixel.append(frontier_center_pixel)     
        
    # returns an array [num_frontiers, 2] of frontier centers in pixel coordinates and a map with the 0: background, 1: frontier 2: frontier 2, etc.
    return np.vstack(frontier_centers_pixel) if len(frontier_centers_pixel) else np.array([]), frontier_img


def classify_frontiers(slam, frontier_img, frontier_centers_world):
    filled_room_map = TopologyMapping.compute_filled_room_map(slam=slam, dilate_erode_fill=False)
    
    # mapping from frontier_centers_world to classification
    frontier_classification = {}
    
    for i, f in enumerate(np.unique(frontier_img)[1:]):
        if OCCUPANCY.UNEXPLORED in np.unique(filled_room_map[frontier_img == f]):
            frontier_classification[tuple(frontier_centers_world[i])] = FRONTIER_CLASSIFICATION.LEADING_OUT
        else:
            frontier_classification[tuple(frontier_centers_world[i])] = FRONTIER_CLASSIFICATION.WITHIN
    return frontier_classification
    