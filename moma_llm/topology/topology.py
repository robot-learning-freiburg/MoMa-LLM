# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import copy

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.ndimage.morphology
import scipy.stats
import skfmm
import skimage
from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity

from moma_llm.topology.graph import sparsify_graph, plot_graph
from moma_llm.navigation.navigation import get_circular_kernel
from moma_llm.topology.room_graph import get_body_properties
from moma_llm.utils.constants import CLASS_NAME_TO_CLASS_ID, OCCUPANCY


def boundary_integral(sdf, x1, y1, x2, y2, normalize):
    # Interpolate N times between the two points
    N_interp = int(np.linalg.norm(np.array([x1-x2, y1-y2])))
    integral = 0
    if N_interp > 0:
        for j in range(N_interp):
            # Interpolate between the two points
            x = int((x1 + (float(j) / N_interp) * (x2 - x1)))
            y = int((y1 + (float(j) / N_interp) * (y2 - y1)))

            # Check if the point is inside the mask
            if sdf[x, y] > 0:
                integral += sdf[x,y]
    return integral/(N_interp+1e8) if normalize else integral

    
def compute_sdf(boundary_mask, distance_scale=1):
    dx = 1 # downsample factor
    f = distance_scale / dx  # distance function scale

    # cv2.resize(boundary_map, None, fx=1/dx, fy=1/dx, interpolation=cv2.INTER_NEAREST)
    sdf = skfmm.distance(1 - boundary_mask)
    sdf[sdf > f] = f
    sdf = sdf / f
    sdf = 1 - sdf
    return sdf


class TopologyMapping:
    def __init__(self, size, voxel_size):
        self.size = size
        self.voxel_size = voxel_size

    @staticmethod
    def update_maps(slam):
        zero_height_pixel = slam.midpoint
        height_cutoff = int(1.75 / slam.voxel_size)
        agg_wall_map = np.isin(slam.voxel_map[:, :, zero_height_pixel:zero_height_pixel+height_cutoff], [CLASS_NAME_TO_CLASS_ID["walls"], 
                                                                                                         CLASS_NAME_TO_CLASS_ID["door"],
                                                                                                         CLASS_NAME_TO_CLASS_ID["window"]]).any(axis=2)
        filled_height_cutoff = int(1.2 / slam.voxel_size)
        agg_wall_map = np.logical_or(agg_wall_map, 
                                     (slam.voxel_map[:, :, zero_height_pixel:zero_height_pixel + height_cutoff] > 0).sum(2) > filled_height_cutoff).astype(np.float32)
        return agg_wall_map
    
    def compute_convex_hull(self, slam):
        self.occupied_map = slam.bev_map_semantic / max(OCCUPANCY)
        dilatation_size = 3
        dilation_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        self.occupied_map = cv2.dilate(self.occupied_map, element).astype(np.uint8)

        contours, _ = cv2.findContours(self.occupied_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest hull
        max_area_contours = [sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]]
        max_area_hull = [cv2.convexHull(max_area_contours[0], False)]

        drawing = np.zeros_like(self.occupied_map, dtype=np.uint8)
        # draw contours and hull points
        for i in range(len(max_area_contours)):
            color = (255, 0, 0) # blue - color for convex hull
            cv2.drawContours(drawing, max_area_hull, i, color, 1, 8)

        self.occupied_map[self.occupied_map > 0.0] = 255.0
        combined = np.concatenate((drawing,
                                    self.occupied_map,
                                    ), axis=1)
        cv2.imshow("HullMapping", combined)
        cv2.waitKey(1)

        self.max_area_hull = max_area_hull
        return max_area_hull
    
    @staticmethod
    def compute_filled_room_map(slam, dilate_erode_fill: bool):
        occupied_map = slam.bev_map_occupancy > 0
        occupied_map[(slam.voxel_map == CLASS_NAME_TO_CLASS_ID["ceilings"]).any(axis=2)] = True
        if dilate_erode_fill:
            kernel = get_circular_kernel(1)
            occupied_map = cv2.erode(cv2.dilate(occupied_map.astype(np.uint8), kernel, iterations=1), kernel, iterations=1)
        free_map = scipy.ndimage.binary_fill_holes(occupied_map.astype(bool)).astype(int)
        return free_map
    
    def compute_voronoi_graph(self, slam, wall_map, sdf_scale=3.0):
        self.obstacle_map = wall_map
        self.sdf_scale = sdf_scale
        self.places = nx.Graph()
        
        free_map = (slam.bev_map_occupancy != OCCUPANCY.UNEXPLORED).astype(int)
        occupancy_map = (slam.bev_map_occupancy == OCCUPANCY.OCCUPIED).astype(int)
        bev_map_occupancy = slam.bev_map_occupancy
        
        # add additional walls at all borders of explored to unexplored space in the filled-out map
        # Create a structuring element for morphological operations
        selem = skimage.morphology.disk(1)
        # Find indices for different conditions
        neighbor_unexp_idx = (skimage.filters.rank.minimum(free_map.astype(np.uint8), selem) == 0)
        neighbor_occp_idx = (skimage.filters.rank.maximum(free_map.astype(np.uint8), selem)== 1)
        frontier_idx = neighbor_unexp_idx & neighbor_occp_idx        
        wall_hull_map = np.maximum(occupancy_map, frontier_idx)
        
        # directly compute voronoi in sdf or inflated map -> don't have to do resolve_boundary_violations() later on
        # inflation outwards of the apartment
        boundary_sdf = compute_sdf(wall_hull_map, distance_scale=sdf_scale)
        boundary_sdf[slam.bev_map_occupancy == OCCUPANCY.FREE] = 0
        # still add small inflation within the apartment, o/w we end up with edges over 1-pixel-wide walls
        boundary_sdf = np.maximum(boundary_sdf, compute_sdf(wall_hull_map, distance_scale=1.1))

        obstacle_points = np.asarray(np.where(boundary_sdf)).T.astype(np.float32)
        # obstacle_points += np.random.normal(0, 0.05, obstacle_points.shape)
        vor = Voronoi(obstacle_points) # unfiltered voronoi graph
        
        # exclude all nodes that lie on obstacles or outside the observed rooms
        clipped_vertices = vor.vertices[np.all(vor.vertices >= 0, axis=1)]
        clipped_vertices = clipped_vertices[np.all(clipped_vertices <= max(np.array(bev_map_occupancy.shape) - 1), axis=1)]
            
        def _ceil(vertices, mask, idx):
            return np.ceil(vertices[:, idx]).astype(int)
                
        def _floor(vertices, mask, idx):
            return np.floor(vertices[:, idx]).astype(int)
                
        mask = (boundary_sdf + (bev_map_occupancy == 0)).astype(bool)
        idx1 = mask[_floor(clipped_vertices, mask, 0), _floor(clipped_vertices, mask, 1)] == 0
        idx2 = mask[_floor(clipped_vertices, mask, 0), _ceil(clipped_vertices, mask, 1)] == 0
        idx3 = mask[_ceil(clipped_vertices, mask, 0), _floor(clipped_vertices, mask, 1)] == 0
        idx4 = mask[_ceil(clipped_vertices, mask, 0), _ceil(clipped_vertices, mask, 1)] == 0
        valid = ((idx1.astype(int) + idx2.astype(int) + idx3.astype(int) + idx4.astype(int)) > 3)
        valid_vor_nodes = clipped_vertices[valid]

        # Add unfiltered nodes and edges
        for vertex in valid_vor_nodes:
            self.places.add_node(tuple(np.round(vertex, 3)))
        
        ridge_vertices_array = np.array(vor.ridge_vertices)
        simplex_mask = np.all(ridge_vertices_array >= 0, axis=1)
        for simplex in ridge_vertices_array[simplex_mask]:
            v1, v2 = vor.vertices[simplex]
            v1, v2 = tuple(np.round(v1, 3)), tuple(np.round(v2, 3))
            if (v1 in self.places.nodes) and (v2 in self.places.nodes):
                self.places.add_edge(tuple(v1), tuple(v2), dist=np.linalg.norm(np.array(v1) - v2))
        
        nodes_stacked = np.stack(self.places.nodes())
        outside_map = np.logical_or(nodes_stacked < 0, nodes_stacked >= np.array(wall_hull_map.shape)).any(axis=1)
        
        nodes_outside_free_space = list()
        for i, node in enumerate(self.places.nodes()):
            if outside_map[i] or wall_hull_map[int(node[0]), int(node[1])]:
                nodes_outside_free_space.append(node)
        for node in nodes_outside_free_space:
            self.places.remove_node(node)
        
        if len(self.places.nodes) == 0:
            print("No nodes in graph, returning empty graph")
        else:
            # Obtain the largest connected component and make that the only places graph
            comp_places_subgraphs = [self.places.subgraph(c).copy() for c in sorted(nx.connected_components(self.places), key=len, reverse=True)]
            self.places = comp_places_subgraphs[0]

        return self.places
    
    def sparsify_topology_graph(self):
        return sparsify_graph(self.places, self.voxel_size, self.obstacle_map, self.sdf_scale)

    def plot_graph(self, map_underlay):
        return plot_graph(self.places, map_underlay)


def detect_rooms(sim, slam, graph, obstacle_map, sdf_scale, thresh, voxel_size, obj_to_neglect, opened_windows):
    graph = copy.deepcopy(graph)
    sdf_edges_tbd = list()
    boundary_sdf = compute_sdf(obstacle_map, distance_scale=sdf_scale)

    # manifold holding the door probability estimate
    xmin, xmax = 0, slam.bev_map_occupancy.shape[0]
    ymin, ymax = 0, slam.bev_map_occupancy.shape[1]
    xx, yy = np.mgrid[xmin:xmax:complex(0, xmax), ymin:ymax:complex(0, ymax)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    door_positions = list()
    # find all the doors and implement a probability density around them
    body_ids = set(sim.renderer.get_pb_ids_for_instance_ids(np.array(list(slam.seen_instances)))) - {-1, 0}
    for bid in body_ids:
        body_property = get_body_properties(sim, bid, obj_to_neglect, opened_windows)
        if body_property is not None:
            if body_property["semantic_class_name"] == "door":
                door_positions.append(slam.world2voxel(np.array(body_property["pos"][:2])))
    
    if len(door_positions) > 0:
        # kernel density estimation
        door_pos = np.array(door_positions).reshape(-1,2)
        # ax.scatter(door_pos[1], door_pos[0], c='b', s=50, zorder=2)
        kde = KernelDensity(kernel='gaussian', bandwidth=2.0)
        kde.fit(door_pos)
        door_prob = np.exp(kde.score_samples(positions.T).reshape(xx.shape))

        edges_tbd = list()
        for edge in list(graph.edges()):
            # obtain x1, y1, x2, y2 coordinates of edge
            x1, y1 = edge[0]
            x2, y2 = edge[1]
            edge_score = boundary_integral(door_prob, x1, y1, x2, y2, normalize=False)
            if edge_score > thresh:
                edges_tbd.append(edge)
        
        graph.remove_edges_from(edges_tbd)
        graph.remove_nodes_from(list(nx.isolates(graph)))

        del_prob = door_prob
    else:
        door_pos = np.empty((0,2))
        del_prob = boundary_sdf


    components = list(nx.connected_components(graph))
    # filter out small components as soon as we have more than one
    if len(components) > 0 and len(graph.nodes) > 4:
        for c in components:
            # filter out components close to boundaries that do not span a significant area
            # but are just an artifact of the pixelized boundary map
            if len(c) < 10:
                edges = list(graph.subgraph(c).edges())
                cum_edge_len = sum([graph.get_edge_data(e[0], e[1])["dist"] for e in edges])*voxel_size
                if cum_edge_len < 0.5:
                    graph.remove_nodes_from(c)

    return graph, del_prob, door_pos





