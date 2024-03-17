# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import copy
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix

import moma_llm.topology
from moma_llm.utils.constants import NODETYPE


def aggregate_close_nodes(graph, voxel_size: float, thres_meter: float):
    nodes_list = list(graph.nodes)
    nodes_stacked = np.stack(nodes_list)
    dist_matrix_meter = distance_matrix(nodes_stacked, nodes_stacked) * voxel_size
    for i, node in enumerate(list(graph.nodes)):
        if i == 0:
            continue
        if dist_matrix_meter[i, :i].min() < thres_meter:
            closest_node = nodes_list[dist_matrix_meter[i, :i].argmin()]
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                graph.add_edge(closest_node, neighbor, dist=graph.get_edge_data(node, neighbor)["dist"])
            graph.remove_node(node)
            

def sparsify_graph(topology_graph: nx.Graph, voxel_size: float, obstacle_map: np.ndarray, sdf_scale: float):
    """
    Sparsify a topology graph by removing nodes with degree 2.
    This algorithm first starts at degree-one nodes (dead ends) and
    removes all degree-two nodes until confluence nodes are found.
    Next, we find close pairs of higher-order degree nodes and
    delete all nodes if the shortest path between two nodes consists
    only of degree-two nodes.
    Args:
        graph (nx.Graph): graph to sparsify
    Returns:
        nx.Graph: sparsified graph
    """
    graph = copy.deepcopy(topology_graph)
    
    if len(graph.nodes) < 10:
        return graph
    # all nodes with degree 1 or 3+
    new_node_candidates = [node for node in list(graph.nodes) if (graph.degree(node) != 2)]
    
    new_graph = nx.Graph()
    for i, node in enumerate(new_node_candidates):
        new_graph.add_node(node)
        
    all_path_dense_graph = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))
    
    sampled_edges_to_add = list()
    new_nodes = set(new_graph.nodes)
    new_nodes_list = list(new_graph.nodes)
    for i in range(len(new_graph.nodes)):
        for j in range(len(new_graph.nodes)):
            if i < j:
                # Go through all edges along path and extract dist
                node1 = new_nodes_list[i]
                node2 = new_nodes_list[j]
                path = all_path_dense_graph[node1][node2]
                for node in path[1:-1]:
                    if graph.degree(node) > 2:
                        break
                else:
                    sampled_edges_to_add.append((path[0], path[-1], np.linalg.norm(np.array(path[0]) - np.array(path[-1]))))
                    dist = [graph.edges[path[k], path[k + 1]]["dist"] for k in range(len(path) - 1)]
                    mov_agg_dist = 0
                    predecessor = path[0]
                    # connect the nodes if there is a path between them that does not go through any other of the new nodes
                    if len(path) and len(set(path[1:-1]).intersection(new_nodes)) == 0:
                        for cand_idx, cand_node in enumerate(path[1:-1]):
                            mov_agg_dist += dist[cand_idx]
                            if mov_agg_dist * voxel_size > 0.6: 
                                sampled_edges_to_add.append((predecessor, cand_node, np.linalg.norm(np.array(predecessor) - np.array(cand_node))))
                                predecessor = cand_node
                                mov_agg_dist = 0
                            else:
                                continue
                        sampled_edges_to_add.append((predecessor, path[-1], np.linalg.norm(np.array(predecessor) - np.array(path[-1]))))
    
    for edge_param in sampled_edges_to_add:
        k, l, dist = edge_param
        if k not in new_graph.nodes:
            new_graph.add_node(k)
        if l not in new_graph.nodes:
            new_graph.add_node(l)
        new_graph.add_edge(k, l, dist=dist)
    return new_graph


def plot_graph(graph, map_underlay, edge_filter=None, ax=None, bounds=None):
    if ax is None:
        ax = plt.gca()
    ax.clear()

    if bounds is None:
        min_x, max_x, min_y, max_y = 0, map_underlay.shape[0], 0, map_underlay.shape[1]
    else:
        min_x, max_x, min_y, max_y = bounds
    ax.set_xlim(min_y, max_y)
    ax.set_ylim(max_x, min_x)
    ax.imshow(map_underlay)
    
    if graph is None or len(graph.nodes()) == 0:
        return
    
    if isinstance(list(graph.nodes())[0], tuple):
        xy = np.array(graph.nodes())
        ax.scatter(xy[:, 1], xy[:, 0], c='r', s=3)
    else:
        xy = []
        node_types = []
        for _, node_attr in graph.nodes(data=True):
            xy.append(node_attr["pos_map"][:2])
            node_types.append(node_attr["node_type"])
        xy = np.stack(xy)
        
        colors = ["c", "k", "m", "r", "g"]
        sizes = [3, 3, 30, 3, 3]
        for nt in np.unique(node_types):
            if nt == NODETYPE.ROOT:
                continue
            ax.scatter(xy[node_types == nt, 1], xy[node_types == nt, 0], c=colors[nt], s=sizes[nt])
        
    for edge in graph.edges():
        if "root" in edge:
            continue
        if isinstance(edge[0], tuple):
            x1, y1 = edge[0]
            x2, y2 = edge[1]
        else:
            x1, y1 = graph.nodes[edge[0]]['pos_map'][:2]
            x2, y2 = graph.nodes[edge[1]]['pos_map'][:2]
        if edge_filter is not None and edge in edge_filter:
            ax.plot([y1, y2], [x1, x2], c='r', linewidth=0.5)
        else:
            ax.plot([y1, y2], [x1, x2], c='b', linewidth=0.5) 
    plt.show()
