# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import networkx as nx
import numpy as np
from igibson import object_states
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

from moma_llm.utils.constants import CLASS_NAME_TO_CLASS_ID, NODETYPE


def get_body_properties(simulator, bid, obj_to_neglect: list, opened_windows):
    obj = simulator.scene.objects_by_id.get(bid, None)
    if obj is None:
        print(f"body_id {bid} not found by simulator")
        return None
    elif obj.name in ["floors", "walls", "ceilings"]:
        return None
    elif obj.name in obj_to_neglect: 
        return None
    # Use position/orientaiton of aligned bounding box
    pos, orn, bbox_extent, _ = obj.get_base_aligned_bounding_box(fallback_to_aabb=True)
    states = {}
    for state_type in [object_states.Open]:
        if obj.states.get(state_type) is not None:
            if (state_type == object_states.Open) and (obj.category == "window"):
                states[state_type] = (obj.name in opened_windows)
            else:
                states[state_type] = obj.states.get(state_type).get_value()    
    
    properties = {"bbox": bbox_extent.astype(np.float32),
                  "semantic_class_name": obj.category,
                  "pos": tuple(pos),
                  "orn": orn,
                  "name": obj.name,
                  "states": states,
                  }
    return properties


def get_seen_object_nodes(simulator, slam, obj_to_neglect, opened_windows) -> dict:
    seen_instance_list = list(slam.seen_instances)
    pb_ids = simulator.renderer.get_pb_ids_for_instance_ids(np.array(seen_instance_list))
    pb_to_instance_id_mapping = dict(zip(pb_ids, seen_instance_list))
    body_ids = set(pb_ids) - {-1, 0}

    body_properties = {}
    
    for bid in body_ids:
        body_property = get_body_properties(simulator, bid, obj_to_neglect, opened_windows)
        if body_property is not None:
            body_properties[tuple(body_property["pos"])] = body_property
            body_property["instance_id"] = pb_to_instance_id_mapping[bid]

    return body_properties


def get_closest_node(query_coords_world: np.ndarray, graph, slam, dist_thresh=None):
    graph_node_pos = np.stack(graph.nodes)
    node_coords_world = slam.voxel2world(graph_node_pos)
    # [num_objects, num_nodes]
    dist_matrix = distance_matrix(query_coords_world, node_coords_world)
    if dist_thresh is not None:
        # return all nodes closer than the thresh
        # list of len objects, each entry is a list of nodes that are closer than the thresh
        closest_nodes, closest_nodes_dists = [], []
        for i in range(len(query_coords_world)):
            idx  = dist_matrix[i] < dist_thresh
            closest_nodes.append(graph_node_pos[idx])
            closest_nodes_dists.append(dist_matrix[i][idx])
        return closest_nodes, closest_nodes_dists
    else:
        # return the closest node
        closest_node_idx = np.argmin(dist_matrix, axis=1)
        closest_nodes_dist = np.min(dist_matrix, axis=1)
        closest_nodes = graph_node_pos[closest_node_idx]    
        return closest_nodes, closest_nodes_dist
    

def map_open_doors_to_components(simulator, slam, separated_vor_graph, opened_doors) -> dict:
    """
    For each open door, find the two closest components in the separated voronoi graph
    """
    if len(opened_doors) == 0:
        return {}
    # retrieve all opened doors
    open_door_pos = dict()

    for door in opened_doors:
        obj = simulator.scene.objects_by_name[door]
        pos, orn, bbox_extent, _ = obj.get_base_aligned_bounding_box(fallback_to_aabb=True)
        open_door_pos[obj.name] = slam.world2voxel(pos)[0:2]
    open_door_pos = np.stack(list(open_door_pos.values()))
    
    # compute min distance between each component and the open doors
    c_min_dists = dict()
    components = list(nx.connected_components(separated_vor_graph))
    for c_id, c_nodes in enumerate(components):
        c_pos = np.array(list(c_nodes))
        
        c_door_dists = cdist(open_door_pos, c_pos, metric="euclidean")
        c_min_dist = np.min(c_door_dists, axis=1)
        c_min_dists[c_id] = c_min_dist

    # get the two closest components to each open door
    c_min_dists = np.array(list(c_min_dists.values()))
    two_closest_c = np.argsort(c_min_dists, axis=0)[0:2].T
    two_closest_doors = dict()
    for d, c in zip(opened_doors, two_closest_c):
        two_closest_doors[d] = list(c)
    return two_closest_doors


def create_room_object_graph(simulator, slam, vor_graph, separated_vor_graph, obj_to_neglect, opened_doors, opened_windows, use_viewpoint_assignment):
    # create main graph with level1: room, level2: objects
    room_graph = nx.Graph()
    room_object_graph = nx.DiGraph()
    
    room_object_graph.add_node("root", node_type=NODETYPE.ROOT, pos_map=(0, 0, 0))
    components = list(nx.connected_components(separated_vor_graph))
    for c_id, c_nodes in enumerate(components):
        # add room id's to room_graph
        for node in c_nodes:
            separated_vor_graph.nodes[node]['room_id'] = c_id
        # add room nodes to new graph
        room_subgraph = separated_vor_graph.subgraph(c_nodes).copy()
        room_center_node = nx.center(room_subgraph)[0]
        room_graph.add_node(NODETYPE.roomname(c_id),
                            pos=tuple(slam.voxel2world(np.array(room_center_node))),
                            pos_map=tuple(room_center_node),
                            room_id=c_id,
                            node_type=NODETYPE.ROOM)
        room_object_graph.add_node(NODETYPE.roomname(c_id), 
                                   pos=tuple(slam.voxel2world(np.array(room_center_node))), 
                                   pos_map=tuple(room_center_node),
                                   room_id=c_id, 
                                   node_type=NODETYPE.ROOM, 
                                   frontier_points=set(),
                                   closed_doors=set(),
                                   open_doors=set())
        room_object_graph.add_edge("root", NODETYPE.roomname(c_id))

    
    # Construct neighborhood-respective room graph (not interconnected)
    for c_id, c_node in enumerate(room_graph.nodes):
        for c_id2, c_node2 in enumerate(room_graph.nodes):
            if c_id < c_id2:
                path = nx.shortest_path(vor_graph, source=tuple(room_graph.nodes[c_node]["pos_map"]), target=tuple(room_graph.nodes[c_node2]["pos_map"]))
                rooms_traveled = set()
                for node in path:
                    if node in separated_vor_graph.nodes:
                        rooms_traveled.add(separated_vor_graph.nodes[node]["room_id"])
                if len(rooms_traveled) < 3:
                    room_graph.add_edge(NODETYPE.roomname(c_id), NODETYPE.roomname(c_id2))

    object_room_assigment(simulator=simulator, 
                          slam=slam,
                          vor_graph=vor_graph,
                          separated_vor_graph=separated_vor_graph,
                          room_object_graph=room_object_graph,
                          obj_to_neglect=obj_to_neglect,
                          opened_doors=opened_doors,
                          opened_windows=opened_windows,
                          use_viewpoint_assignment=use_viewpoint_assignment)

    return room_graph, room_object_graph


def object_room_assigment(simulator, slam, vor_graph, separated_vor_graph, room_object_graph, obj_to_neglect, opened_doors, opened_windows, use_viewpoint_assignment: bool):
    # get all seen objects based on slam map
    object_nodes = get_seen_object_nodes(simulator, slam, obj_to_neglect, opened_windows)
    if len(object_nodes) == 0:
        return
    
    door2comp = map_open_doors_to_components(simulator, slam, separated_vor_graph, opened_doors)
    
    # 1) find node closest to the closest point from which we've seen the object in the separated voronoi graph
    # 2) add object to all sep voronoi nodes within object_closeness_thresh
    # 3) assign room label of the node that has the shortest path to the viewpoint node + euclidean distance from object to its closest node
    object_closeness_thresh = 2.5
    vp_closeness_thresh = 5.0
    
    viewpoint_coords_world = np.stack([slam.instance_viewpoints[on["instance_id"]][0] for on in object_nodes.values()])[..., :2]
    vp_closer_than_thresh, vp_closest_nodes_dists = get_closest_node(viewpoint_coords_world, graph=separated_vor_graph, slam=slam, dist_thresh=vp_closeness_thresh)

    object_coords_world = np.stack(list(object_nodes.keys()))[..., :2]
    closest_nodes, closest_nodes_dists = get_closest_node(object_coords_world, graph=separated_vor_graph, slam=slam, dist_thresh=object_closeness_thresh)
    # NOTE: these dists are in pixel space as the node coords are in pixel space
    all_path_sep_voronoi = dict(nx.all_pairs_dijkstra_path_length(vor_graph, weight="dist"))

    # add all objects as children of their room
    for i, node_prop in enumerate(object_nodes.values()):
        nodes_closer_than_thresh = closest_nodes[i]
        nodes_closer_than_thresh_dists = closest_nodes_dists[i]
        if len(nodes_closer_than_thresh) == 0:
            print(f"Object {node_prop['name']} too far from sparsified room graph - not assigned to any room. Dist: {get_closest_node(object_coords_world, graph=separated_vor_graph, slam=slam)[1][i]}")
            continue
        
        if use_viewpoint_assignment:
            # d / slam.voxel_size to convert from distance in world space to distance in pixel space
            # **1.3 to weight it a bit more to prefer taking nodes that are very close to the object
            dists_to_vp = []
            for n, nd in zip(nodes_closer_than_thresh, nodes_closer_than_thresh_dists):
                aaa = []
                for closest_vp, closest_vp_d in zip(vp_closer_than_thresh[i], vp_closest_nodes_dists[i]):
                    ddd = all_path_sep_voronoi.get(tuple(closest_vp), np.inf).get(tuple(n), np.inf) + (nd / slam.voxel_size)**1.3 + (closest_vp_d / slam.voxel_size)
                    aaa.append(ddd)
                dists_to_vp.append(np.min(aaa))
            
            if np.min(dists_to_vp) == np.inf:
                print(f"Object {node_prop['name']} not reachable from any viewpoint")
                continue
            node_closest_to_viewpoint_idx = np.argmin(dists_to_vp)
            node_closest_to_viewpoint = nodes_closer_than_thresh[node_closest_to_viewpoint_idx]
        else:
            node_closest_to_viewpoint = nodes_closer_than_thresh[np.argmin(nodes_closer_than_thresh_dists)]

        node_prop["room_id"] = separated_vor_graph.nodes.get(tuple(node_closest_to_viewpoint))["room_id"]
        node_prop["pos_map"] = tuple(slam.world2voxel(np.array(node_prop["pos"])))
        # for 'normal' objects, we take the closest voronoi node as determined by closest-viewpoint logic. For doors, just take the overall closest voronoi node, as they are connecting multiple rooms
        node_prop["closest_vor_node"] = tuple(node_closest_to_viewpoint)
        
        if (node_prop["semantic_class_name"] == "door"):
            if node_prop["name"] not in opened_doors:
                node_prop["closest_vor_node"] = tuple(node_closest_to_viewpoint)
                room_object_graph.nodes.get(NODETYPE.roomname(node_prop["room_id"]))["closed_doors"].add(node_prop["name"])
            else: 
                comps = door2comp[node_prop["name"]]
                # add as tuple(door, room it connects to - None if unknown which room it connects to    )
                if len(comps) == 1:
                    room_object_graph.nodes[NODETYPE.roomname(comps[0])]["open_doors"].add((node_prop["name"], None))
                elif len(comps) == 2:
                    room_object_graph.nodes[NODETYPE.roomname(comps[0])]["open_doors"].add((node_prop["name"], NODETYPE.roomname(comps[1])))
                    room_object_graph.nodes[NODETYPE.roomname(comps[1])]["open_doors"].add((node_prop["name"], NODETYPE.roomname(comps[0])))
                else:
                    raise ValueError(f"Door {node_prop['name']} is connected to more than 2 components {comps}")
                # NOTE: do not add open doors to the graph at all, as they are irrelevant for current search tasks
                # TODO: should we add them to the graph as well? Would need to modify node_prop["room_id"] and add edges to both rooms that they belong to
                continue
            
        room_object_graph.add_node(node_prop["name"], **node_prop, node_type=NODETYPE.OBJECT)
        room_object_graph.add_edge(NODETYPE.roomname(node_prop["room_id"]), node_prop["name"])
         