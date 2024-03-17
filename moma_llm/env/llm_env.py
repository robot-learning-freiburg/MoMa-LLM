# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple

import networkx as nx
import numpy as np
from igibson import object_states
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import distance_matrix

from moma_llm.env.env import OurIGibsonEnv
from moma_llm.env.high_level_env import HighLevelEnv
from moma_llm.llm.llm import LLM, Conversation, inflect_engine
from moma_llm.topology.room_graph import get_closest_node
from moma_llm.utils.constants import FRONTIER_CLASSIFICATION, NODETYPE

DIST_MAPPING = OrderedDict({
    3.0: "very close",
    10.0: "near",
    20.0: "far",
    np.inf: "distant"})


def distance_mapping(dist: float):
    for k, v in DIST_MAPPING.items():
        if dist < k:
            return v
    
    
def split_frontier_points(frontier_points):
    frontier_points_within, frontier_points_leading_out = [], []
    for frontier_point in frontier_points:
        if frontier_point[1] == FRONTIER_CLASSIFICATION.WITHIN:
            frontier_points_within.append(frontier_point[0])
        elif frontier_point[1] == FRONTIER_CLASSIFICATION.LEADING_OUT:
            frontier_points_leading_out.append(frontier_point[0])
        else:
            raise ValueError(f"Unknown frontier point type {frontier_point[1]}")
    return frontier_points_within, frontier_points_leading_out

@dataclass
class ActionHistory:
    action: str
    object_name_graph: str
    position: tuple
    subtask_success: bool
    opendoors_roompos: Any = None
    orig_api_call: str = None
    

class LLMEnv(HighLevelEnv):
    def __init__(self, env: OurIGibsonEnv, llm: LLM, seed: int) -> None:
        super().__init__(env, seed=seed)

        # action: (argument, description)
        self.possible_actions = {
            "navigate": ("room_name, object_name", "navigate to this object in this room."),
            "go_to_and_open": ("room_name, object_name", "go to this articulated object, door or container and open it."),
            "close": ("room_name, object_name", "close this articulated object, door or container."),
            "explore": ("room_name", "explore the unknown space near one of the rooms that is not fully explored yet."),
            "done": ("", "call when the task is completed or if you are unable to take any further actions.")
        }
        if not self.env.config["consider_open_actions"]:
            del self.possible_actions["go_to_and_open"]
            del self.possible_actions["close"]
        
        self.llm = llm
        self.room_classification = dict()

    def reset(self, config_file, scene_id, episode_num):
        self.prev_responses = [""]
        self.action_history = []
        self.last_env_feedback = {"role": "env", "content": ""}
        return super().reset(config_file, scene_id, episode_num, compute_scene_graph=True)

    def classify_rooms(self, obs):
        room_classification: dict = self.llm.classify_rooms(obs)
        # nx can't handle duplicate node names, so add a number to the end if there are multiple instances of a room type
        tot_counter = Counter(room_classification.values())
        counter = defaultdict(int)
        for k, v in room_classification.items():
            if tot_counter[v] > 1:
                room_classification[k] = v + f"-{counter[v] + 1}"
                counter[v] += 1
        self.room_classification = room_classification
        self._label_rooms_on_map(obs)

    def evaluate_room_labeling(self, obs):
        separated_components = [obs['separated_voronoi_graph'].subgraph(c).copy() for c in nx.connected_components(obs['separated_voronoi_graph'])]
        components = {}
        c2room_gt_sem = defaultdict(list)
        c2room_pred_sem = defaultdict(list)
        for pred_c_idx, c in enumerate(separated_components):
            components[pred_c_idx] = c
            for n in c.nodes:
                world_coords = self.unwrapped.slam.voxel2world(n)
                seg_map_coords = self.unwrapped.scene.world_to_seg_map(world_coords)
                room_name = self.unwrapped.scene.get_room_type_by_point(world_coords)
                c2room_gt_sem[pred_c_idx].append(room_name) if room_name is not None else "unknown"
                c2room_pred_sem[pred_c_idx].append(self.room_classification[NODETYPE.roomname(c.nodes[n]["room_id"])])

        # go through all rooms and measure the semantic accuracy between the ground truth and the predicted room
        pred, gt = [], []
        for c_idx, c in components.items():
            pred.extend(c2room_pred_sem[c_idx])
            gt.extend(c2room_gt_sem[c_idx])
        semantic_accuracy = sum(1 for x,y in zip(pred, gt) if x == y) / len(gt)
        self.episode_room_sem_acc.append(semantic_accuracy)
        print(semantic_accuracy)

    def _label_rooms_on_map(self, obs, ax_idx=1):
        # delete all existing text
        for txt in self.env.ax[ax_idx].texts:
            Artist.remove(txt)

        # put labels on the map
        labeled_rooms_not_found = list()
        for room, labelled_room in self.room_classification.items():
            if room in obs["room_object_graph"].nodes:
                room_node = obs["room_object_graph"].nodes[room]
                room_center = room_node["pos_map"]
                self.env.ax[ax_idx].text(room_center[1], room_center[0], labelled_room, fontsize=8, color="m", horizontalalignment='center')
            else:
                labeled_rooms_not_found.append((room, labelled_room))
        if len(labeled_rooms_not_found) > 0:
            print("Rooms that were labeled but not contained in the room object graph: ", labeled_rooms_not_found)

    def parse_llm_action(self, response: str):
        action = None
        command_found = False
        for line in response.split("\n"):
            # Go through answers until we find a "command", then answer may be on the same line or the next. 
            # Just go line by line until there is something in the form of an api call
            command_found = command_found or line.lower().startswith("command")
            if not command_found:
                continue
            try:
                action, argument = re.search(r"(\w+)\((.*)\)", line).groups()
                argument = argument.replace("'", "").replace('"', "")
                if not action:
                    print("Empty action parsed:", (action, line))
                break
            except:
                continue
            
        if not action:
            # TODO: better to re-prompt the LLM instead?
            print("Could not parse any command. Assuming this is because the LLM did not find a reasonable action. So calling done() instead.")
            action = "done"
            argument = ""
            self.episode_info["failure_reason"] = "Unable to prompt LLM command."
            
        if self.possible_actions[action][0]:
            assert argument, (argument, response)
        return action, argument

    def _create_prompt(self, 
                       task_description, 
                       labelled_rooms, 
                       current_room, 
                       room_dict, 
                       rooms_with_frontier_within, 
                       rooms_with_frontier_leading_out, 
                       rooms_with_closed_doors, 
                       close_objects, 
                       nlp_history,
                       room_distances,
                       *args, 
                       **kwargs) -> Conversation:
        system_prompt = f"You are a robot in an unexplored house. Your task is to {task_description}."
        system_prompt += f" You have the following actions available that you can use to achieve this task:\n"
        for i, (action, description) in enumerate(self.possible_actions.items()):
            system_prompt += f"""{i+1}. {action}({description[0]}): {description[1]}\n"""
        system_prompt += f"\nOutput Response Format:\n"\
        "Analysis: describe where you could find the objects of interest and what actions you need to execute to get there.\n"\
        "Reasoning: justify why the next action is important to solve the task.\n"\
        "Command: function call"

        prompt = f"You are currently in the {current_room}. You are standing next to the following objects: [{', '.join(sorted(close_objects))}]."
        prompt += f" Furthermore, you have found the following rooms and objects in the house so far:\n"
        for room in sorted(labelled_rooms):
            objects = room_dict[room] + (["unexplored area"] if room in [r[0] for r in rooms_with_frontier_within] else [])
            prompt += f"""- {room}: [{", ".join(objects)}].\n"""

        if len(nlp_history):
            prompt += f"Your {len(nlp_history)} previous actions were: {', '.join(nlp_history)}.\n"

        rooms_with_frontier_descr = [f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_frontier_leading_out]
        prompt += f"These rooms have unexplored space leading out of the room: [{', '.join(rooms_with_frontier_descr)}].\n"
        if len(rooms_with_closed_doors):
            rooms_with_closed_doors_descr = [f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_closed_doors]
            prompt += f"These rooms contain closed doors that might open up new space: [{', '.join(rooms_with_closed_doors_descr)}].\n"

        prompt += f"""What is the best next action to complete the task as efficiently as possible? I you don't think that the object can be found in a known room, prioritize opening doors over exploring a room.\n"""  # In general, prioritize opening doors if there is not enough evidence to explore a promising room.\n""" 
        prompt += f"Remember:\n"\
            "1. Respond with a function call.\n"\
            "2. You can only use the objects and rooms that you have already found. Object names have to match the description exactly.\n"\
            "3. You can only explore rooms that are listed as having unexplored space.\n"\
            "4. If you have found the object you are looking for, directly call done(). You do not need to navigate to it or interact with it.\n"\
            "5. If some actions failed repeatedly, they may not be possible.\n"

        conversation = Conversation(messages=[self.last_env_feedback,
                                              {"role": "system", "content": system_prompt},
                                              {"role": "user", "content": prompt}])
        return conversation

    def _get_close_objects(self, graph, current_room, closeness_thresh: float):
        current_room_objects = list(graph.successors(current_room))
        if not current_room_objects:
            return []
        graph_node_pos = np.stack([graph.nodes.get(n)["pos_map"][:2] for n in current_room_objects])
        node_coords_world = self.unwrapped.slam.voxel2world(graph_node_pos)
        # [1, num_objects]
        dist_matrix = distance_matrix(self.unwrapped.robots[0].get_position()[np.newaxis, :2], node_coords_world)
        close_objects = np.array(current_room_objects)[np.squeeze(dist_matrix < closeness_thresh, 0)]
        return set([self.llm.to_human_readable_object_name(o) for o in close_objects])

    def _match_action_history(self, graph, separated_voronoi_graph):
        max_history_length = 5
        
        def _match_room_pos(position):
            if position is None:
                return None
            closest_nodes, _ = get_closest_node(np.array([position]), separated_voronoi_graph, self.slam)
            room_name = separated_voronoi_graph.nodes[tuple(closest_nodes[0])]["room_id"]
            return room_name
        
        nlp_history = []
        last_k = len(self.action_history) - max_history_length
        
        for i, h in enumerate(self.action_history):
            room_name, object_name, invalid = None, None, False
            
            if h.object_name_graph is not None:
                object_name = self.llm.to_human_readable_object_name(h.object_name_graph)
                if (h.opendoors_roompos is not None) and (h.object_name_graph in self.env.opened_doors):
                    room_name = _match_room_pos(h.opendoors_roompos)
                elif h.object_name_graph in graph.nodes:
                    room_name = list(graph.predecessors(h.object_name_graph))[0]
                else:
                    room_name = _match_room_pos(h.position)
            elif h.position is not None:
                room_name = _match_room_pos(h.position)
            
            nlp = None
            if h.action == "done":
                nlp = "done()"
            elif (h.action == "explore") and (room_name is not None): 
                nlp = f"explore({room_name})"
            elif (object_name is not None) and (room_name is not None):
                nlp = f"{h.action}({room_name}, {object_name})"
            else:
                nlp = f"{h.orig_api_call} - invalid argument"
                invalid = True
            
            if not invalid:
                nlp += f" - {'success' if h.subtask_success else 'failure'}"
            
            # keep history of last k actions & all invalid calls (e.g. trying to open an object that can't be opened)
            if i >= last_k or invalid:
                nlp_history.append(nlp)
        return nlp_history
    
    def send_query(self, conversation: str):
        response = self.llm.send_query(conversation=conversation)
        action, argument = self.parse_llm_action(response)
        return response, action, argument

    def take_action(self, obs: dict, task_description: str):
        def _apply_room_classification(obs):
            obs["room_object_graph"] = nx.relabel_nodes(obs["room_object_graph"], self.room_classification)
            for n, d in obs["separated_voronoi_graph"].nodes(data=True):
                d["room_id"] = self.room_classification[NODETYPE.roomname(d["room_id"])]
        self.classify_rooms(obs)
        _apply_room_classification(obs)
        
        graph = obs["room_object_graph"]
        labelled_rooms = list(graph.successors("root"))

        room_dict = self.llm.create_room_object_dict(graph,
                                                     open_door_inclusion="ignore",
                                                     room_classification=self.room_classification)
        current_room = self.room_classification[obs["robot_current_room"]]

        def _get_closest_dist(points):
            closest_idx, _closest_frontier, _costs, paths = self._find_closest_point(points)
            return self.env.slam.voxel_size * len(paths[closest_idx])

        rooms_with_frontier_within = []
        rooms_with_frontier_leading_out = []
        rooms_with_closed_doors = []
        for n in labelled_rooms:
            frontier_points = graph.nodes.get(n)["frontier_points"]
            if len(frontier_points):
                frontier_points = graph.nodes[n].get("frontier_points", {})
                frontier_points_within, frontier_points_leading_out = split_frontier_points(frontier_points)
                if frontier_points_within:
                    rooms_with_frontier_within.append((n, _get_closest_dist(frontier_points_within)))
                if frontier_points_leading_out:
                    rooms_with_frontier_leading_out.append((n, _get_closest_dist(frontier_points_leading_out)))
            closed_doors = graph.nodes.get(n)["closed_doors"]
            if len(closed_doors):
                rooms_with_closed_doors.append((n, _get_closest_dist([graph.nodes.get(n) for n in closed_doors])))
        rooms_with_frontier_within = sorted(rooms_with_frontier_within, key=lambda x: x[1])
        rooms_with_frontier_leading_out = sorted(rooms_with_frontier_leading_out, key=lambda x: x[1])
        rooms_with_closed_doors = sorted(rooms_with_closed_doors, key=lambda x: x[1])

        close_objects = self._get_close_objects(graph=graph, current_room=current_room, closeness_thresh=2.5)
        nlp_history = self._match_action_history(graph=graph, separated_voronoi_graph=obs["separated_voronoi_graph"])
        
        def _calc_dist_to_room(current_room, separated_voronoi_graph) -> dict:
            # find distance from robot to closest voronoi node with that room label
            vnodes = defaultdict(list)
            for node_pos, node_data in separated_voronoi_graph.nodes(data=True):
                vnodes[node_data["room_id"]].append(node_pos)
            
            room_distances = {}
            for room in self.room_classification.values():
                if room == current_room:
                    room_distances[room] = "current location"
                else:
                    _idx, _closest_node, _costs, paths = self._find_closest_point([self.env.slam.voxel2world(pos) for pos in vnodes[room]])
                    dist = self.env.slam.voxel_size * len(paths)
                    room_distances[room] = distance_mapping(dist)
            return room_distances
        room_distances = _calc_dist_to_room(current_room, obs["separated_voronoi_graph"])

        conversation = self._create_prompt(task_description=task_description,
                                           labelled_rooms=labelled_rooms,
                                           current_room=current_room,
                                           room_dict=room_dict,
                                           rooms_with_frontier_within=rooms_with_frontier_within,
                                           rooms_with_frontier_leading_out=rooms_with_frontier_leading_out,
                                           rooms_with_closed_doors=rooms_with_closed_doors,
                                           close_objects=close_objects,
                                           nlp_history=nlp_history,
                                           graph=graph,
                                           room_graph=obs["room_graph"],
                                           room_distances=room_distances)
        response, action, argument = self.send_query(conversation=conversation)

        robot_pose_pre = np.concatenate((self.env.robots[0].get_position_orientation()))
        subpolicy_success, done, self.last_env_feedback = self.execute_action(action=action,
                                                                              argument=argument,
                                                                              task_desc=task_description,
                                                                              graph=graph,
                                                                              vor_graph=obs["separated_voronoi_graph"],)
        conversation.add_message(self.last_env_feedback)
        self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
        
        robot_pose_post = np.concatenate((self.env.robots[0].get_position_orientation()))

        num_retries, max_retries = 0, 5
        # only re-try if robot pose didn't change. Otherwise do a normal next high-level step with the new observation
        while (not subpolicy_success) and np.all((robot_pose_post - robot_pose_pre) < 0.1) and (not done) and (num_retries < max_retries):
            # recompute obs so that num_high_level_steps counter is correctly increased
            obs = self.env.get_state(compute_scene_graph=True)
            try:
                _apply_room_classification(obs)
            except:
                break
            retrial_prompt = f"The last action {action}({argument}) failed. Please try another command."
            conversation.add_message({"role": "user", "content": retrial_prompt})
            response, action, argument = self.send_query(conversation=conversation)
            subpolicy_success, done, self.last_env_feedback = self.execute_action(action=action,
                                                                                  argument=argument,
                                                                                  task_desc=task_description,
                                                                                  graph=graph,
                                                                                  vor_graph=obs["separated_voronoi_graph"])
            conversation.add_message(self.last_env_feedback)
            self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
            num_retries += 1
        if (num_retries == max_retries) and (not subpolicy_success) and (not done):
            done = True
            self.episode_info["failure_reason"] = "max retrials reached"
        self.episode_info["total_num_retrials"] += num_retries
        self.episode_info["steps_with_retrial"] += (num_retries > 0)

        if done:
            task_success = self.evaluate_success()
        else:
            task_success = False

        if sum([response == r for r in self.prev_responses]) >= 3:
            print("WARNING: LLM response is the same as in the previous steps. Probably stuck.")
            if not self.episode_info.get("failure_reason", None):
                self.episode_info["failure_reason"] = "llm stuck"
                self.episode_info["task_success"] = False
            done = True
        if len(self.prev_responses) > 6:
            del self.prev_responses[0]
        self.prev_responses.append(response)

        return done, task_success, self.episode_info

    def open_graph_node(self, graph, vor_graph, argument):
        def _get_possible_nodes(room_name, object_name):
            # Check that the chosen object is actually articulated while checking for doors as well
            if "door" in object_name:
                possible_nodes = [closed_int_door for closed_int_door in graph.nodes.get(room_name, {}).get("closed_doors", [])]
            else:
                # Check that the chosen object type is actually articulated
                possible_nodes = list()
                for n in graph.successors(room_name):
                    is_closed = (not graph.nodes[n]['states'].get(object_states.Open, True))
                    graph_name = self.llm.human_to_graph_name(object_name)
                    # account for plural form of object name
                    if ((graph_name in n) or (graph_name.strip('s') in n)) and is_closed:
                        possible_nodes.append(n)
            return possible_nodes, [graph.nodes[n] for n in possible_nodes]
        try:
            room_name, object_name = self._parse_room_object_argument(argument)
            possible_node_names, possible_nodes = _get_possible_nodes(room_name, object_name)
        except:
            room_name, object_name = "None", "None"
            possible_node_names, possible_nodes = [], []

        if not possible_nodes:
            feedback = f"Object opening failed: no closed object instances of type {object_name} in {room_name}."
            return False, (None, graph.nodes.get(room_name, {}).get("pos_map", None)), feedback
        
        # for other objects, we take the closest voronoi node as determined by closest-viewpoint logic. For doors, just take the overall closest voronoi node, as they are connecting multiple rooms
        if "door" in argument:
            vor_nodes = np.stack(vor_graph.nodes)
            closest_vor_nodes = dict()
            for node_name, node in zip(possible_node_names, possible_nodes):
                pos = self.env.slam.world2voxel(np.array(node)) if isinstance(node, tuple) else node["pos_map"][:2]
                dists_to_door = np.linalg.norm(np.array(pos) - vor_nodes, axis=1)
                is_close_to_door = dists_to_door <= max(2.0 / self.slam.voxel_size, dists_to_door.min())
                points = [tuple(i) for i in self.env.slam.voxel2world(vor_nodes[is_close_to_door])]
                _idx, _point, vn_costs, _paths = self._find_closest_point(points)
                # also add 2x distance of node <-> door to, from those on the shortest path, also take the vornoi-node that is closest to the door
                vn_costs += 2 * dists_to_door[is_close_to_door]
                closest_to_robot = points[np.argmin(vn_costs)]
                closest_vor_nodes[node_name] = closest_to_robot        
            idx, point, _costs, _paths = self._find_closest_point(list(closest_vor_nodes.values()), euclidean_heuristic=True)
        else:
            idx, point, _costs, _paths = self._find_closest_point(possible_nodes)

        obj = self.scene.objects_by_name.get(possible_node_names[idx], None)
        open_success, feedback = self.open_object(obj, point)
        return open_success, (obj, graph.nodes[room_name]["pos_map"]), feedback

    @staticmethod
    def _parse_room_argument(room_name):
        # don't replace any '-' that connect the room to the room number
        room_name = room_name[:-3].replace("_", " ").replace("-", " ") + room_name[-3:].replace("_", "-")
        for v in DIST_MAPPING.values():
            room_name = room_name.replace(f"({v})", "")
        return room_name

    @staticmethod
    def _parse_room_object_argument(argument):
        room_name, object_name = argument.split(", ")
        # returns false if it doesn't have a singular. E.g. singular_noun("book") == False -> return original word
        singular = inflect_engine.singular_noun(object_name)
        if singular:
            object_name = singular
        return LLMEnv._parse_room_argument(room_name), object_name

    def execute_action(self, action, argument, task_desc, graph, vor_graph=None):
        print(f"Executing {action}({argument})")
        done = False
        feedback = ""
        
        if action == "navigate":
            assert vor_graph is not None, "voronoi graph is required for navigation"
            try:
                room_name, object_name = self._parse_room_object_argument(argument)
                possible_nodes = [n for n in graph.successors(room_name) if self.llm.human_to_graph_name(object_name) in n]
            except:
                room_name, object_name = "None", "None"
                possible_nodes = []
            if len(possible_nodes) == 0:
                subtask_success = False
                feedback = f"Navigation to object failed because there is no {object_name} in {room_name}"
                history = ActionHistory(action, object_name_graph=self.llm.human_to_graph_name(object_name), position=graph.nodes.get(room_name, {}).get("pos_map", None), subtask_success=subtask_success)
            else:
                idx, point, _costs, _paths = self._find_closest_point([graph.nodes.get(n) for n in possible_nodes])
                # don't require to be exactly at the object, as it might be on a table or similar
                subtask_success = self.navigate_to_point(np.array(point),
                                                         success_thres_dist=1.5,
                                                         face_target=True,
                                                         early_termination_dist=0.3)
                if not subtask_success:
                    feedback = "Navigation to object failed"
                history = ActionHistory(action, object_name_graph=possible_nodes[idx], position=point, subtask_success=subtask_success)
        elif action == "go_to_and_open":
            assert vor_graph is not None, "voronoi graph is required for preliminary object navigation"
            subtask_success, (obj, room_pos), feedback = self.open_graph_node(graph, vor_graph, argument=argument)
            history = ActionHistory(action, 
                                    object_name_graph=obj.name if (obj is not None) else None, 
                                    position=obj.get_position()[:2] if (obj is not None) else room_pos, 
                                    subtask_success=subtask_success,
                                    opendoors_roompos=room_pos)
        elif action == "close":
            subtask_success = False
            raise NotImplementedError()
        elif action == "explore":
            try:
                room_name = self._parse_room_argument(argument)
                frontier_points = graph.nodes.get(room_name, {}).get("frontier_points", {})
                frontier_points_within, frontier_points_leading_out = split_frontier_points(frontier_points)
                # for now just always prioritize those frontiers leading out of the room
                frontier_points = frontier_points_leading_out if len(frontier_points_leading_out) else frontier_points_within
            except:
                room_name = "None"
                frontier_points = []
            if len(frontier_points) == 0:
                subtask_success = False
                feedback = f"Frontier exploration failed because there are not frontier points in {room_name}"
                history = ActionHistory(action, object_name_graph=None, position=graph.nodes.get(room_name, {}).get("pos_map", None), subtask_success=subtask_success)
            else:
                _closest_idx, closest_frontier, _costs, _paths = self._find_closest_point(list(frontier_points))
                subtask_success = self.navigate_to_point(closest_frontier, success_thres_dist=0.5, face_target=True, early_termination_dist=0.5)
                if not subtask_success:
                    feedback = f"Navigation to the frontier point failed."
                history = ActionHistory(action, object_name_graph=None, position=closest_frontier, subtask_success=subtask_success)
        elif action == "done":
            subtask_success = True
            done = True
            history = ActionHistory(action, object_name_graph=None, position=None, subtask_success=subtask_success)
        else:
            raise ValueError(f"Unknown action {action}")

        history.orig_api_call = f"{action}({argument})"
        self.action_history.append(history)

        print(f"Subtask success: {subtask_success}")
        if (self.episode_info.get("failure_reason", "") == "max_high_level_steps timeout"):
            done = True

        task_success = self.task.evaluate_success(env=self.env)
        if task_success:
            if self.config["ground_truth_done_decision"]:
                done = True
            elif self.episode_info.get("num_low_level_steps_gtDone", None) is None:
                # log as alternative metric to evaluate impact of done decision
                self.episode_info["num_low_level_steps_gtDone"] = self.episode_info["num_low_level_steps"]
                self.episode_info["num_high_level_steps_gtDone"] = self.episode_info["num_high_level_steps"]
                self.episode_info["magic_open_actions_gtDone"] = self.episode_info["magic_open_actions"]

        print(feedback)
        feedback_msg = {"role": "env", "content": "; ".join([f"{action}({argument}) success: {subtask_success}", feedback])}
        return subtask_success, done, feedback_msg

    def plot_conversation(self, conversation: Conversation, action: str, argument: str, ax: plt.Axes, font_size=6.5, add_fig_title: bool = True):
        if add_fig_title:
            self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, {action}({argument})")

        ax.clear()
        ax.set_axis_off()
        colors = {"system": "wheat", "user": "orange", "assistant": "orangered", "env": "green"}
        
        h = 0.05
        for m in conversation.messages_including_env:
            orig_txt = f"{m['role']}: {m['content']}"
            # remove empty lines
            txt = "".join([s for s in orig_txt.splitlines(True) if s.strip("\r").strip("\n")])
            txt = txt.strip("\n").strip("\r")
            # props = dict(boxstyle='round', facecolor=colors[m["role"]], alpha=0.5)
            props = dict(boxstyle='round', facecolor="white", alpha=0.0, edgecolor="white")
            t = ax.text(-0.05,
                        0.975 - h, 
                        txt,
                        transform=ax.transAxes, 
                        fontsize=font_size, 
                        verticalalignment='top', 
                        bbox=props,
                        wrap=True
                        )
            t._get_wrap_line_width = lambda : 1.05 * ax.get_window_extent().width
            self.env.f.canvas.draw()
            last_h = t.get_bbox_patch().get_height() / ax.get_window_extent().height
            h += last_h + 0.025
            
            b = t.get_bbox_patch()
            # bb = b.get_window_extent()
            p = FancyBboxPatch(xy=[-0.05, 1.0 - h], 
                               width=1.05, 
                               height=last_h, 
                               transform=ax.transAxes, 
                               alpha=0.5,
                               boxstyle="round,pad=0.01",
                               facecolor=colors[m["role"]],
                               edgecolor="k",
                               clip_on=False)
            ax.add_patch(p)
            
        if (h > 1.025) and (font_size > 4.5):
            # just try again with smaller font size to fit everything into the figure
            self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=ax, font_size=font_size - 0.5, add_fig_title=False)

    def visualize(self, state=None):
        super().visualize(state=state)
        self._label_rooms_on_map(state, ax_idx=1)


class JsonLLMEnv(LLMEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def _create_prompt(self, 
                       task_description, 
                       current_room,
                       graph,
                       room_graph,
                       nlp_history,
                       *args, 
                       **kwargs) -> Conversation:
        system_prompt = f"You are a robot in an unexplored house. Your task is to {task_description}."
        system_prompt += f" You have the following actions available that you can use to achieve this task:\n"
        for i, (action, description) in enumerate(self.possible_actions.items()):
            system_prompt += f"""{i+1}. {action}({description[0]}): {description[1]}\n"""
        system_prompt += f"\nOutput Response Format:\n"\
        "Analysis: describe where you could find the objects of interest and what actions you need to execute to get there.\n"\
        "Reasoning: justify why the next action is important to solve the task.\n"\
        "Command: function call"

        # roughly following SayPlan's prompt (see page 50):
        # {nodes: {room: [{id: bobs_room}, {id: toms_room}, {id: jacks_room}, {id: kitchen}, {id: livingroom}], 
        #          pose: [{id: pose1}, {id: pose2}, {id: pose3}, {id: pose4}, {id: pose5}], 
        #          agent: [{location: bobs_room, id: agent}], 
        #          asset: [{room: toms_room, state: free, affordances: [release], id: bed2}, 
        #                  {room: toms_room, state: closed, affordances: [open, close, release], id: wardrobe2}, 
        #                  {room: kitchen, state: closed, affordances: [open, close, release], id: fridge}, 
        #                  {room: kitchen, affordances: [turn_on, turn_off], state: off, id: coffee_machine}, 
        #                  {room: bobs_room, state: free, affordances: [release], id: bed1}, 
        #                  {room: bobs_room, state: closed, affordances: [open, close, release], id: wardrobe1}], 
        #          object: [{affordances: [pickup], state: inside_of(wardrobe1), attributes: "blue", id: coffee_mug}]
        #          }, 
        #  links: [bobs_room↔pose1, bobs_room↔agent, bobs_room↔bed1, bobs_room↔wardrobe1, 
        #          toms_room↔pose1, toms_room↔pose2, toms_room↔pose5, toms_room↔bed2,
        #          toms_room↔wardrobe2, jacks_room↔pose2, jacks_room↔pose3, kitchen↔pose3,
        #          kitchen↔pose4, kitchen↔pose5, kitchen↔fridge, kitchen↔coffee_machine,
        #          livingroom↔pose4, wardrobe1↔coffee_mug]}
        graph_with_robot = nx.Graph()
        for _node_name, node_data in graph.nodes(data=True):
            if node_data.get("semantic_class_name", None) == "door":
                continue
            if "room_id" in node_data:
                room_name = self.room_classification[NODETYPE.roomname(node_data["room_id"])]
            if node_data["node_type"] == NODETYPE.ROOM:
                keys = ["frontier_points", "closed_doors"]
                props = {k: node_data.get(k, set()) for k in keys if (k in node_data)}
                props["id"] = room_name
            elif node_data["node_type"] == NODETYPE.OBJECT:
                keys = []
                props = {k: node_data[k] for k in keys if (k in node_data)}
                props["id"] = node_data["name"]
                props["room"] = room_name
                if "states" in node_data:
                    for state, value in node_data["states"].items():
                        if state == object_states.Open:
                            props["state"] = "open" if value else "closed"
            else:
                continue
            props["node_type"] = str(node_data["node_type"]).split('.')[1].lower()
            graph_with_robot.add_node(props["id"], **props)
            
            if node_data["node_type"] == NODETYPE.OBJECT:
                graph_with_robot.add_edge(room_name, props["id"])
            elif node_data["node_type"] == NODETYPE.ROOM:
                for room in room_graph.neighbors(NODETYPE.roomname(node_data["room_id"])):
                    graph_with_robot.add_edge(room_name, room)
        
        graph_with_robot.add_node("robot", location=current_room, id="robot")
        graph_with_robot.add_edge(current_room, "robot")
        
        data = nx.node_link_data(graph_with_robot)
        links = list({f"{d['source']} - {d['target']}" for d in data["links"]})
        prompt = f"Scene Graph: {{nodes: {data['nodes']}, links: {links}}},\n".replace("'", "")
        
        if len(nlp_history):
            prompt += f"History: {', '.join(nlp_history)}.\n"

        prompt += f"""What is the best next action to complete the task as efficiently as possible? I you don't think that the object can be found in a known room, prioritize opening doors over exploring a room.\n"""  # In general, prioritize opening doors if there is not enough evidence to explore a promising room.\n""" 
        prompt += f"Remember:\n"\
            "1. Respond with a function call\n"\
            "2. You can only use the objects and rooms that you have already found\n"\
            "3. You can only explore rooms that have frontier points\n"\
            "4. If you have found the object you are looking for, directly call done(). You do not need to navigate to it or interact with it.\n"\
            "5. If some actions failed repeatedly, they may not be possible.\n"

        conversation = Conversation(messages=[self.last_env_feedback,
                                              {"role": "system", "content": system_prompt},
                                              {"role": "user", "content": prompt}])
        return conversation
    
