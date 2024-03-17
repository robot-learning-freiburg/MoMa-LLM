# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import os
import random

import networkx as nx
import numpy as np
from igibson import object_states

from moma_llm.env.llm_env import LLMEnv
from moma_llm.utils.constants import NODETYPE, POSSIBLE_ROOMS


class GreedyBaseline(LLMEnv):
    def __init__(self, env, llm, seed) -> None:
        super().__init__(env, llm, seed=seed)
        self.nav_fails = 0
        self.max_nav_fails = 10

    def reset(self, *args, **kwargs):
        self.nav_fails = 0
        return super().reset(*args, **kwargs)

    def frontier_navigation(self, frontier_point):
        subtask_success = self.navigate_to_point(frontier_point, success_thres_dist=1.5, face_target=True, early_termination_dist=0.5)
        if not subtask_success:
            print(f"Navigation to the frontier point failed.")
        return subtask_success

    def _get_all_closed_doors(self, graph):
        door_nodes = list()
        # Go through all rooms and get closed doors:
        for n, data in graph.nodes(data=True):
            if data["node_type"] == NODETYPE.ROOM:
                if "closed_doors" in data.keys():
                    for door in data["closed_doors"]:
                        door_nodes.append(graph.nodes.get(door))
        return door_nodes
    
    def _get_all_closed_objects(self, graph):
        closed_object_nodes = list()
        for n, data in graph.nodes(data=True):
            if (data["node_type"] == NODETYPE.OBJECT):
                if (not (data["states"].get(object_states.Open, True))
                    or ((data["semantic_class_name"] == "door") and (n not in self.env.opened_doors))):
                    closed_object_nodes.append(data)
        return closed_object_nodes
    
    def _object_node_to_argument(self, node):
        if isinstance(node['room_id'], int):
            room_name = 'room_{}'.format(node['room_id'])
        else:
            room_name = node['room_id']
        return f"{room_name}, {self.llm.to_human_readable_object_name(node['name'])}"

    def _action_selection(self, frontier_points, object_nodes, closed_door_nodes, obs):
        all_points = frontier_points + object_nodes
        closest_idx, _closest_door_pos, _costs, _paths = self._find_closest_point(all_points)
        if closest_idx < len(frontier_points):
            return all_points[closest_idx], "frontier"
        else:
            return all_points[closest_idx], "object" 

    def take_action(self, obs: dict, task_description: str):
        # the llm has to take a step to make this decision - so to make it fair, the baseline can only evaluate this in the beginning of the next step after moving somewhere
        task_success = self.env.task.evaluate_success(self.env)
        if not task_success:
            graph = obs["room_object_graph"]

            rooms_with_frontier = [graph.nodes.get(n)["frontier_points"] for n in graph.successors("root")]
            frontier_points = list(set().union(*rooms_with_frontier))
            frontier_points = [p[0] for p in frontier_points]

            object_nodes = self._get_all_closed_objects(graph)
            # also contained within object_nodes
            closed_door_nodes = self._get_all_closed_doors(graph)
            
            if len(object_nodes + frontier_points):
                target, point_type = self._action_selection(frontier_points=frontier_points, object_nodes=object_nodes, closed_door_nodes=closed_door_nodes, obs=obs)
                if point_type == "object":
                    argument = self._object_node_to_argument(target)
                    nav_success, _done, _feedback = super().execute_action("go_to_and_open", argument, task_desc=task_description, graph=graph, vor_graph=obs["separated_voronoi_graph"])
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, go_to_and_open({argument})")
                elif point_type == "frontier":
                    nav_success = self.frontier_navigation(target)
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, explore({target})")
                else:
                    raise ValueError(f"Unknown point type {point_type}")
            else:
                self.episode_info["failure_reason"] = "no_exploration_points_left"
                nav_success = False
                print(f"take_action failed because there are no closed objects or frontier points left")  
            
            self.nav_fails += (not nav_success)
            if self.nav_fails > self.max_nav_fails:
                self.episode_info["failure_reason"] = "max_nav_fails_reached"

        done = task_success or (self.episode_info.get("failure_reason", None) is not None)
        return done, task_success, self.episode_info


class RandomBaseline(GreedyBaseline):
    def _action_selection(self, frontier_points, object_nodes, closed_door_nodes, obs):
        all_points = frontier_points + object_nodes
        rnd_idx = np.random.randint(0, len(all_points))
        if rnd_idx < len(frontier_points):
            return all_points[rnd_idx], "frontier"
        else:
            return all_points[rnd_idx], "object" 
