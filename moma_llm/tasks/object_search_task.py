# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import logging
from collections import defaultdict

import cv2
import numpy as np
import pyastar
import pybullet as p
from bddl.object_taxonomy import ObjectTaxonomy
from igibson import object_states
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.termination_condition_base import \
    BaseTerminationCondition
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.utils import l2_distance, restoreState

from moma_llm.llm.llm import LLM
from moma_llm.navigation.navigation import (PyAstarHelper, find_floor_idx,
                                             get_circular_kernel,
                                             plan_waypoints)
from moma_llm.tasks.object_sampling import add_objects_from_our_distribution
from moma_llm.utils.constants import (BLOCKING_DOORS, CLASS_NAME_TO_CLASS_ID,
                                       EXTERIOR_DOORS, OCCUPANCY)

log = logging.getLogger(__name__)


class ObjectSearchTask(BaseTask):
    """
    Object Search Task
    The goal is to find a random target object.
    """

    def __init__(self, env):
        super(ObjectSearchTask, self).__init__(env)
        assert isinstance(env.scene, InteractiveIndoorScene), "room rearrangement can only be done in InteractiveIndoorScene"
        # TODO: terminate if success
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        self.reward_functions = []
        
        self.floor_num = 0
        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))
        self.target_category = ""
        self.shortest_dist = None
        self.reachable_targets = []

        self.new_objects = []
        self.new_obj_relations = {}
        self.new_parent_relations = {}
        
        self.all_object_categories = list(set(env.scene.category_ids).union(set(env.scene.objects_by_category.keys())))

        self.scene = env.scene

    @property
    def task_description(self):
        return f"find a {LLM.to_human_readable_object_name(self.target_category)}"
        
    @property
    def task_info(self):
        return {"target_category": self.target_category,
                "shortest_dist": self.shortest_dist}
        
    @staticmethod
    def get_random_point_inflated(env, floor, uniform_over_room: bool):
        trav = env.scene.floor_map[floor] == 0
        trav_inflated = cv2.dilate(trav.astype(np.uint8), get_circular_kernel(radius=5), iterations=1)
        can_trav = (trav_inflated == 0)
        
        if uniform_over_room:
            # sampling random free-space leads to many start poses in the same rooms, as there are usually 1 or 2 rooms with large open areas
            # so may be preferable to first sample a room, then sample a point in that room
            while True:
                room_id = env.np_random.choice(sorted(set(np.unique(env.scene.room_ins_map)) - {0}))
                new_can_trav = np.logical_and(can_trav, env.scene.room_ins_map == room_id)
                if np.any(new_can_trav):
                    can_trav = new_can_trav
                    break

        trav_space = np.where(can_trav)
        idx = env.np_random.integers(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = env.scene.map_to_world(xy_map)
        z = env.scene.floor_heights[floor]
        return floor, np.array([x, y, z])
    
    def _draw_target(self, env, valid_object_categories):
        target_obj_category = env.np_random.choice(sorted(valid_object_categories))
        # check that there is at least one reachable instance of the category, Any instance is valid to solve the task
        possible_targets = [o for o in env.scene.objects_by_category[target_obj_category] if (find_floor_idx(env, o.get_position()[2]) == self.floor_num)]
        return target_obj_category, possible_targets
        
    def sample_initial_pose_and_target_category(self, env):
        _, initial_pos = self.get_random_point_inflated(env=env, floor=self.floor_num, uniform_over_room=True)
        start_room = env.scene.get_room_instance_by_point(initial_pos[:2])
        i, max_trials = 0, 100
        # NOTE: remove door as we don't show open doors to the agent
        valid_object_categories = set(env.scene.objects_by_category.keys()) - {"walls", "floors", "ceilings", "agent", "door"}
        if env.simulator.scene.scene_id == "Pomaria_0_int":
            # these either don't actually exist or ar not visible from anywhere
            valid_object_categories -= {"picture"}
        
        m = env.scene.floor_map[self.floor_num]
        occupancy_map = np.full_like(m, OCCUPANCY.FREE)
        occupancy_map[m == 0] = OCCUPANCY.OCCUPIED
        weights = PyAstarHelper.map_to_weights(occupancy_map, 
                                               inflation_radius_m=env.config["navigation_inflation_radius"], 
                                               resolution=env.slam.voxel_size,
                                               add_wall_avoidance_cost=False)
        
        reachable_targets = []
        
        success = False
        while not success:
            source_map = env.scene.world_to_map(initial_pos[:2])

            target_obj_category = env.np_random.choice(sorted(valid_object_categories))
            # check that there is at least one reachable instance of the category, Any instance is valid to solve the task
            possible_targets = [o for o in env.scene.objects_by_category[target_obj_category] if (find_floor_idx(env, o.get_position()[2]) == self.floor_num)]
            target_obj_category, possible_targets = self._draw_target(env, valid_object_categories)
            
            if any([start_room == env.scene.get_room_instance_by_point(target.get_position()[:2]) for target in possible_targets]):
                # ignore objects in the room that we start, as they are too easy to find immediately
                continue
            
            dists = []
            for target in possible_targets:
                if target.name in self.new_obj_relations:
                    parent_obj, relation = self.new_obj_relations[target.name]
                    if (not env.config["allow_inside_objects_as_targets"]) and (relation == object_states.Inside):
                        continue
                    # if the target is on-top or inside another object, just check that the parent object is reachable
                    pos = parent_obj.get_position()[:2]
                else:
                    pos = target.get_position()[:2]                
                target_map = env.scene.world_to_map(pos[:2])
                path_pixels = pyastar.astar_path(weights, 
                                                 source_map, 
                                                 target_map, 
                                                 allow_diagonal=True, 
                                                 costfn='linf')
                costs = weights[path_pixels[:, 0], path_pixels[:, 1]]
                # assume we have to be able to get at least within 1m of the object
                resolution = env.scene.trav_map_resolution
                in_collision = (costs >= PyAstarHelper.OCCUPIED_COST).cumsum()
                if in_collision.max() > int(1.0 / resolution):
                    continue
                
                path_world = env.scene.map_to_world(path_pixels)
                dist = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
                dists.append(dist)
                reachable_targets.append(target)
                
            success = len(reachable_targets) > 0
            i += 1
            assert i <= max_trials, "Failed to sample initial and target positions"

        # for SPL calculation
        shortest_dist = np.min(dists)

        initial_orn = np.array([0, 0, env.np_random.uniform(0, np.pi * 2)])
        log.debug("Sampled initial pose: {}, {}".format(initial_pos, initial_orn))
        log.debug("Sampled target category: {}".format(target_obj_category))
        return initial_pos, initial_orn, target_obj_category, shortest_dist, reachable_targets

    @staticmethod
    def _open_interior_doors(env):
        if env.config["should_open_all_interior_doors"]:
            for obj in env.simulator.scene.objects_by_category["door"]:
                if obj.name not in EXTERIOR_DOORS[env.simulator.scene.scene_id]:  
                    obj.states[object_states.Open].set_value(True, fully=True)
                
    def reset_scene(self, env):
        self.floor_num = env.scene.get_random_floor()
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)
            
        for door_name in BLOCKING_DOORS.get(env.scene.scene_id, []):
            # remove by moving far away
            env.scene.objects_by_name[door_name].set_position([250, 250, 0])
            
        if env.config["use_prior_graph_distribution"]:
        # NOTE: we now recreate the full environment instead because adding objects is not supported after the render has optimized
        #     for obj in self.new_objects:
        #         self.scene.remove_object(obj)
            self.new_objects, self.new_obj_relations = add_objects_from_our_distribution(env=env)
            self.new_parent_relations = {parent: {relation: child} for child, (parent, relation) in self.new_obj_relations.items()}
            
        # open doors before we spawn the robot to ensure we don't spawn it in a collision
        env.robots[0].reset()
        self._open_interior_doors(env)

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        # We need to first reset the robot because otherwise we will move the robot in the joint conf. last seen before
        # the reset
        env.robots[0].reset()
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_category, shortest_dist, reachable_targets = self.sample_initial_pose_and_target_category(env)
            reset_success = env.test_valid_position(env.robots[0], initial_pos, initial_orn, ignore_self_collision=True)
            restoreState(state_id)
            if reset_success:
                break

        assert reset_success, "WARNING: Failed to reset robot without collision"

        env.land(env.robots[0], initial_pos, initial_orn)
        p.removeState(state_id)

        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.target_category = target_category
        self.shortest_dist = shortest_dist
        self.reachable_targets = reachable_targets

        # super(ObjectSearchTask, self).reset_agent(env)

    def get_task_obs(self, env):
        """
        No task-specific observation
        """
        return
    
    def evaluate_success(self, env) -> bool:
        # don't use the voxel map, as the room-object graph is built based on slam.seen_instances. In a few cases can lead to inconsistencies with the llm prompt if success is not also based on seen_instances.
        # success = CLASS_NAME_TO_CLASS_ID[self.target_category] in set(np.unique(env.slam.voxel_map))
        success = False
        for i in env.slam.seen_instances:
            obj = env.scene.objects_by_id.get(i, None)
            if (obj is not None):
                if (obj.category == self.target_category):
                    success = True
                    break
            else:
                print(f"OBJECT ID NOT FOUND IN SCENE: {i}")
        env.episode_info["task_success"] = success
        return success
