# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
from typing import Any

import cv2
import gymnasium
import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

try:
    matplotlib.use("TkAgg")
    matplotlib.interactive(True)
except:
    pass
from igibson import object_states
from igibson.external.pybullet_tools.utils import (get_link_pose,
                                                   get_link_position_from_name,
                                                   link_from_name,
                                                   set_base_values_with_z)

from moma_llm.env.env import OurIGibsonEnv, create_igibson_env
from moma_llm.navigation.navigation import (PyAstarHelper,
                                             cand_poses_around_object,
                                             drive_to_target_position,
                                             normalize_angle_minuspi_pi,
                                             plan_waypoints, turn_full_circle,
                                             turn_to_target_point)
from moma_llm.utils.constants import OCCUPANCY


class HighLevelEnv(gymnasium.Wrapper):
    def __init__(self, env: OurIGibsonEnv, seed: int) -> None:
        super().__init__(env)
        self.seed = seed

    def __getattr__(self, name: str) -> Any:
        """Overwritten just to get rid of the annoying warning."""
        return getattr(self.env, name)
    
    def reset(self, config_file, scene_id, episode_num, compute_scene_graph: bool = False):
        success = False
        i = 0
        while not success:
            # if ((self.env.config["use_prior_graph_distribution"] and episode_num and (episode_num % self.env.config["prior_graph_object_resample_freq"] == 0))
            #     or (scene_id != self.env.scene.scene_id)):
            # simulator seems to not reliably reset, e.g. leaving doors open but then also recognizing them in the close space on the bev map. So to be sure, always recreate the simulator at each reset
            if True:
                self.env.close()
                self.env = create_igibson_env(config_file=config_file, 
                                              control_freq=self.env.config["control_freq"], 
                                              scene_id=scene_id,
                                              seed=self.seed + 69 * episode_num + 999 * i,)

            _obs = self.env.reset()
            self.episode_info.update({"steps_with_retrial": 0,
                                      "total_num_retrials": 0,
                                      "target_category": self.env.task.target_category,})

            for _ in range(10):
                _obs = self.env.step(np.zeros(self.env.action_space.shape))

            _obs, success = turn_full_circle(self.env)
            
            if success and self.env.config["reject_onestep_episodes"] and self.task.evaluate_success(self.env):
                success = False
            i += 1
            
        return self.env.get_state(compute_scene_graph=compute_scene_graph)
    
    def navigate_to_point(self, target_pos_world, success_thres_dist: float, face_target: bool, early_termination_dist: float) -> bool:        
        self.env.plot_object_position(target_pos_world, color="lime", marker="*")
        success = drive_to_target_position(env=self.env,
                                           target_pos_world=target_pos_world,
                                           inflation_radius_m=self.env.config["navigation_inflation_radius"],
                                           success_thres_dist=success_thres_dist,
                                           face_target=face_target,
                                           early_termination_dist=early_termination_dist)
        return success

    def _find_closest_point(self, points, euclidean_heuristic: bool = True):
        def _to_navpoint(point):
            if isinstance(point, dict):
                # point is a node: take closest voronoi point + euclidean distance from voronoi to actual position
                p = self.env.slam.voxel2world(point["closest_vor_node"][:2])
                euclidean_dist = np.linalg.norm(np.array(point["pos"][:2]) - p)
            else:
                p = point
                euclidean_dist = 0
            return p, euclidean_dist
        
        nav_points = []
        costs = []
        paths = []
        for point in points:
            p, euclidean_dist = _to_navpoint(point)
            path, cost = plan_waypoints(env=self.env,
                                        target_pos_world=p,
                                        inflation_radius_m=self.env.config["navigation_inflation_radius"],
                                        add_wall_avoidance_cost=False)
            nav_points.append(p)
            
            if not isinstance(point, dict):
                # e.g. for frontier points, the last steps will be in (inflated) unexplored space. For fair distance comparison with object-nav, ignore unexplored costs of the last few cells
                last_cells = int(0.25 / self.env.slam.voxel_size)
                mask = (cost[-last_cells:] == PyAstarHelper.UNEXPLORED_COST + 1)
            else:
                last_cells = int(np.ceil(self.env.config["navigation_inflation_radius"] / self.env.slam.voxel_size))
                mask = (cost[-last_cells:] >= PyAstarHelper.OCCUPIED_COST)
            cost[-last_cells:][mask] = 1
            total_cost = cost.sum()
            if euclidean_heuristic:
                total_cost += euclidean_dist
            costs.append(total_cost)
            paths.append(path)
        idx = np.argmin(costs)
        return idx, nav_points[idx], costs, paths

    def open_object(self, obj, nav_point):
        """
        object: the scene object instance we want to open
        nav_point: a safe navigation point to go to before opening the object. Passing the object location directly, might fail or end up navigating to the wrong side e.g. of a door - untested
        """
        feedback = ""
        subtask_success_nav = self.navigate_to_point(np.array(nav_point), success_thres_dist=1.5, face_target=False, early_termination_dist=0.5)
        if not subtask_success_nav:
            feedback = f"Navigation to {obj.name} failed"
            return False, feedback
        if obj.name in self.env.opened_doors:
            feedback = f"Door {obj.name} already open."
            return True, feedback
        
        # note: this would fail for opened_doors, as they are no nodes in the graph atm. So make sure to keep after return statement for those cases
        # obj_instance_id = graph.nodes[possible_node_names[idx]]["instance_id"]
        body_ids = obj.get_body_ids()
        assert len(body_ids) == 1, "Open state only supports single-body objects."
        obj_instance_id = body_ids[0]
        
        try:
            both_sides, relev_joint_infos, joint_dirs = object_states.open.get_relevant_joints(obj)
            assert relev_joint_infos is not None
        except:
            feedback = f"Object {obj.name} has no joints to open it."
            return False, feedback
        
        def _chose_pose_in_front(obj, obj_com_pos, radius, bbox_center_world, bbox_extent_frame):
            base_link_pos_world, base_link_orn_world = get_link_pose(obj_instance_id, link_from_name(obj_instance_id, str(relev_joint_infos[0].linkName).strip("b").strip("'")))
            base_link_orn_world_euler = R.from_quat(base_link_orn_world).as_euler("xyz")
        
            robot_inflation = self.env.config["navigation_inflation_radius"]
            if "door" in obj.name:
                # door "direction" and positive axis are always pointing in opposite directions
                object_yaw = normalize_angle_minuspi_pi(base_link_orn_world_euler[2] + np.pi)
                obj_width = np.linalg.norm(2*(obj_com_pos - base_link_pos_world)[0:2])
                pot_robot_pos, pot_robot_yaw = cand_poses_around_object(base_link_pos_world, object_yaw, np.pi/2.7, radius=obj_width + robot_inflation, num_proposals=20)            
            else:
                object_yaw = base_link_orn_world_euler[2]
                obj_depth = bbox_extent_frame[1]
                if obj.category == "window":
                    possible_object_yaws = [object_yaw - np.pi/2, object_yaw + np.pi - np.pi/2]
                    robot_to_obj_vec = obj_com_pos[:2] - self.env.robots[0].get_position()[:2]
                    dot_prods = []
                    for yaw in possible_object_yaws:
                        window_vec = np.array([np.cos(yaw), np.sin(yaw)])
                        dot_prods.append(np.dot(window_vec, robot_to_obj_vec))
                    # take the one that is closer to -1 (antiparallel robot_to_obj_vec and window_vec)
                    if abs(dot_prods[0] + 1) < abs(dot_prods[1] + 1):
                        object_yaw = possible_object_yaws[0]
                    else:
                        object_yaw = possible_object_yaws[1]
                    obj_depth = 0.4
                    object_yaw += np.pi/2.7 # only do this here to superimpose the -np.pi/2.7 yaw offset later
                # idea: 0.5 until edge of object + lenght of the drawer (equal to object depth)
                dist_to_obj = radius * obj_depth + robot_inflation
                dist_to_obj = max(dist_to_obj, 0.5)
                pot_robot_pos, pot_robot_yaw = cand_poses_around_object(bbox_center_world, object_yaw - np.pi/2, np.pi/2, radius=dist_to_obj, num_proposals=20)            

            if "door" in obj.name:
                # chose the nearest point
                idx, chosen_robot_pos, costs, _paths = self._find_closest_point(pot_robot_pos)
                costs = np.ones(len(pot_robot_pos))
            else:
                # chose the free point that is most central in front of the object, as we have to be able to look inside it
                _idx, _pos, costs, _paths = self._find_closest_point(pot_robot_pos)
                costs = np.array(costs)
                feasible = (costs < PyAstarHelper.OCCUPIED_COST)
                if any(feasible):
                    dist_from_center = np.abs(np.arange(len(pot_robot_pos)) - (len(pot_robot_pos) / 2) )
                    centered_weigths = (1 / (1 + dist_from_center)) * feasible 
                    idx = np.argmax(centered_weigths)
                else:
                    idx = np.argmin(costs)
                
            return pot_robot_pos[idx], costs[idx], pot_robot_pos
            
        # First, navigate to a safe position on a circle around the object
        obj_com_pos, obj_com_orn = obj.get_base_link_position_orientation() 
        # pos, orn of base link (in center of mass) given in world frame
        bbox_center_world, bbox_orn_world, bbox_extent_frame, bbox_center_frame = obj.get_base_aligned_bounding_box()
        costs, found_valid, radii, chosen_robot_poses = [], False, [1.7, 2.7], []
        for radius in radii:  # 
            chosen_robot_pos, path_cost, _pot_robot_pos = _chose_pose_in_front(obj, obj_com_pos, radius=radius, bbox_center_world=bbox_center_world, bbox_extent_frame=bbox_extent_frame)
            costs.append(path_cost)
            chosen_robot_poses.append(chosen_robot_pos)
            if path_cost < PyAstarHelper.OCCUPIED_COST:
                found_valid = True
                break
        if not found_valid:
            chosen_robot_pos = chosen_robot_poses[np.argmin(costs)]
        subtask_success_nav = self.navigate_to_point(chosen_robot_pos, success_thres_dist=0.21, face_target=False, early_termination_dist=0.0)
        turn_to_target_point(env=self.env, target_pos_world=obj_com_pos, max_turn_angle=0.35, z_offset=0.05)
        if not subtask_success_nav:
            feedback = f"Point-wise navigation to safe position for {obj.name} position failed"
            return False, feedback

        # delete the object from the map, as we change it
        # NOTE: only do this here, as generally deleting stixel from the map somehow leads to incorrectly replacing occupied with un-occupied space
        #   this basically assumes that the map is static except when we manipulate an object
        if obj.category != "window":
            self.env.slam.delete_obj_from_voxel_map(obj)

        # Actual object opening action
        counter = 0
        num_retries = 20
        effective = False
        while counter < num_retries and not effective:
            effective = obj.states[object_states.Open].set_value(True, fully=True)
            self.env.simulator.sync(force_sync=True)
            counter += 1
        # needed for obj.states[object_states.Open].get_value() to update
        self.env.run_simulation()
        self.env.get_state()
        self.env.episode_info["magic_open_actions"] += 1
        # window's can't be openened in igibson, so we just assume it worked
        if effective or (obj.category == "window"):
            if "door" in obj.name:
                self.env.opened_doors.append(obj.name)
            subtask_success = True
            self.env.simulator.sync(force_sync=True)
            print("Object {} opened. ".format(obj.name)) # might need time to see it reflected in the sim, 
        else:
            feedback = "Manipulation of {} failed.".format(obj.name)
            return False, feedback
        
        if obj.category == "window":
            self.opened_windows.append(obj.name)

        if "door" in obj.name:
            subtask_success_nav = self.navigate_to_point(bbox_center_world, success_thres_dist=0.3, face_target=False, early_termination_dist=0.0)
            if not subtask_success_nav:
                feedback = f"Point-wise navigation to door-frame (after opening) failed for {obj.name}"
                return False, feedback
        else:
            inside_obj_name = self.env.task.new_parent_relations.get(obj, {}).get(object_states.Inside, None)
            if inside_obj_name is not None:               
                # HACK: add objects that are inside to the seen objects, whether they've been seen on camera or not
                inside_obj = self.env.scene.objects_by_name[inside_obj_name]
                instance_id = inside_obj.renderer_instances[0].id
                is_seen = instance_id in self.env.slam.instance_viewpoints
                
                if not is_seen:
                    eyes_pos = self.env.robots[0].eyes.get_position()
                    dist = np.linalg.norm(eyes_pos - inside_obj.get_position())
                    self.env.slam.instance_viewpoints[instance_id] = (eyes_pos, dist)
                    
                print(f"{inside_obj_name} revealed inside {obj.name}.")
            
        return subtask_success, feedback

    def evaluate_success(self):
        task_success = self.env.task.evaluate_success(self.env)
        if (not task_success) and (not self.episode_info.get("failure_reason", None)):
            self.episode_info["failure_reason"] = "wrong termination by llm"
        return task_success
    
    def visualize(self, state=None):
        self.env.visualize(state=state)
