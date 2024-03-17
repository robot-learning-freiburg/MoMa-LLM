# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import cv2
import numpy as np
import pyastar
import pybullet as pb
from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import set_base_values_with_z
from igibson.utils.utils import cartesian_to_polar

from moma_llm.utils.constants import MAX_TURN_ANGLE, OCCUPANCY


def cand_poses_around_object(object_pos, object_yaw, yaw_extent, radius, num_proposals=20):
    """
    Returns a circle around the object with the given radius.
    :param object_pos: position of the object in world frame
    :param radius: radius of the circle in meters around object
    :param num_proposals: number of pose proposals around the object
    """
    # positions around object based on yaw_extent and radius
    object_pos = np.array(object_pos)[:2]
    angles = normalize_angle_minuspi_pi(np.linspace(object_yaw - yaw_extent, object_yaw + yaw_extent, num_proposals))
    cand_pos = object_pos + radius*np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # orientations
    cand_yaw = np.arctan2(object_pos[1] - cand_pos[:, 1], object_pos[0] - cand_pos[:, 0])
    return cand_pos, cand_yaw


def get_circular_kernel(radius):
    if radius > 2:
        def _get_circular_kernel(radius):
            img = np.zeros((2*radius + 1, 2*radius + 1))
            center = (radius, radius)
            color = 1
            thickness = -1
            return cv2.circle(img, center, radius, color, thickness)
        kernel = _get_circular_kernel(radius)
    elif radius > 0:
        kernel = np.ones((2*radius + 1, 2*radius + 1))
    else:
        raise ValueError(f"radius {radius} too small to inflate the map given the current resolution")
    return kernel.astype(np.uint8)


class PyAstarHelper:
    UNEXPLORED_COST = 500
    OCCUPIED_COST = 10_000

    @staticmethod
    def get_inflated_map_weights(m, inflation_radius_m: float, resolution):
        if inflation_radius_m:
            radius = round(inflation_radius_m / resolution)
            kernel = get_circular_kernel(radius)
            # return dilation(m.to(torch.float32), kernel=torch.tensor(circular_kernel, device=m.device, dtype=torch.float32), engine="convolution")
            inflated_map = cv2.dilate(m.copy(), kernel, iterations=1)
        else:
            inflated_map = m.copy()
        assert inflated_map.min() >= 0.0 and inflated_map.max() <= 1.0, (inflated_map.min(), inflated_map.max())
        weights = inflated_map.astype(np.float32)
        return weights
    
    @staticmethod
    def map_to_weights(occupancy_map, inflation_radius_m: float, resolution: float, add_wall_avoidance_cost: bool):
        """inflate the occupied areas, but not the unexplored areas"""
        binary_occupancy_map = (occupancy_map == OCCUPANCY.OCCUPIED).astype(np.uint8)
        assert binary_occupancy_map.dtype == np.uint8, binary_occupancy_map.dtype

        weights = PyAstarHelper.UNEXPLORED_COST * PyAstarHelper.get_inflated_map_weights((occupancy_map == OCCUPANCY.UNEXPLORED).astype(np.uint8), inflation_radius_m=inflation_radius_m, resolution=resolution)
        occupied_weights = PyAstarHelper.get_inflated_map_weights(binary_occupancy_map, inflation_radius_m=inflation_radius_m, resolution=resolution)
        weights[occupied_weights > 0] = PyAstarHelper.OCCUPIED_COST
        if add_wall_avoidance_cost:
            avoid_wall_weights = PyAstarHelper.get_inflated_map_weights(binary_occupancy_map, inflation_radius_m=inflation_radius_m + resolution, resolution=resolution)
            weights[avoid_wall_weights > 0]  += 1
        # weights need to be at least 1 for pyastar -> set walls to high value, all other to low values
        weights += 1
        return weights


def find_floor_idx(env, z: float):
    floor_heights = np.array(env.scene.floor_heights)
    return np.argmin(np.abs(floor_heights - z))
    

def plan_waypoints(env, target_pos_world, inflation_radius_m: float, filter_collision_points: bool = False, add_wall_avoidance_cost: bool = True):   
    if len(target_pos_world) == 3:
        robot_floor = find_floor_idx(env, env.robots[0].base_link.get_position()[2])
        target_pos_floor = find_floor_idx(env, target_pos_world[2])
        assert robot_floor == target_pos_floor, "robot and target object must be on same floor"
    
    start_pos_world = env.robots[0].base_link.get_position()
    
    start_pos_pixel = tuple(env.slam.world2voxel(start_pos_world[:2]))
    target_pos_pixel = tuple(env.slam.world2voxel(np.array(target_pos_world[:2])))
    
    occupancy_map = env.slam.bev_map_occupancy
    weights = PyAstarHelper.map_to_weights(occupancy_map, inflation_radius_m=inflation_radius_m, resolution=env.slam.voxel_size, add_wall_avoidance_cost=add_wall_avoidance_cost)
    # ensure robot position is free
    weights[start_pos_pixel[0] - 1:start_pos_pixel[0] + 2, 
            start_pos_pixel[1] - 1:start_pos_pixel[1] + 2] = np.minimum(weights[start_pos_pixel[0] - 1:start_pos_pixel[0] + 2, 
                                                                        start_pos_pixel[1] - 1:start_pos_pixel[1] + 2], 
                                                                        PyAstarHelper.UNEXPLORED_COST)
    weights[start_pos_pixel[0], start_pos_pixel[1]] = 1

    path_pixels = pyastar.astar_path(weights, 
                                     start_pos_pixel, 
                                     target_pos_pixel, 
                                     allow_diagonal=True, 
                                     costfn='linf')
    path_world = env.slam.voxel2world(path_pixels)
    
    costs = weights[path_pixels[:, 0], path_pixels[:, 1]]

    if filter_collision_points:
        # only return waypoints that go through unexplored or free space
        collision_free = costs.cumsum() < PyAstarHelper.OCCUPIED_COST
        return path_world[collision_free], costs[collision_free]
    else:
        return path_world, costs


def normalize_angle_minuspi_pi(angle):
    two_pi = 2 * np.pi
    return angle - two_pi * np.floor((angle + np.pi) / two_pi)


def turn_to_target_point(env, target_pos_world, max_turn_angle: float, z_offset: float):
    current_pos = env.robots[0].base_link.get_position()[:2]
    target_yaw = cartesian_to_polar(target_pos_world[0] - current_pos[0], target_pos_world[1] - current_pos[1])[1]
    return turn_to_target_yaw(env=env, target_yaw=target_yaw, max_turn_angle=max_turn_angle, z_offset=z_offset)
    

def turn_to_target_yaw(env, target_yaw, max_turn_angle: float, z_offset: float):
    current_pos = env.robots[0].base_link.get_position()[:2]
    current_yaw = env.robots[0].get_rpy()[2]
    
    angle_diff = normalize_angle_minuspi_pi(target_yaw - current_yaw)
    while abs(angle_diff) > max_turn_angle:
        current_yaw += np.sign(angle_diff) * max_turn_angle
        set_base_values_with_z(env.robots[0].get_body_ids()[0], [current_pos[0], current_pos[1], current_yaw], z=z_offset)
        angle_diff = normalize_angle_minuspi_pi(target_yaw - current_yaw)
        
        env.get_state()
        env.simulator.sync() 
    if abs(angle_diff) > 0.05:
        set_base_values_with_z(env.robots[0].get_body_ids()[0], [current_pos[0], current_pos[1], target_yaw], z=z_offset)
        
        env.get_state()
        env.simulator.sync()         
    
    
def drive_to_target_position(env: iGibsonEnv, 
                             target_pos_world, 
                             inflation_radius_m: float, 
                             max_turn_angle: float = MAX_TURN_ANGLE, 
                             success_thres_dist: float = 0.5,
                             early_termination_dist: float = 0.0,
                             face_target: bool = True,
                             debug: bool = False) -> list:
    """
    Plan waypoints to goal, then set robot to waypoints and collect sensor observations to update map afterwards.
    Drives to the target position, stops in the orientation of the path leading to the position.
    """
    waypoints, costs = plan_waypoints(env=env, 
                                      target_pos_world=target_pos_world, 
                                      inflation_radius_m=inflation_radius_m,
                                      filter_collision_points=True,
                                      add_wall_avoidance_cost=True)
    
    def _calc_dist(point_a, point_b):
        return np.linalg.norm(point_a[:2] - point_b[:2])
    
    z_offset = 0.05
    current_pos = env.robots[0].get_position()[:2]
    next_wp = None
    max_replans = max(min(int(50 * _calc_dist(current_pos, target_pos_world)), 250), 50)
    replans = 0
    prev_pose = np.concatenate(env.robots[0].get_position_orientation())
    
    # Note: not checking collisions for the waypoints during execution
    # skip last two waypoints to not clip through walls in final step sometimes
    while len(waypoints) > 1:
        current_pos = env.robots[0].get_position()[:2]
        # waypoints[0] is the current position
        next_wp, next_wp_cost = waypoints[1], costs[1]
        waypoints, costs = waypoints[2:], costs[2:] 
    
        next_wp_free = next_wp_cost < PyAstarHelper.UNEXPLORED_COST
        freespace_wps = waypoints[costs.cumsum() < PyAstarHelper.UNEXPLORED_COST]
        if next_wp_free:
            # turn towards the next waypoint
            next_wp_yaw = cartesian_to_polar(next_wp[0] - current_pos[0], next_wp[1] - current_pos[1])[1]
            turn_to_target_yaw(env=env, target_yaw=next_wp_yaw, max_turn_angle=max_turn_angle, z_offset=z_offset)
                
            # move to next waypoint
            set_base_values_with_z(env.robots[0].get_body_ids()[0], [next_wp[0], next_wp[1], next_wp_yaw], z=z_offset)
            env.get_state()
            env.simulator.sync()

        if early_termination_dist and (_calc_dist(env.robots[0].get_position(), target_pos_world) <= early_termination_dist):
            break
        # if we are not yet close to the goal & the current plan goes through unexplored space, replan based on updated map
        waypoints, costs = plan_waypoints(env=env, 
                                          target_pos_world=target_pos_world, 
                                          inflation_radius_m=inflation_radius_m,
                                          filter_collision_points=True,
                                          add_wall_avoidance_cost=True)
        replans += 1
            
        new_pose = np.concatenate(env.robots[0].get_position_orientation())
        if np.all(np.abs(new_pose - prev_pose) < 0.01):
            #  we didn't make any further progress
            break
        else:
            prev_pose = new_pose
        if (replans > max_replans):
            break

    # turn towards the target position
    if face_target:
        turn_to_target_point(env=env, target_pos_world=target_pos_world, max_turn_angle=max_turn_angle, z_offset=z_offset)

    # update rendering without stepping physics
    env.simulator.sync(force_sync=True)    
    success = _calc_dist(env.robots[0].get_position(), target_pos_world) <= success_thres_dist
    return success


def turn_full_circle(env: iGibsonEnv):
    def _get_yaw():
        robot_orn = env.robots[0].base_link.get_orientation()
        return pb.getEulerFromQuaternion(robot_orn)[2]
    
    current_pos = env.robots[0].get_position()[:2]
    z_offset = 0.05
    max_turn_angle = MAX_TURN_ANGLE
    
    i, max_i = 0, 500
    prev_yaw = _get_yaw()
    dist_travelled = 0.0
    yaw = -np.inf
    while dist_travelled < 2 * np.pi:
        set_base_values_with_z(env.robots[0].get_body_ids()[0], [current_pos[0], current_pos[1], _get_yaw() + max_turn_angle], z=z_offset)
        obs = env.get_state()
        env.simulator.sync() 

        yaw = _get_yaw()
        dist_travelled += abs(abs(yaw) - abs(prev_yaw))
        prev_yaw = yaw
        
        i += 1
        if i >= max_i:
            print("Robot failed to turn full circle")
            break

    success = i < max_i
    return obs, success
