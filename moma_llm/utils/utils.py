# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
from pathlib import Path
import numpy as np
import trimesh
import igibson
from igibson.utils.utils import quat_pos_to_mat
from moma_llm.utils.constants import PROJECT_DIR


def get_config(config_file: str) -> Path:
    if (PROJECT_DIR / "configs" / config_file).exists():
        config_file = str(PROJECT_DIR / "configs" / config_file)
    else:
        config_file = Path(igibson.configs_path) / config_file
    return config_file


def get_random_action(env):
    action = np.zeros(env.action_space.shape)
    action[[0, 1]] = np.random.uniform(-1, 1, size=2)
    return action


def get_obj_bounding_box(obj):
    half_extent = obj.bounding_box / 2.0
    corners = np.stack([- half_extent, + half_extent])

    bbox_transform = quat_pos_to_mat(obj.get_position(), obj.get_orientation())
    world_frame_vertex_positions = trimesh.transformations.transform_points(corners, bbox_transform)
    return world_frame_vertex_positions