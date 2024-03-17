# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
from enum import IntEnum, Enum
import numpy as np
from pathlib import Path

from moma_llm.tasks.patched_scene import CLASS_NAME_TO_CLASS_ID

PROJECT_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = Path(__file__).parent.parent
TRAINING_SCENES = ["Merom_0_int", "Benevolence_0_int", "Pomaria_0_int", "Wainscott_1_int", "Rs_int", "Ihlen_0_int", "Beechwood_1_int", "Ihlen_1_int"]
TEST_SCENES = ["Benevolence_1_int", "Wainscott_0_int", "Pomaria_2_int", "Benevolence_2_int", "Beechwood_0_int", "Pomaria_1_int", "Merom_1_int"]
MAX_TURN_ANGLE = 0.35

POSSIBLE_ROOMS = ["kitchen", "living room", "combined kitchen and living room", "bedroom", "bathroom", "hallway", "office", "other room", "unknown room"]

EXTERIOR_DOORS = {
    "Beechwood_0_int": ['door_93', 'door_109'],
    "Beechwood_1_int": [],
    "Benevolence_0_int": ['door_9', 'door_12', 'door_13'],
    "Benevolence_1_int": ['door_52'],
    "Benevolence_2_int": ["door_37"],  # door_37: not an exterior door, but when opening it, the robot is trapped in a corner.
    "Ihlen_0_int": ['door_42'],
    "Ihlen_1_int": ['door_86', 'door_91'],
    "Merom_0_int": ['door_60'],
    "Merom_1_int": ['door_74', 'door_93', 'door_90'],  # door_90: not an exterior door, but when opening it, the robot is trapped in a corner.
    "Pomaria_0_int": ['door_41', 'door_42', 'door_52'],  # door_52: not an exterior door, but when opening it, the robot is trapped in a corner. As there is nothing in the tiny room behind the door, just ignore it
    "Pomaria_1_int": ['door_65', 'door_70', "bottom_cabinet_26"],  # "bottom_cabinet_1047"
    "Pomaria_2_int": [],
    "Rs_int": ['door_54'],
    "Wainscott_0_int": ['door_126', 'door_128', 'door_132', 'door_135', "bottom_cabinet_79"],
    "Wainscott_1_int": [],
}

BLOCKING_DOORS = {
    "Beechwood_1_int": ["door_75"],
    "Ihlen_1_int": ["door_106"],
    "Wainscott_0_int": ["door_138"],
}



CLASS_ID_TO_CLASS_NAME = {v: k for k,v in CLASS_NAME_TO_CLASS_ID.items()}

class OCCUPANCY(IntEnum):
    UNEXPLORED = 0
    FREE = 1
    OCCUPIED = 2

    @staticmethod
    def cmap():
        return {OCCUPANCY.UNEXPLORED: (0, 0, 0),
                OCCUPANCY.FREE: (0, 1, 0),
                OCCUPANCY.OCCUPIED: (0, 0, 1)}
        
    @staticmethod
    def to_rgb(arr):
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3))
        for k, v in OCCUPANCY.cmap().items():
            rgb[arr == k] = v
        return rgb


class NODETYPE(IntEnum):
    ROOT = 0
    FLOOR = 1
    ROOM = 2
    OBJECT = 3
    
    @staticmethod
    def roomname(room_id: int):
        return f"room-{room_id}"


class EDGETYPE(Enum):
    onTop = "onTop"
    inside = "inside"
    under = "under"
    inHand = "inHand"
    inRoom = "inRoom"
    roomConnected = "roomConnected"


class FRONTIER_CLASSIFICATION(IntEnum):
    WITHIN = 0
    LEADING_OUT = 1