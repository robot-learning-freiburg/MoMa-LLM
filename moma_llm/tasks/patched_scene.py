# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import logging
import os
from collections import OrderedDict
from functools import partial

import igibson
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.constants import MAX_CLASS_COUNT, SemanticClass

log = logging.getLogger(__name__)


# original category: new category
REPLACED_CATEGORIES = {
    "bottom_cabinet_no_top": "bottom_cabinet",
    "countertop": {"bedroom": "nightstand", "kitchen": "kitchen_counter", "childs_room": "nightstand"},
    "chest": {"bedroom": "dresser", "childs_room": "dresser"},
    "breakfast_table": "table",
}


def get_prefixed_joint_name(name, self):
    return self.orig_name + "_" + name


def get_prefixed_link_name(name, self):
    if name == "world":
        return name
    elif name == self.base_link_name:
        # The base_link get renamed as the link tag indicates
        # Just change the name of the base link in the embedded urdf
        return self.orig_name
    else:
        # The other links get also renamed to add the name of the link tag as prefix
        # This allows us to load several instances of the same object
        return self.orig_name + "_" + name


# To maintain backward compatibility, the starting class id should be SemanticClass.SCENE_OBJS + 1
def get_class_name_to_class_id(starting_class_id=SemanticClass.SCENE_OBJS + 1):
    """
    Get mapping from semantic class name to class id

    :param starting_class_id: starting class id for scene objects
    """
    category_txt = os.path.join(igibson.ig_dataset_path, "metadata", "categories.txt")
    class_name_to_class_id = OrderedDict()
    class_name_to_class_id["agent"] = SemanticClass.ROBOTS  # Agents should have the robot semantic class.
    if os.path.isfile(category_txt):
        with open(category_txt) as f:
            for line in f.readlines():
                # The last few IDs are reserved for DIRT, STAIN, WATER, etc.
                assert starting_class_id < SemanticClass.DIRT, "Class ID overflow"
                class_name_to_class_id[line.strip()] = starting_class_id
                starting_class_id += 1
    
    for old_category, new_category_map in REPLACED_CATEGORIES.items():
        if isinstance(new_category_map, str) and new_category_map not in class_name_to_class_id:
            class_name_to_class_id[new_category_map] = starting_class_id
            starting_class_id += 1
        elif isinstance(new_category_map, dict):
            for room_type, new_category in new_category_map.items():
                if new_category not in class_name_to_class_id:
                    class_name_to_class_id[new_category] = starting_class_id
                    starting_class_id += 1

    return class_name_to_class_id
CLASS_NAME_TO_CLASS_ID = get_class_name_to_class_id()


class MonkeyPatchedInteractiveIndoorScene(InteractiveIndoorScene):
    def _orig_add_object(self, obj):
        """
        Adds an object to the scene

        :param obj: Object instance to add to scene.
        """
        # Give the object a name if it doesn't already have one.
        if obj.name in self.objects_by_name.keys():
            log.error("Object names need to be unique! Existing name " + obj.name)
            exit(-1)

        # Add object to database
        self.objects_by_name[obj.name] = obj
        if obj.category not in self.objects_by_category.keys():
            self.objects_by_category[obj.category] = []
        self.objects_by_category[obj.category].append(obj)

        if hasattr(obj, "states"):
            for state in obj.states:
                if state not in self.objects_by_state:
                    self.objects_by_state[state] = []

                self.objects_by_state[state].append(obj)

        if hasattr(obj, "in_rooms"):
            in_rooms = obj.in_rooms
            if in_rooms is not None:
                for in_room in in_rooms:
                    if in_room not in self.objects_by_room.keys():
                        self.objects_by_room[in_room] = []
                    self.objects_by_room[in_room].append(obj)

        if obj.get_body_ids() is not None:
            for id in obj.get_body_ids():
                self.objects_by_id[id] = obj
    
    def _add_object(self, obj):
        if obj.category in REPLACED_CATEGORIES:
            old_name = obj.name
            old_category = obj.category
            new_category_map = REPLACED_CATEGORIES[old_category]
            
            contained_rooms = set("_".join(obj.in_rooms[0].split("_")[0:-1]) for r in obj.in_rooms)

            applies_to_all_rooms = False
            applies_to_some_rooms = False
            if isinstance(new_category_map, dict):
                # category switch applies only to selected rooms
                applies_to_some_rooms = set(new_category_map.keys()).intersection(contained_rooms)
                if applies_to_some_rooms:
                    room_type = list(applies_to_some_rooms)[0]
                    new_category = new_category_map[room_type]
            if isinstance(new_category_map, str):
                # category switch applies to all rooms
                applies_to_all_rooms = True
                new_category = new_category_map

            if applies_to_all_rooms or applies_to_some_rooms:

                old_number = int(obj.name.split("_")[-1])
                new_number = 1000 + old_number
                new_name = obj.name.replace(old_category, new_category).replace(str(old_number), str(new_number))
                
                obj.category = new_category
                obj.name = new_name
                obj.orig_name = old_name
                obj.orig_class_id = obj.class_id
                obj.class_id = CLASS_NAME_TO_CLASS_ID[new_category]
                # pybullet-names of the joints keep using the old name -> make sure we still refer to them with that name
                obj.get_prefixed_joint_name = partial(get_prefixed_joint_name, self=obj)
                obj.get_prefixed_link_name = partial(get_prefixed_link_name, self=obj)
                self.object_states[obj.name] = self.object_states[old_name]

        return self._orig_add_object(obj)