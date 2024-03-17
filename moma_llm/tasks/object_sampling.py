# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum, IntEnum
from pathlib import Path

import igibson
import numpy as np
import pybullet as p
from bddl.object_taxonomy import ObjectTaxonomy
from cachetools import cached
from cachetools.keys import hashkey
from igibson import object_states
from igibson.external.pybullet_tools.utils import set_base_values_with_z
from igibson.object_states.utils import clear_cached_states, sample_kinematics
from igibson.objects.articulated_object import URDFObject
from igibson.utils.assets_utils import (get_ig_category_path,
                                        get_ig_model_path, get_ig_scene_path)
from igibson.utils.utils import restoreState
from IPython import embed

from moma_llm.llm.sbert import SentenceBERT
from moma_llm.tasks.patched_scene import REPLACED_CATEGORIES
from moma_llm.utils.constants import PACKAGE_DIR, TEST_SCENES, TRAINING_SCENES


def load_object_distribution_prior():
    # load object placement json
    obj_placement_file = os.path.join(PACKAGE_DIR, "configs", "object_placement.json")
    obj_placement_ig_pt = json.load(open(obj_placement_file, "r"))
    object_distribution_prior = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    # go through all relation keys and replace with object states
    for room_type, furniture_dict in obj_placement_ig_pt.items():
            for furniture, obj_dict in furniture_dict.items():
                for obj, relation_dict in obj_dict.items():
                    for relation, prob in relation_dict.items():
                        if relation == "ON_TOP":
                            object_distribution_prior[room_type][furniture][object_states.OnTop][obj] = obj_placement_ig_pt[room_type][furniture][obj][relation]
                        elif relation == "INSIDE":
                            object_distribution_prior[room_type][furniture][object_states.Inside][obj] = obj_placement_ig_pt[room_type][furniture][obj][relation]
                        elif relation == "UNDER":
                            object_distribution_prior[room_type][furniture][object_states.Under][obj] = obj_placement_ig_pt[room_type][furniture][obj][relation]
                        else:    
                            raise NotImplementedError(f"Relation {relation} not implemented")
    return object_distribution_prior


OBJECT_DISTRIBUTION_PRIOR = load_object_distribution_prior()


def create_new_object(env, scene, category, obj_number, in_rooms=None, rendering_params=None, scale=None, bounding_box=None):
    """Based on InteractiveIndoorScene.__init__"""
    
    object_name = f"{category}_{obj_number}"
    model = "random"

    # Non-robot object
    # Do not load these object categories (can blacklist building structures as well)
    if scene.not_load_object_categories is not None and category in scene.not_load_object_categories:
        return

    # An object can in multiple rooms, seperated by commas,
    # or None if the object is one of the walls, floors or ceilings
    # in_rooms = link.attrib.get("room", None)
    if in_rooms is not None:
        in_rooms = in_rooms.split(",")

    # This object is not located in one of the selected rooms, skip
    if scene.load_room_instances is not None and len(set(scene.load_room_instances) & set(in_rooms)) == 0:
        return

    category_path = get_ig_category_path(category)
    assert len(os.listdir(category_path)) != 0, "No models in category folder {}".format(category_path)

    if model == "random":
        model = env.np_random.choice(os.listdir(category_path))

    model_path = get_ig_model_path(category, model)
    filename = os.path.join(model_path, model + ".urdf")

    if (bounding_box is not None) and (scale is not None):
        raise Exception("You cannot define both scale and bounding box size for a URDFObject")
    if scale is not None and isinstance(scale, float):
        scale = np.array([scale, scale, scale])
    elif bounding_box is not None:
        bounding_box = np.array(bounding_box)
    else:
        scale = np.array([1.0, 1.0, 1.0])

    bddl_object_scope = None
    fixed_base = False

    obj = URDFObject(filename,
                     name=object_name,
                     category=category,
                     model_path=model_path,
                     bounding_box=bounding_box,
                     scale=scale,
                     fixed_base=fixed_base,
                     avg_obj_dims=scene.avg_obj_dims[category],
                     in_rooms=in_rooms,
                     texture_randomization=scene.texture_randomization,
                     overwrite_inertial=True,
                     scene_instance_folder=scene.scene_instance_folder,
                     bddl_object_scope=bddl_object_scope,
                     merge_fixed_links=scene.merge_fixed_links,
                     rendering_params=rendering_params,
                     fit_avg_dim_volume=True)
    env.simulator.import_object(obj)
    return obj


def match_furniture(given_furniture, spawnable_object_distribution):
    # match furniture objects from the prior graph to igibson furniture objects - mix of sbert and human judgement
    object_taxonomy = ObjectTaxonomy()
    all_igibson_furniture = [ic for c in object_taxonomy.get_descendants('furniture.n.01') for ic in object_taxonomy.get_igibson_categories(c)]
    all_igibson_furniture += ["shelf", "shelving_unit", "counter_top", "kitchen_counter"]
    for k, v in REPLACED_CATEGORIES.items():
        if isinstance(v, dict):
            v = list(v.values())
        else:
            v = [v]
        all_igibson_furniture += v
    all_igibson_furniture = list(set(all_igibson_furniture))
        
    furniture_mapping = get_sbert_matching(given_furniture, all_igibson_furniture, score_thresh=0.6, closest_k=50)
    furniture_mapping["shelf"].append("shelving_unit")
    furniture_mapping["shelving_unit"].append("shelf")
    furniture_mapping["dresser"].append("bottom_cabinet_no_top")
    furniture_mapping["dresser"].append("bottom_cabinet")
    furniture_mapping["dresser"].append("top_cabinet")
    furniture_mapping["top_cabinet"].append("dresser")

    for room_type, furniture_dict in OBJECT_DISTRIBUTION_PRIOR.items():
        for furniture, relation_dict in furniture_dict.items():
            if furniture in furniture_mapping:
                for igibson_furniture in furniture_mapping[furniture]:
                    if (furniture != igibson_furniture):
                        spawnable_object_distribution[room_type][igibson_furniture].update(spawnable_object_distribution[room_type][furniture])
    return spawnable_object_distribution


def match_rooms(given_room_types, spawnable_object_distribution):
    """mapping from igibson gt room label to room types in the prior graph"""
    human_room_mapping = {"bathroom": "bathroom", 
                          "bedroom": "bedroom",
                          "childs_room": "bedroom",
                          "closet": None,
                          "corridor": None,
                          "dining_room": "living_room",
                          "empty_room": None,
                          "exercise_room": None,
                          "garage": None,
                          "home_office": None,
                          "kitchen": "kitchen",
                          "living_room": "living_room",
                          "lobby": None,
                          "pantry_room": None,
                          "playroom": None,
                          "staircase": None,
                          "storage_room": None,
                          "television_room": "living_room",
                          "utility_room": None,
                          "balcony": None,
                          "library": None,
                          "auditorium": None,
                          "undefined": None}
    inverse_room_mapping = {r: [] for r in given_room_types}
    for k, v in human_room_mapping.items():
        if v is not None:
            inverse_room_mapping[v].append(k)
    
    for room_type in given_room_types:
        for matched_room in inverse_room_mapping[room_type]:
            if (matched_room != room_type):
                spawnable_object_distribution[matched_room].update(spawnable_object_distribution[room_type])
    return spawnable_object_distribution


@cached(cache={})
def match_distribution_against_objs():
    sbert = SentenceBERT()
    # these include half fruits, walls and ceilings. So just take all objects that have a urdf
    # spawnable_objs = env.task.all_object_categories
    spawnable_objs = [p.name for p in Path(get_ig_category_path("apple")).parent.iterdir()]
    
    given_objs = set()
    given_furniture = set()
    given_room_types = set()
    for room_type, furniture_dict in OBJECT_DISTRIBUTION_PRIOR.items():
        given_room_types.add(room_type)
        for furniture, relation_dict in furniture_dict.items():
            given_furniture.add(furniture)
            for relation, relation_obj_dict in relation_dict.items():
                given_objs.update(list(relation_obj_dict.keys()))
    given_objs = list(given_objs)
    given_furniture = list(given_furniture)
    given_room_types = list(given_room_types)

    obj_sim = sbert.compute_cooccurrence_bipartite(["A " + obj for obj in spawnable_objs], ["A " + obj for obj in given_objs])
    # find the 50 closest objects for each spawnable object and threshold similarity at 0.7 (reliable)
    closest_k_given = np.argsort(obj_sim, axis=1)[:,-50:]
    spawnable_object_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    for spawnable_idx, given_idcs in enumerate(closest_k_given):
        spawnable_obj = spawnable_objs[spawnable_idx]
        for given_idx in given_idcs:
            given_obj = given_objs[given_idx]
            if (given_obj != spawnable_obj) and (obj_sim[spawnable_idx, given_idx] > 0.7):                
                # assign to all objects in the distribution
                # i.e. add the spawnable object that is matching the given_obj to all relations that contain the given_obj
                for room_type, furniture_dict in OBJECT_DISTRIBUTION_PRIOR.items():
                    for furniture, relation_dict in furniture_dict.items():
                        for relation, relation_obj_dict in relation_dict.items():
                            if given_obj in relation_obj_dict.keys():
                                spawnable_object_distribution[room_type][furniture][relation][spawnable_obj] = relation_obj_dict[given_obj]

    spawnable_object_distribution = match_furniture(given_furniture, spawnable_object_distribution)
    spawnable_object_distribution = match_rooms(given_room_types, spawnable_object_distribution)

    return spawnable_object_distribution


def relation_to_str(relation):
    return str(relation).split('.')[-1].split("'")[0]


def get_sbert_matching(categories, possible_categories, score_thresh: float = 0.0, closest_k: int = 0):
    from moma_llm.llm.sbert import SentenceBERT
    sbert = SentenceBERT()
    sim = sbert.compute_cooccurrence_bipartite(categories, possible_categories)
    closest_categories = np.argsort(sim, axis=1)[:,-closest_k:]
    
    mapping = {c: [] for c in categories}
    
    for i, close_cat_idcs in enumerate(closest_categories):
        for close_cat_idx in close_cat_idcs:
            if sim[i, close_cat_idx] > score_thresh:
                # print(categories[i], possible_categories[close_cat_idx], sim[i, close_cat_idx])
                mapping[categories[i]].append(possible_categories[close_cat_idx])
    return mapping


def fast_set_value(state, other, new_value, use_ray_casting_method=False):
    """
    Based on object_states.Inside._set_value(), which does 10x100 sampling attemps, thereby being super slow if it's not possible to spawn 
    a relation -> reduce to 1x50 sampling attempts
    """
    state_id = p.saveState()
    
    if isinstance(state, object_states.Inside):
        prefix = "inside"
    elif isinstance(state, object_states.OnTop):
        prefix = "onTop"
    else:
        raise NotImplementedError(type(state))

    for _ in range(1):
        sampling_success = sample_kinematics(prefix, state.obj, other, new_value, use_ray_casting_method=use_ray_casting_method, max_trials=50)
        if sampling_success:
            clear_cached_states(state.obj)
            clear_cached_states(other)
            if (other.category != "shelf") and (state.get_value(other) != new_value):
                sampling_success = False
            if igibson.debug_sampling:
                print("Inside checking", sampling_success)
                embed()
        if sampling_success:
            break
        else:
            restoreState(state_id)

    p.removeState(state_id)

    return sampling_success
    
    
def is_receptacle(obj_category):
    if obj_category in ["door", "window"]:
        return False
    elif obj_category in ["shelf", "shelving_unit"]:
        return True
    object_taxonomy = ObjectTaxonomy()
    n = object_taxonomy.get_class_name_from_igibson_category(obj_category)
    return n and "openable" in object_taxonomy.get_abilities(n)
    

def add_objects_from_our_distribution(env):
    env.scene.force_wakeup_scene_objects()
    
    # add all igibson objects that match the objects in the prior graph to the relations
    spawnable_object_distribution_real = match_distribution_against_objs()

    # to make object names unique
    obj_number = 10000
    new_objects = []
    new_relations = {}
    
    for existing_obj in env.scene.get_objects():
        room_type = env.scene.get_room_type_by_point(existing_obj.get_position()[:2])
        prior_relations = spawnable_object_distribution_real.get(room_type, {}).get(existing_obj.category, {})
        
        # assume objects that are on top (or below) can also be inside or vice versa
        possible_objects = set()
        for objects in prior_relations.values():
            possible_objects.update(set(objects.keys()))
        # very few objects don't have an average size defined - just ignore them
        possible_objects_wo_metadata = set(possible_objects) - set(env.scene.avg_obj_dims.keys())
        for o in possible_objects_wo_metadata:
            del possible_objects[o]
            
        possible_objects = sorted(possible_objects)
        
        if possible_objects:
            # draw a random relation uniformly
            r = set([object_states.Inside] + list(prior_relations.keys())) if is_receptacle(existing_obj.category) else list(prior_relations.keys())
            relation = env.np_random.choice(sorted(r, key=lambda x: str(x)))

            max_tries = min(5, len(possible_objects))
            i = 0
            while True:
                # sample from the distribution
                new_obj_category = env.np_random.choice(list(possible_objects))
                if (np.prod(env.scene.avg_obj_dims[new_obj_category]["size"]) > 0.9 * np.prod(env.scene.avg_obj_dims.get(existing_obj.category, {"size": np.inf})["size"])):
                    # if volume of new object is larger than 90% of the existing object, pass as we won't be able to spawn it anyway
                    possible_objects.remove(new_obj_category)
                    i += 1
                    continue 
                
                while f"{new_obj_category}_{obj_number}" in env.scene.objects_by_name.keys():
                    obj_number += 1
                
                new_obj = create_new_object(env=env, 
                                            scene=env.scene,
                                            category=new_obj_category,
                                            in_rooms=room_type, 
                                            obj_number=obj_number,
                                            scale=None,
                                            bounding_box=None)
                obj_number += 1
                try:
                    if relation in [object_states.OnTop, object_states.Inside]:
                        assert fast_set_value(state=new_obj.states[relation], other=existing_obj, new_value=True, use_ray_casting_method=True), f"{existing_obj.category} > {relation_to_str(relation)} > {new_obj_category}"
                    else:
                        assert new_obj.states[relation].set_value(existing_obj, True), f"{existing_obj.category} > {relation_to_str(relation)} > {new_obj_category}"
                    
                    new_objects.append(new_obj)
                    new_relations[new_obj.name] = (existing_obj, relation)
                    print(f"Spawned new object: {room_type}/{existing_obj.name} > {relation_to_str(relation)} > {new_obj_category}")                
                    break
                except:
                    print(f"Failed to spawn new object: {room_type}/{existing_obj.name} > {relation_to_str(relation)} > {new_obj_category}")                
                    # don't know how to delete object, so just move far away
                    set_base_values_with_z(new_obj.get_body_ids()[0], [200, 200 + (obj_number - 10000), 0.0], z=0.05)
                    env.scene.remove_object(new_obj)
                    
                    possible_objects.remove(new_obj_category)
                    i += 1
                if i >= max_tries:
                    print("Unable to spawn any object for this object/relation combo. Skipping it.")
                    break
        else:
            print(f"{room_type}/{existing_obj.name}: no possible objects to spawn")
    return new_objects, new_relations
