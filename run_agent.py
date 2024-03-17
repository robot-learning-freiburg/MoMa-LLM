# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
# NOTE: monkey-patching; needs to be imported before any other file imports it
from moma_llm.tasks.patched_scene import MonkeyPatchedInteractiveIndoorScene
from igibson.scenes import igibson_indoor_scene
igibson_indoor_scene.InteractiveIndoorScene._add_object = MonkeyPatchedInteractiveIndoorScene._add_object
igibson_indoor_scene.InteractiveIndoorScene._orig_add_object = MonkeyPatchedInteractiveIndoorScene._orig_add_object

import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from igibson.utils.utils import parse_config
from sklearn.metrics import auc
import wandb

from moma_llm.env.baselines import (GreedyBaseline,
                                     RandomBaseline)
from moma_llm.env.env import OurIGibsonEnv, create_igibson_env
from moma_llm.env.llm_env import JsonLLMEnv, LLMEnv
from moma_llm.llm.llm import LLM
from moma_llm.utils.constants import TEST_SCENES, TRAINING_SCENES
from moma_llm.utils.utils import get_config


def create_env(cfg, agent: str, config_file: str, scene_id: str, control_freq: float, cheap: bool, seed: int) -> LLMEnv:
    if agent == "moma_llm":
        env_fn = LLMEnv
    elif agent == "json_llm":
        env_fn = JsonLLMEnv
    elif agent == "greedy":
        env_fn = GreedyBaseline
    elif agent == "random":
        env_fn = RandomBaseline
    else:
        raise ValueError(f"Unknown agent {agent}")

    llm_variant = "gpt-3.5-turbo" if cheap else "gpt-4-1106-preview"  # "gpt-4"
    llm = LLM(debug=True, model=llm_variant, room_classification_model="gpt-3.5-turbo-1106", open_set_rooms=cfg["open_set_room_categories"]) 

    low_level_env = create_igibson_env(config_file=config_file, 
                                       control_freq=control_freq, 
                                       scene_id=scene_id,
                                       seed=seed)
    high_level_env = env_fn(env=low_level_env, llm=llm, seed=seed)  
    return high_level_env


def calc_area_under_curve(x, y, max_x):
    if max(x) > max_x:
        idx = (x <= max_x)
        x = x[idx]
        y = y[idx]
    if max(x) < max_x:
        x = np.concatenate([x, [max_x]])
        y = np.concatenate([y, [y[-1]]])
    
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    return auc(x, y) / max_x
    

def plot_efficiency_curves(episode_infos, max_hl_steps: int):
    ll_steps = []
    ll_steps_gtDone = []
    hl_steps = []
    task_success = []
    task_success_gtDone = []
    for scene_id in sorted(episode_infos.keys()):
        for e in episode_infos[scene_id]:
            ll_steps.append(e["num_low_level_steps_with_open_cost"])
            ll_steps_gtDone.append(e["num_low_level_steps_with_open_cost_gtDone"])
            hl_steps.append(e["num_high_level_steps"])
            task_success.append(e["task_success"])
            task_success_gtDone.append(e["task_success_gtDone"])
    task_success = np.array(task_success)
    task_success_gtDone = np.array(task_success_gtDone)
    ll_steps = np.array(ll_steps)
    hl_steps = np.array(hl_steps)
    
    def _plot(steps, task_success):
        df = pd.DataFrame({"steps": steps, "task_success": task_success})
        df = df.sort_values("steps")
        
        values = [np.logical_and(df["task_success"].values, df["steps"].values <= max_steps).mean() for max_steps in df["steps"]]
        df2 = pd.DataFrame({"steps": df["steps"].values, "success": values})
        return wandb.Table(dataframe=df2)
    
    def _get_auc(steps, task_success, max_x: int, title: str):
        table = _plot(steps=steps, task_success=task_success)
        auc = calc_area_under_curve(table.get_dataframe()["steps"].values, table.get_dataframe()["success"].values, max_x=max_x)
        wandb_plot = wandb.plot_table("wandb/area-under-curve/v0",
                                      table,
                                      {"x": "steps", "y": "success"},
                                      {"title": title,
                                          "x-axis-title": "Steps",
                                          "y-axis-title": "Success rate",},)
        return auc, wandb_plot
    
    hl_auc, hl_plot = _get_auc(steps=hl_steps, task_success=task_success, max_x=max_hl_steps, title="High-level-step-curve")
    wandb.log({"efficiency_curve_high_level_steps": hl_plot, "hl_auc": hl_auc})
    
    ll_auc_max_steps = 5000
    ll_auc, ll_plot = _get_auc(steps=ll_steps, task_success=task_success, max_x=ll_auc_max_steps, title="Low-level-step-curve")
    wandb.log({"efficiency_curve_low_level_steps": ll_plot, "ll_auc": ll_auc})
        
    ll_auc_gtDone, ll_plot_gtDone = _get_auc(steps=ll_steps_gtDone, task_success=task_success_gtDone, max_x=ll_auc_max_steps, title="Low-level-step-curve-gtDone")
    wandb.log({"efficiency_curve_low_level_steps_gtDone": ll_plot_gtDone, "ll_auc_gtDone": ll_auc_gtDone}) 
   

def calculcate_metric_means(episode_infos):
    columns = sorted(list(episode_infos.values())[0][0].keys())
    scene_logs = defaultdict(dict)
    for scene_id in sorted(episode_infos.keys()):
        for column in columns:
            if isinstance(episode_infos[scene_id][0].get(column, None), str):
                continue
            elif isinstance(episode_infos[scene_id][0].get(column, None), (np.ScalarType)):
                d = np.nanmean([e.get(column, np.nan) for e in episode_infos[scene_id]])
                # NOTE: if value is not defined for an episode, we take the mean over those that have the value!
                #   e,g, [steps]_gtDone are only defined for successful episodes
                scene_logs[scene_id][column] = d
            else:
                continue
            print(column, d) 
    return scene_logs


def log_summary_table(episode_infos):
    def _check_float(v):
        try:
            float(v)
            return True
        except:
            return False
    
    scene_logs = calculcate_metric_means(episode_infos)
    data = []
    scenes = list(scene_logs.keys())
    columns = ["scene_id"]
    for k in scene_logs[scenes[0]].keys():
        # only take metrics that exist for all scenes
        if all([k in scene_logs[s] for s in scenes]):
            columns.append(k)
    
    for scene_id in sorted(scenes):
        data.append([scene_id] + [scene_logs[scene_id][c] for c in columns[1:]])
        
    avg_row = ["Overall avg"]
    avg_dict = {}
    for i, column_values in enumerate(np.array(data).T):
        if _check_float(column_values[0]):
            d = np.mean(column_values.astype(float))  
            avg_dict[f"avg_{columns[i]}"] = d
            avg_row.append(str(d))
    data.append(avg_row)
    avg_dict["overview_table"] = wandb.Table(columns=columns, data=np.array(data).astype(str))
    wandb.log(avg_dict)
    
    
def evaluate_scene(config_file: str, cfg, scene_id: str, tot_ep: int) -> list:
    episode_infos = []
    high_level_env = create_env(cfg, agent=cfg["agent"], config_file=config_file, scene_id=scene_id, control_freq=cfg["control_freq"], cheap=cfg["cheap"], seed=cfg["seed"])
    for i in range(cfg["num_episodes_per_scene"]):
        done = False
        obs = high_level_env.reset(config_file=config_file, scene_id=scene_id, episode_num=i)
        print("########################################")
        print(f"{scene_id} - Starting episode {i + 1} in scene {scene_id}, {tot_ep + 1} overall. Task: {high_level_env.unwrapped.task.task_description}")
        print("########################################")
        while not done:
            high_level_env.visualize(obs)
            done, task_success, episode_info = high_level_env.take_action(obs=obs, task_description=high_level_env.unwrapped.task.task_description)
            # env adds last action to the figure title, that's why we log it after the env step
            wandb.log({"bev_maps": high_level_env.unwrapped.f})
            obs = high_level_env.get_state(compute_scene_graph=True)
            pprint(episode_info)
            
        high_level_env.visualize(obs)
        if "failure_reason" in episode_info:
            high_level_env.env.f.suptitle(f"{high_level_env.env.f._suptitle.get_text()}, {episode_info['failure_reason']}")
        episode_info["bev_maps"] = high_level_env.unwrapped.f
        episode_info["num_low_level_steps_with_open_cost"] = episode_info["num_low_level_steps"] + high_level_env.env.config["magic_open_cost"] * episode_info["magic_open_actions"]
        if episode_info.get("num_low_level_steps_gtDone", None) is not None:
            episode_info["num_low_level_steps_with_open_cost_gtDone"] = episode_info["num_low_level_steps_gtDone"] + high_level_env.env.config["magic_open_cost"] * episode_info["magic_open_actions_gtDone"]
            episode_info["task_success_gtDone"] = True
        else:
            episode_info["num_low_level_steps_with_open_cost_gtDone"] = episode_info["num_low_level_steps_with_open_cost"]
            episode_info["task_success_gtDone"] = task_success
        episode_info["episode_step"] = tot_ep
        episode_info["spl"] = episode_info["task_success"] * (episode_info["shortest_dist"] / max(episode_info["shortest_dist"], episode_info["dist_travelled"]))
        pprint(episode_info)
        # episode_info["rgb"] = wandb.Video((255 * np.transpose(np.stack(high_level_env.env.rgb_frames, axis=0), (0, 3, 1, 2))).astype(np.uint8), fps=6)
        wandb.log({k: float(v) if isinstance(v, bool) else v for k, v in episode_info.items()})

        episode_infos.append(episode_info)
        successes = [e["task_success"] for e in episode_infos]
        tot_ep += 1
        print(f"Task success: {task_success} (wandb_step: {wandb.run.step}). Current successes: {sum(successes)}/{len(successes)}")
    
    scene_logs = calculcate_metric_means({scene_id: episode_infos})
    wandb.log({f"{scene_id}_{k}": v for k, v in scene_logs[scene_id].items()})
    
    high_level_env.close()
    return episode_infos, tot_ep
    
    
def main():
    np.set_printoptions(precision=3, suppress=True)
       
    config_file = get_config("moma_llm.yaml")
    # NOTE: igibson will reload the config file, so changes here won't be relfected! Just for wandb logging
    cfg = parse_config(config_file)
    if cfg["seed"] > 0:
        np.random.seed(cfg["seed"])

    if cfg["datasplit"] == "train":
        scene_ids = TRAINING_SCENES
    elif cfg["datasplit"] == "test":
        scene_ids = TEST_SCENES
    else:
        raise ValueError(f"Unknown datasplit {cfg['datasplit']}")

    cfg.update({"scene_ids": scene_ids, "agent": cfg["agent"]})
    wandb.init(project="[scene-llm]", 
               entity="robot-learning-lab", 
               config=cfg,
               mode="online" if cfg["wandb"] else "disabled",
               #name=f"{agent}"
               )
    # copy config file to wandb run dir, so modifications to the main config file won't affect current runs
    new_config_file = Path(wandb.run.dir) / Path(config_file).name
    shutil.copy(config_file, new_config_file)
    config_file = str(new_config_file)

    episode_infos = defaultdict(list)

    tot_ep = 0
    if isinstance(scene_ids, str):
        scene_ids = [scene_ids]
    for scene_id in scene_ids:
        infos, tot_ep = evaluate_scene(config_file=config_file, cfg=cfg, scene_id=scene_id, tot_ep=tot_ep)
        episode_infos[scene_id] = infos
    log_summary_table(episode_infos=episode_infos)
    plot_efficiency_curves(episode_infos=episode_infos, max_hl_steps=cfg["max_high_level_steps"])
    
    wandb.run.finish()
    print("Done!")


if __name__ == "__main__":
    main()
