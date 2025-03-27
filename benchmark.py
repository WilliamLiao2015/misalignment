import argparse
import os
import pickle
import random
import sys
import time
import yaml

from typing import Optional

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "lctgen"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from shapely.geometry import LineString
from shapely.ops import linemerge
from trajdata.data_structures import StateArray
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import RoadLane, Polyline

from trafficgen.utils.typedef import *
from lctgen.config.default import get_config
from lctgen.datasets.utils import fc_collate_fn
from lctgen.inference.utils import output_formating_cot, vis_decode
from lctgen.models import LCTGen

from benchmark.lateral import is_turning_left, is_turning_right
from benchmark.longitudinal import is_accelerating, is_cruising, is_decelerating, is_standing_still
from benchmark.road_structure import get_offroad_ratios, has_continuous_path
from dataset import LCTGenBaseDataset
from llm import LCTGenBaseLLM
from utils.get_possible_activities import get_possible_activities

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

type_map = {
    "longitudinal:driving-forward:accelerating": "Accelerating",
    "longitudinal:driving-forward:cruising": "Cruising",
    "longitudinal:driving-forward:decelerating": "Decelerating",
    "longitudinal:standing-still": "Standing Still",
    "lateral:turning:right": "Turning Right",
    "lateral:turning:left": "Turning Left"
}

test_map = {
    "longitudinal:driving-forward:accelerating": is_accelerating,
    "longitudinal:driving-forward:cruising": is_cruising,
    "longitudinal:driving-forward:decelerating": is_decelerating,
    "longitudinal:standing-still": is_standing_still,
    "lateral:turning:right": is_turning_right,
    "lateral:turning:left": is_turning_left
}

def describe_for_lctgen(config: dict, combine=True) -> Optional[dict]:
    description = ""

    participants_name_map = {participant: f"V{i + 1}" for i, participant in enumerate(set([participant for activity in config["activities"] for participant in activity["participants"]]))}

    if not combine:
        for activity in config["activities"]:
            description += f"{' and '.join(participants_name_map[activity['participants']])} {'is' if len(participants_name_map[activity['participants']]) == 1 else 'are'} {type_map[activity['type']].lower()}, "
        description = description[:-2]
    else:
        single_participant_activities = sorted([activity for activity in config["activities"] if len(activity["participants"]) == 1], key=lambda x: participants_name_map[x["participants"][0]])
        multi_participant_activities = [activity for activity in config["activities"] if len(activity["participants"]) > 1]

        current_participant = None
        for activity in single_participant_activities:
            if current_participant != participants_name_map[activity["participants"][0]]:
                if current_participant is not None: description += ", "
                description += f"{participants_name_map[activity['participants'][0]]} is {type_map[activity['type']].lower()}"
            else: description += f" and {type_map[activity['type']].lower()}"
            current_participant = participants_name_map[activity["participants"][0]]
        for activity in multi_participant_activities:
            description += f"{' and '.join(participants_name_map[activity['participants']])} are {type_map[activity['type']].lower()}, "
        if len(multi_participant_activities): description = description[:-2]

    if description == "": return None

    config["description"] = description + "."
    config["participant_name_map"] = participants_name_map

    return config

def add_test_results(config: dict, trajectories: np.ndarray):
    for i, activity in enumerate(config["activities"]):
        indices = [int(config["participant_name_map"][participant][1:]) - 1 for participant in activity["participants"]] # remove the "V" prefix and subtract 1

        action = activity["type"]
        if action not in test_map: continue

        try:
            print(f"Running test for activity: {action}", indices)
            print([test_map[action](vector_map, StateArray.from_array(trajectories[index], "x,y")) for index in indices])
            states = [StateArray.from_array(trajectories[index], "x,y") for index in indices]
            config["activities"][i]["results"] = [bool(test_map[action](vector_map, state)) for state in states]
            config["activities"][i]["offroad_ratios"] = [get_offroad_ratios(vector_map, state) for state in states]
            config["activities"][i]["has_continuous_path"] = [has_continuous_path(vector_map, state) for state in states]
        except:
            config["activities"][i]["results"] = [None] * len(indices)
            config["activities"][i]["offroad_ratios"] = [None] * len(indices)
            config["activities"][i]["has_continuous_path"] = [None] * len(indices)

    return config

def generate_scenario(model, llm, config, batch):
    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32

    config = describe_for_lctgen(config)
    print(f"Running scenario generation based on text: \"{config['description']}\"")

    llm_text = llm.forward(config["description"])
    agent_vector, map_vector = output_formating_cot(llm_text) if isinstance(llm_text, str) else llm_text

    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)

    # inference with LLM-output Structured Representation
    batch["text"] = torch.tensor(agent_vector, dtype=batch["text"].dtype)[None, ...]
    batch["agent_mask"] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), dtype=batch["agent_mask"].dtype)[None, ...]

    model_output = model.forward(batch, "val")["text_decode_output"]
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    return output_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a scenario from the given text.")
    parser.add_argument("--num_configs", type=int, help="The number of configurations to generate.", default=1)
    parser.add_argument("--save_image", action="store_true", help="Save the generated image.", default=True)
    parser.add_argument("--llm_base_url", type=str, help="The base URL for the LLM API.")
    parser.add_argument("--llm_model", type=str, help="The model to use for the LLM API.")
    parser.add_argument("--llm_api_key", type=str, help="The API key for the LLM API.")
    parser.add_argument("--required_activities", help="The activities that must be included in the generated scenario.", nargs="*", default=[])
    parser.add_argument("--num_activities", type=int, help="The number of activities to include in the generated scenario.", default=5)
    args = parser.parse_args()

    try:
        os.environ["LLM_BASE_URL"] = args.llm_base_url
        os.environ["LLM_MODEL"] = args.llm_model
        os.environ["LLM_API_KEY"] = args.llm_api_key
    except: pass

    try:
        cfg = get_config(os.path.join(folder, "lctgen/lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml"))
        llm = LCTGenBaseLLM(cfg, base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY"), model=os.environ.get("LLM_MODEL"))

        cfg = get_config(os.path.join(folder, "configs/lctgen.yaml"))
        model = LCTGen.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
        model.eval()

        cfg["DATASET"]["CACHE"] = False
        dataset = LCTGenBaseDataset(cfg, "train")

        success_count = 0

        while success_count < args.num_configs:
            batch = random.choices(dataset, k=1)

            for data in batch:
                with open(os.path.join(os.path.dirname(__file__), "data/processed", data["file"].strip()), "rb") as fp:
                    scenario = pickle.load(fp)

                vector_map = VectorMap(map_id="waymo_val:temp")
                vector_map.extent = [np.inf, np.inf, 0, 0, 0, 0]
                lanes = {}
                for i, center_id in enumerate(data["center_id"].squeeze().tolist()):
                    if center_id == 0: break
                    try:
                        current_lane = scenario["center_info"][center_id]
                        x1, y1, x2, y2 = data["center"][i, :4]
                        vector_map.extent = [
                            min(vector_map.extent[0], np.min([x1, x2])),
                            min(vector_map.extent[1], np.min([y1, y2])),
                            0,
                            max(vector_map.extent[3], np.max([x1, x2])),
                            max(vector_map.extent[4], np.max([y1, y2])),
                            0
                        ]

                        # print(f"Adding lane {center_id} with center ({x1}, {y1}) to ({x2}, {y2})")

                        if center_id not in lanes:
                            lanes[center_id] = {
                                "center": [LineString([(x1, y1), (x2, y2)])],
                                "adj_lanes_left": set([lane["id"] for lane in current_lane["left_neighbor"]]),
                                "adj_lanes_right": set([lane["id"] for lane in current_lane["right_neighbor"]]),
                                "next_lanes": set([lane_id for lane_id in current_lane["exit"]]),
                                "prev_lanes": set([lane_id for lane_id in current_lane["entry"]])
                            }
                        else: lanes[center_id]["center"].append(LineString([(x1, y1), (x2, y2)]))
                    except Exception as e:
                        raise e

                vector_map.lanes = []
                for lane_id, lane in lanes.items():
                    try:
                        center = linemerge(lane["center"])
                        if center.is_empty: continue
                        x, y = center.xy
                        lane = RoadLane(
                            id=lane_id,
                            center=Polyline(np.asarray([x, y]).T),
                            adj_lanes_left=lane["adj_lanes_left"],
                            adj_lanes_right=lane["adj_lanes_right"],
                            next_lanes=lane["next_lanes"],
                            prev_lanes=lane["prev_lanes"]
                        )
                        vector_map.add_map_element(lane)
                        vector_map.lanes.append(lane)
                    except: continue

                try: vector_map.compute_search_indices()
                except: continue

                # config = {"tests": {}}
                # for test_name, test_func in tests.items():
                #     agents = [StateArray.from_array(traj, "x,y") for traj in data["agent_abs"].swapaxes(0, 1)]
                #     config["tests"][test_name] = [test_func(vector_map, agent) for agent in agents]
                # configs.append(config)

                # activities = [{
                #     "type": test_type,
                #     "participants": [f"V{i + 1}"]
                # }  for test_type, results in config["tests"].items() for i, result in enumerate(results) if result is not None and result[0] < 15 and test_type != "longitudinal:standing-still"]

                activities = get_possible_activities(vector_map, [StateArray.from_array(agent[:2], "x,y") for agent in data["gt_pos"].swapaxes(0, 1)])

                if len(activities) == 0: continue
                # print(activities)

                config = {"activities": []}
                if len(args.required_activities) > 0:
                    for activity_type in args.required_activities:
                        for activity in activities:
                            if activity_type in activity["type"]:
                                config["activities"].append(activity)
                                break
                    if len(config["activities"]) < len(args.required_activities):
                        print(f"Failed to find all required activities: {args.required_activities} with current map, retrying...")
                        continue
                for _ in range(min(len(activities), args.num_activities - len(config["activities"]))):
                    existing_participants = set([participant for activity in config["activities"] for participant in activity["participants"]])
                    candidates = [activity for activity in activities if activity["participants"][0] not in existing_participants]
                    if len(candidates) == 0: break
                    config["activities"].append(random.choice(candidates))
                # print(config["activities"])

                # import cv2
                # import matplotlib.pyplot as plt

                # image = vector_map.rasterize(resolution=10, incl_lane_area=False)
                # alpha = (~np.all(image == [0, 0, 0], axis=-1) * 255).astype(np.uint8)
                # image = np.dstack((image * 255, alpha)).astype(np.uint8)
                # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                # image = np.flipud(image)

                # plt.imshow(image)
                # plt.show()

                # for agent in scene["all_agent"].swapaxes(0, 1):
                #     agent = agent[:, :2]
                #     agent = agent[agent != [0, 0]].reshape(-1, 2)
                #     if len(agent) == 0: continue
                #     plt.plot(agent[:, 0], agent[:, 1], color="blue", linewidth=1.5)
                #     plt.scatter(agent[-1, 0], agent[-1, 1], color="blue", s=2)
                # print(scene["all_agent"].swapaxes(0, 1).shape)

                time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                folder = os.path.join(os.path.dirname(__file__), "data/results", os.environ["LLM_MODEL"], f"{time_str}")

                if not os.path.exists(folder):
                    os.makedirs(folder)

                if args.save_image:
                    plt.figure(figsize=(10, 10))
                    plt.axis("equal")
                    plt.axis("off")
                    for lane in vector_map.iter_elems():
                        plt.plot(*zip(*lane.center.xy), color="black", linewidth=0.5)
                    for i, agent in enumerate(data["gt_pos"].swapaxes(0, 1)):
                        # agent = agent[:, :2]
                        # agent = agent[agent != [0, 0]].reshape(-1, 2)
                        # if len(agent) == 0: continue
                        # plt.plot(agent[:, 0], agent[:, 1], color="red", linewidth=1)
                        plt.scatter(agent[0, 0], agent[0, 1], color="red", s=2)
                        plt.text(agent[0, 0], agent[0, 1], f"V{i + 1}", fontsize=8, color="red")
                    plt.xlim(-60, 60)
                    plt.ylim(-60, 60)
                    plt.savefig(os.path.join(folder, "original-scenario.png"), bbox_inches="tight", pad_inches=0, dpi=300)
                    plt.close()

                batch = fc_collate_fn(batch)
                scenario = generate_scenario(model, llm, config, batch)
                trajectories = scenario[0]["traj"].swapaxes(0, 1)
                config = add_test_results(config, trajectories)
                config["trajectories"] = trajectories.tolist()

                if args.save_image:
                    image = vis_decode(batch, scenario)

                    with open(os.path.join(folder, "lctgen-scenario.png"), "wb") as fp:
                        image.save(fp)

                with open(os.path.join(folder, "config.yaml"), "w") as fp:
                    yaml.dump(config, fp)

                success_count += 1
    except Exception as e:
        raise e
