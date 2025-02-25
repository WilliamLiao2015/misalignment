import argparse
import os
import sys

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../lctgen"))

import openai
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pydantic import BaseModel
from typing import List, Tuple
from trajdata.maps import VectorMap

from trafficgen.utils.typedef import *
from lctgen.models import LCTGen
from lctgen.config.default import get_config
from lctgen.core.basic import BasicLLM
from lctgen.inference.utils import output_formating_cot, map_retrival, load_all_map_vectors, get_map_data_batch
from lctgen.models.utils import visualize_input_seq

class AgentVector(BaseModel):
    position_negative_for_ego: int
    distance_from_ego: int
    direction_wrt_ego: int
    speed_interval: int
    action_in_next_4s: Tuple[int, int, int, int]

class MapVector(BaseModel):
    parallel_lane_counts: Tuple[int, int]
    perpendicular_lane_counts: Tuple[int, int]
    distance_to_intersection: int
    lane_id: int

class LCTGenStructuredRepresentation(BaseModel):
    agents: List[AgentVector]
    map: MapVector

class OpenAIModel(BasicLLM):
    def __init__(self, config, base_url=None, model="gpt-4o-mini"):
        super().__init__(config)

        self.base_url = base_url
        self.model = model

        self.client = openai.Client(base_url=base_url, api_key="ollama" if base_url is not None else None)

        self.codex_cfg = config.LLM.CODEX
        folder = os.path.join(os.path.dirname(__file__), "../lctgen/lctgen/gpt")
        prompt_path = os.path.join(folder, "prompts", self.codex_cfg.PROMPT_FILE)
        self.base_prompt = open(prompt_path).read().strip()

        sys_prompt_file = self.codex_cfg.SYS_PROMPT_FILE
        if sys_prompt_file:
            sys_prompt_path = os.path.join(folder, "prompts", sys_prompt_file)
            self.sys_prompt = open(sys_prompt_path).read().strip()
        else:
            self.sys_prompt = "Only answer with a function starting def execute_command."

    def prepare_prompt(self, query, base_prompt):
        extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", query)
        return extended_prompt

    def llm_query(self, extended_prompt):
        if self.codex_cfg.MODEL == "debug":
            resp = self.sys_prompt
        elif self.model.startswith("gpt-"):
            responses = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": extended_prompt}
                ],
                temperature=self.codex_cfg.TEMPERATURE,
                max_tokens=self.codex_cfg.MAX_TOKENS,
                top_p = 1.,
                frequency_penalty=0,
                presence_penalty=0,
            )
            resp = responses["choices"][0]["message"]["content"]
        else:
            responses = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": extended_prompt}
                ],
                temperature=self.codex_cfg.TEMPERATURE,
                max_tokens=-1,
                top_p = 1.,
                frequency_penalty=0,
                presence_penalty=0,
                response_format=LCTGenStructuredRepresentation
            )
            representation = responses.choices[0].message.parsed
            agents = [[agent.position_negative_for_ego, agent.distance_from_ego, agent.direction_wrt_ego, agent.speed_interval, *agent.action_in_next_4s] for agent in representation.agents]
            map = [*representation.map.parallel_lane_counts, *representation.map.perpendicular_lane_counts, representation.map.distance_to_intersection, representation.map.lane_id]
            resp = (agents, map)

        return resp

    def post_process(self, response):
        return response

def vis_decode(batch, ae_output):
    img = visualize_input_seq(batch, agents=ae_output[0]["agent"], traj=ae_output[0]["traj"])
    return Image.fromarray(img)

def gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids):
    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector = output_formating_cot(llm_text) if isinstance(llm_text, str) else llm_text

    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)

    # retrive map from map dataset
    sorted_idx = map_retrival(map_vector, map_vecs)[:1]
    map_id = map_ids[sorted_idx[0]]

    # load map data
    batch = get_map_data_batch(map_id, cfg)

    # inference with LLM-output Structured Representation
    batch["text"] = torch.tensor(agent_vector, dtype=batch["text"].dtype)[None, ...]
    batch["agent_mask"] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), dtype=batch["agent_mask"].dtype)[None, ...]

    model_output = model.forward(batch, "val")["text_decode_output"]
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    return vis_decode(batch, output_scene)

def down_sampling(line, type=0):
    SAMPLE_NUM = 10

    # if is center lane
    point_num = len(line)

    ret = []

    if point_num < SAMPLE_NUM or type == 1:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, SAMPLE_NUM):
            ret.append(line[i])

    return ret

def generate_scenario(query: str, vector_map: VectorMap, log=True):
    cfg = get_config(os.path.join(folder, "../lctgen/lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml"))
    llm = OpenAIModel(cfg, base_url="http://localhost:11434/v1", model="llama3.1")
    llm_text = llm.forward(query)

    cfg = get_config(os.path.join(folder, "../configs/lctgen.yaml"))
    model = LCTGen.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
    model.eval()

    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector = output_formating_cot(llm_text) if isinstance(llm_text, str) else llm_text

    if log: print(agent_vector)

    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)

    batch = {
        "center": [],
        "bound": [],
        "rest": torch.empty(0).unsqueeze(0),
    }

    for lane in vector_map.lanes:
        center = down_sampling(lane.center.xy)
        left_edge = down_sampling(lane.left_edge.xy) if lane.left_edge is not None else None
        right_edge = down_sampling(lane.right_edge.xy) if lane.right_edge is not None else None

        for p1, p2 in zip(center[:-1], center[1:]):
            batch["center"].append([p1[0], p1[1], p2[0], p2[1], 1, 0]) # center type is 1
        if left_edge is not None:
            for p1, p2 in zip(left_edge[:-1], left_edge[1:]):
                batch["bound"].append([p1[0], p1[1], p2[0], p2[1], 15, 0]) # bound type is 15
        if right_edge is not None:
            for p1, p2 in zip(right_edge[:-1], right_edge[1:]):
                batch["bound"].append([p1[0], p1[1], p2[0], p2[1], 15, 0]) # bound type is 15

    batch["center"] = torch.tensor(batch["center"], dtype=torch.float32).unsqueeze(0)
    batch["bound"] = torch.tensor(batch["bound"], dtype=torch.float32).unsqueeze(0)
    batch["center_mask"] = torch.ones(batch["center"].shape[:-1], dtype=torch.bool)
    batch["bound_mask"] = torch.ones(batch["bound"].shape[:-1], dtype=torch.bool)
    batch["lane_inp"] = torch.cat([batch["center"], batch["bound"]], dim=1)
    batch["lane_mask"] = torch.cat([batch["center_mask"], batch["bound_mask"]], dim=1)

    # inference with LLM-output Structured Representation
    batch["text"] = torch.tensor(agent_vector, dtype=torch.float32).unsqueeze(0)
    batch["agent"] = torch.tensor(agent_vector, dtype=torch.float32).unsqueeze(0)
    batch["agent_mask"] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), dtype=torch.bool).unsqueeze(0)
    batch["file"] = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
    batch["center_id"] = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
    batch["agent_vec_index"] = torch.tensor([0], dtype=torch.int64).unsqueeze(0)

    model_output = model.trafficgen_model(batch)
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    return output_scene[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scenarios.")
    parser.add_argument("--map", type=str, help="Map file.")
    parser.add_argument("--config", type=str, help="Configuration file.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")

    args = parser.parse_args()

    cfg = get_config(os.path.join(folder, "../lctgen/lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml"))
    llm = OpenAIModel(cfg, base_url="http://localhost:11434/v1", model="llama3.1")
    llm_text = llm.forward("V2 is striking V1 from behind. V1 is going straight and cruising slowly, V2 is going straight and accelerating. Two vehicles, on a highway.")
    print(llm_text)

    cfg = get_config(os.path.join(folder, "../configs/lctgen.yaml"))
    model = LCTGen.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
    model.eval()

    map_data_file = os.path.join(folder, "../lctgen/data/demo/waymo/demo_map_vec.npy")
    map_vecs, map_ids = load_all_map_vectors(map_data_file)

    scene = gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids)
    plt.switch_backend("TkAgg")
    plt.imshow(scene)
    plt.show()
