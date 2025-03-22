import argparse
import os
import sys

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../lctgen"))

import matplotlib.pyplot as plt
import numpy as np
import openai
import torch

from PIL import Image
from pydantic import BaseModel
from trajdata.maps import VectorMap
from typing import List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

# copied from https://stackoverflow.com/a/45669280/16082247
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with HiddenPrints():
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
    def __init__(self, config, base_url=None, api_key=None, model="gpt-4o-mini"):
        super().__init__(config)

        self.base_url = base_url
        self.model = model

        self.client = openai.Client(base_url=base_url, api_key=api_key)

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
            resp = responses.choices[0].message.content
        # elif self.model.startswith("qwq-"):
        #     responses = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=[
        #             {"role": "system", "content": self.sys_prompt},
        #             {"role": "user", "content": extended_prompt}
        #         ],
        #         max_tokens=-1
        #     )
        #     resp = responses.choices[0].message.content
        #     resp = re.sub(r"<think>.*</think>", "", resp)
        else:
            responses = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": extended_prompt}
                ],
                max_tokens=-1,
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

def gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids, save_image=False):
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

    if save_image:
        image = vis_decode(batch, output_scene)

        with open(os.path.join(os.path.dirname(__file__), "..", "data/lctgen-scenario.png"), "wb") as fp:
            image.save(fp)

    return output_scene[0]

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

def draw(center, agents, traj=None, other=None, edge=None, heat_map=False):
    """
    - V1: red (ego car)
    - V2: blue
    - V3: orange
    - V4: green
    - V5: purple
    - V6: brown
    - V7: pink
    - V8: gray
    - V9: olive
    - V10: cyan
    - V11 (V2): blue
    - V12 (V3): orange
    - V13 (V4): green
    - ...
    """
    ax = plt.gca()
    plt.axis('equal')

    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    lane_color = 'black'
    alpha = 0.12
    linewidth = 0.5

    if heat_map:
        lane_color = 'white'
        alpha = 0.2
        linewidth = 6

    ax.axis('off')

    for j in range(center.shape[0]):
        traf_state = center[j, -1]

        x0, y0, x1, y1, = center[j, :4]

        if x0 == 0: break
        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=0.5, alpha=0.2)

        if traf_state == 1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 3:
            color = 'green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)

    if edge is not None:
        for j in range(len(edge)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.5)
            # ax.arrow(x0, y0, x1-x0, y1-y0,head_width=1.5,head_length=0.75,width = 0.1)

    if other is not None:
        for j in range(len(other)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.7, alpha=0.9)

    limits = [ax.get_xlim(), ax.get_ylim()]

    for i in range(len(agents)):
        # if i in collide: continue
        if i == 0:
            col = colors[0]
        else:
            ind = (i-1) % 9 + 1
            col = colors[ind]
            if traj is not None:
                traj_i = traj[:, i]
                len_t = traj_i.shape[0] - 1
                for j in range(len_t):
                    x0, y0 = traj_i[j]
                    x1, y1 = traj_i[j + 1]

                    ax.plot((x0, x1), (y0, y1), '-', color=col, linewidth=0.3, marker='.', markersize=0.5)

                    limits[0] = [min(limits[0][0], x0), max(limits[0][1], x0)]
                    limits[1] = [min(limits[1][0], y0), max(limits[1][1], y0)]

        agent = agents[i]
        rect = agent.get_rect()[0]
        rect = plt.Polygon(rect, edgecolor='black', facecolor=col, linewidth=0.5, zorder=10000)
        ax.add_patch(rect)

    plt.xlim(limits[0])
    plt.ylim(limits[1])
    plt.autoscale(tight=True)

def generate_scene(config, query: str, save_image=True, log=False):
    cfg = get_config(os.path.join(folder, "../lctgen/lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml"))
    llm = OpenAIModel(cfg, base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY"), model=os.environ.get("LLM_MODEL"))
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

    for lane in config["vec_map"].lanes:
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
    batch["agent"] = torch.tensor(np.asarray([history[0] for history in config["history"]]), dtype=torch.float32).unsqueeze(0)
    batch["agent_mask"] = torch.tensor([1] * agent_num + [0] * (MAX_AGENT_NUM - agent_num), dtype=torch.bool).unsqueeze(0)
    batch["file"] = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
    batch["center_id"] = torch.tensor([0], dtype=torch.int64).unsqueeze(0)

    participants_name_map = {f"V{i + 1}": participant for i, participant in enumerate(set([participant for activity in config["activities"] for participant in activity["participants"]]))}
    batch["agent_vec_index"] = torch.tensor([int(participant[1:]) - 1 for participant in participants_name_map.values()], dtype=torch.int64).unsqueeze(0)

    print(participants_name_map, batch["agent_vec_index"])

    model_output = model.trafficgen_model(batch)
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    if save_image:
        draw(batch["center"][0].cpu().numpy(), output_scene[0]["agent"], traj=output_scene[0]["traj"], other=batch["rest"][0].cpu().numpy(), edge=batch["bound"][0].cpu().numpy())
        plt.savefig(os.path.join(os.path.dirname(__file__), "..", "data/lctgen-scenario.png"), bbox_inches='tight', dpi=300)
        plt.close()

    return output_scene[0]

if __name__ == "__main__":
    with HiddenPrints():
        parser = argparse.ArgumentParser(description="Generate a scenario from the given text.")
        parser.add_argument("--text", type=str, help="The text to generate the scenario from.", default="V2 is striking V1 from behind. V1 is going straight and cruising slowly, V2 is going straight and accelerating. Two vehicles, on a highway.")
        parser.add_argument("--save_image", action="store_true", help="Save the generated image.")
        parser.add_argument("--llm_base_url", type=str, help="The base URL for the LLM API.")
        parser.add_argument("--llm_model", type=str, help="The model to use for the LLM API.")
        parser.add_argument("--llm_api_key", type=str, help="The API key for the LLM API.")
        args = parser.parse_args()

        try:
            os.environ["LLM_BASE_URL"] = args.llm_base_url
            os.environ["LLM_MODEL"] = args.llm_model
            os.environ["LLM_API_KEY"] = args.llm_api_key
        except:
            pass

        cfg = get_config(os.path.join(folder, "../lctgen/lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml"))
        llm = OpenAIModel(cfg, base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY"), model=os.environ.get("LLM_MODEL"))
        llm_text = llm.forward(args.text)

        cfg = get_config(os.path.join(folder, "../configs/lctgen.yaml"))
        model = LCTGen.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
        model.eval()

        map_data_file = os.path.join(folder, "../lctgen/data/demo/waymo/demo_map_vec.npy")
        map_vecs, map_ids = load_all_map_vectors(map_data_file)

        scene = gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids, save_image=args.save_image)

    print(scene["traj"].swapaxes(0, 1).tolist())
