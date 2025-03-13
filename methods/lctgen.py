import argparse
import os
import re
import sys

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../lctgen"))

import openai
import torch
from PIL import Image
from pydantic import BaseModel
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

    with HiddenPrints():
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

if __name__ == "__main__":
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
