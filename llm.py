import os
import re
import sys

from typing import List, Tuple

import openai

from pydantic import BaseModel

from lctgen.core.basic import BasicLLM

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

class LCTGenBaseLLM(BasicLLM):
    def __init__(self, config, base_url=None, api_key=None, model="gpt-4o-mini"):
        super().__init__(config)

        self.base_url = base_url
        self.model = model

        self.client = openai.Client(base_url=base_url, api_key=api_key)

        self.codex_cfg = config.LLM.CODEX
        folder = os.path.join(os.path.dirname(__file__), "lctgen/lctgen/gpt")
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

    def llm_query(self, extended_prompt, use_structured_output=False):
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
                response_format=LCTGenStructuredRepresentation if use_structured_output else None
            )
            representation = responses.choices[0].message.parsed
            agents = [[agent.position_negative_for_ego, agent.distance_from_ego, agent.direction_wrt_ego, agent.speed_interval, *agent.action_in_next_4s] for agent in representation.agents]
            map = [*representation.map.parallel_lane_counts, *representation.map.perpendicular_lane_counts, representation.map.distance_to_intersection, representation.map.lane_id]
            resp = (agents, map)

        return resp

    def post_process(self, response):
        return response
