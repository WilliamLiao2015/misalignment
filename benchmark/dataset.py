import numpy as np

from trajdata.dataset import UnifiedDataset
from trajdata.data_structures import AgentBatchElement
from typing import List, Tuple

from . import tests

valid_types = [1] # Vehicle

def create_dataset(standardize_data: bool = False) -> UnifiedDataset:
    return UnifiedDataset(["waymo_val"], data_dirs={
        "waymo_val": "../../data/waymo"
    }, incl_vector_map=True, vector_map_params={
        "incl_road_lanes": True,
        "incl_road_areas": False,
        "incl_ped_crosswalks": False,
        "incl_ped_walkways": False,
        "collate": True,
        "keep_in_memory": False
    }, incl_raster_map=True, raster_map_params={
        "map_size_px": 512,
        "px_per_m": 4
    }, standardize_data=standardize_data)

def describe(batch: List[AgentBatchElement], log: bool = False) -> List[dict]:
    configs = []

    for element in batch:
        history = [element.agent_history_np]
        future = [element.agent_future_np]
        for i, neighbor in enumerate(element.neighbor_histories):
            if element.neighbor_types_np[i] in valid_types: history.append(neighbor)
        for i, neighbor in enumerate(element.neighbor_futures):
            if element.neighbor_types_np[i] in valid_types: future.append(neighbor)

        config = {
            "history": history,
            "future": future,
            "vec_map": element.vec_map,
            "tests": {}
        }

        for test_name, test_func in tests.items():
            if log: print(f"Running test {test_name}...")
            config["tests"][test_name] = [test_func(element, agent) for agent in config["future"]]

        configs.append(config)

    return configs

def collate_fn(batch: List[AgentBatchElement]) -> Tuple[List[AgentBatchElement], List[dict]]:
    configs = describe(batch)

    return batch, configs
