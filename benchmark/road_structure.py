import numpy as np

from trajdata.data_structures import StateArray
from trajdata.maps.vec_map import VectorMap

def get_offroad_ratios(vector_map: VectorMap, state: StateArray) -> float:
    offroad_counts = 0
    for t in range(len(state)):
        lanes = vector_map.get_lanes_within(np.asarray([*state[t, :2], 1]), 5)
        if len(lanes) == 0: offroad_counts += 1
    return offroad_counts / len(state)

def has_continuous_path(vector_map: VectorMap, state: StateArray) -> bool:
    for t in range(len(state) - 1):
        lanes = vector_map.get_lanes_within(np.asarray([*state[t, :2], 1]), 5)
        next_lanes = vector_map.get_lanes_within(np.asarray([*state[t + 1, :2], 1]), 5)
        if len(lanes) == 0 or len(next_lanes) == 0: return False
        if len(set([lane.id for lane in lanes]).intersection([lane.id for lane in next_lanes])) == 0: return False
    return True
