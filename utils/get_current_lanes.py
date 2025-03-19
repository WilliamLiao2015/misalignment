import numpy as np

from trajdata.data_structures import AgentBatchElement, StateArray
from trajdata.maps.vec_map import RoadLane
from typing import List

from .flatten_map import flatten_map

def get_current_lanes(element: AgentBatchElement, state: StateArray) -> List[List[RoadLane]]:
    vector_map = flatten_map(element.vec_map)

    xyzh = np.asarray(np.insert(state.as_format("x,y,h"), 2, 1, axis=1))
    current_lanes = [vector_map.get_current_lane(xyzh[t], 5) for t in range(xyzh.shape[0])]

    return current_lanes
