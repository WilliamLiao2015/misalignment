import numpy as np

from trajdata.data_structures import StateArray
from trajdata.maps.vec_map import VectorMap, RoadLane
from typing import List

from .flatten_map import flatten_map

def get_current_lanes(vector_map: VectorMap, state: StateArray) -> List[List[RoadLane]]:
    vector_map = flatten_map(vector_map)

    xy = state.as_format("x,y")
    diffs = xy[1:] - xy[:-1]
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.append(headings, headings[-1])
    xyzh = np.asarray(np.insert(np.insert(xy, 2, 1, axis=1), 3, headings, axis=1))
    current_lanes = [vector_map.get_current_lane(xyzh[t], 5) for t in range(xyzh.shape[0])]

    return current_lanes
