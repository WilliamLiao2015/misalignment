import numpy as np

from shapely.geometry import LineString, Point
from trajdata.data_structures import StateArray
from trajdata.maps.vec_map import VectorMap
from typing import List, Optional, Tuple

from utils.get_current_lanes import get_current_lanes
from utils.get_connected_paths import get_connected_paths

minimum_period = 7

epsilon_progress = 0
epsilon_velocity = 1e-2
epsilon_acceleration = 1e-2

def get_progress_infos(vector_map: VectorMap, state: StateArray) -> List[np.ndarray]:
    current_lanes = get_current_lanes(vector_map, state)
    connected_paths = get_connected_paths(current_lanes)

    state = state.as_format("x,y")

    progress_infos = []
    for path in connected_paths:
        i = 0
        current_lane = path[i]
        progress = []

        for t, lanes in enumerate(current_lanes):
            if current_lane.id in [lane.id for lane in lanes]:
                line = LineString(current_lane.center.xy)
                p = line.project(Point(state[t]), normalized=True)
                progress.append(p)
            else:
                if i == len(path) - 1: break
                i += 1
                current_lane = path[i]
                progress = [p - 1 for p in progress]

        progress_infos.append(progress)

    return progress_infos

def is_accelerating(vector_map: VectorMap, state: StateArray) -> Optional[Tuple[int, int]]:
    for t in range(len(state) - minimum_period):
        deltas = np.linalg.norm(np.diff(state[t:t + minimum_period], axis=0), axis=1)
        diffs = deltas[1:] - deltas[:-1]
        if not np.all(diffs > -epsilon_acceleration): continue
        return t, t + minimum_period

        # starts = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t, :2], 1]), 5)])
        # ends = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t + minimum_period, :2], 1]), 5)]).difference(starts)
        # currents = starts.copy()
        # for _ in range(5):
        #     new_currents = currents.copy()
        #     for lane_id in currents:
        #         try: new_currents.update(vector_map.get_road_lane(lane_id).reachable_lanes)
        #         except: pass
        #         if ends.intersection(new_currents):
        #             return t, t + minimum_period
        #     currents = new_currents
    return None

def is_cruising(vector_map: VectorMap, state: StateArray) -> Optional[Tuple[int, int]]:
    for t in range(len(state) - minimum_period):
        deltas = np.linalg.norm(np.diff(state[t:t + minimum_period], axis=0), axis=1)
        diffs = deltas[1:] - deltas[:-1]
        if not np.all(np.abs(diffs) < epsilon_acceleration): continue
        return t, t + minimum_period

        # starts = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t, :2], 1]), 5)])
        # ends = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t + minimum_period, :2], 1]), 5)]).difference(starts)
        # currents = starts.copy()
        # for _ in range(5):
        #     new_currents = currents.copy()
        #     for lane_id in currents:
        #         try: new_currents.update(vector_map.get_road_lane(lane_id).reachable_lanes)
        #         except: pass
        #         if ends.intersection(new_currents):
        #             return t, t + minimum_period
        #     currents = new_currents
    return None

def is_decelerating(vector_map: VectorMap, state: StateArray) -> Optional[Tuple[int, int]]:
    for t in range(len(state) - minimum_period):
        deltas = np.linalg.norm(np.diff(state[t:t + minimum_period], axis=0), axis=1)
        diffs = deltas[1:] - deltas[:-1]
        if not np.all(diffs < epsilon_acceleration): continue
        return t, t + minimum_period

        # starts = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t, :2], 1]), 5)])
        # ends = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t + minimum_period, :2], 1]), 5)]).difference(starts)
        # currents = starts.copy()
        # for _ in range(5):
        #     new_currents = currents.copy()
        #     for lane_id in currents:
        #         try: new_currents.update(vector_map.get_road_lane(lane_id).reachable_lanes)
        #         except: pass
        #         if ends.intersection(new_currents):
        #             return t, t + minimum_period
        #     currents = new_currents
    return None

def is_standing_still(vector_map: VectorMap, state: StateArray) -> Optional[Tuple[int, int]]:
    for t in range(len(state) - minimum_period):
        deltas = np.linalg.norm(np.diff(state[t:t + minimum_period], axis=0), axis=1)
        if not np.all(np.abs(deltas) < epsilon_velocity): continue
        return t, t + minimum_period

        # starts = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t, :2], 1]), 5)])
        # ends = set([lane.id for lane in vector_map.get_lanes_within(np.asarray([*state[t + minimum_period, :2], 1]), 5)]).difference(starts)
        # currents = starts.copy()
        # for _ in range(5):
        #     new_currents = currents.copy()
        #     for lane_id in currents:
        #         try: new_currents.update(vector_map.get_road_lane(lane_id).reachable_lanes)
        #         except: pass
        #         if ends.intersection(new_currents):
        #             return t, t + minimum_period
        #     currents = new_currents
    return None
