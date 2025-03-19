import numpy as np

from shapely.geometry import LineString, Point
from trajdata.data_structures import AgentBatchElement, StateArray
from typing import List, Optional, Tuple

from utils.get_current_lanes import get_current_lanes
from utils.get_connected_paths import get_connected_paths

minimum_period = 10

epsilon_progress = 0
epsilon_velocity = 0
epsilon_acceleration = 1e-5

def get_progress_infos(element: AgentBatchElement, state: StateArray) -> List[np.ndarray]:
    current_lanes = get_current_lanes(element, state)
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

def is_driving_forward(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    for progress in get_progress_infos(element, state):
        if len(progress) < minimum_period: continue
        for t in range(len(progress)):
            if t + minimum_period > len(progress): break
            if np.all(np.diff(progress[t:t + minimum_period]) > epsilon_progress): return t, t + minimum_period
    return None

def is_accelerating(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    if not is_driving_forward(element, state): return None
    for progress in get_progress_infos(element, state):
        if len(progress) < minimum_period: continue
        for t in range(len(progress)):
            if t + minimum_period > len(progress): break
            if np.all(np.diff(np.diff(progress[t:t + minimum_period])) > epsilon_acceleration): return t, t + minimum_period
    return None

def is_cruising(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    if not is_driving_forward(element, state): return None
    for progress in get_progress_infos(element, state):
        if len(progress) < minimum_period: continue
        for t in range(len(progress)):
            if t + minimum_period > len(progress): break
            longitudinal_v = np.diff(progress[t:t + minimum_period])
            if np.all(np.abs(np.diff(longitudinal_v)) <= epsilon_acceleration):
                # Check if the agent is actually standing still or just projected onto the wrong lane
                if np.allclose(longitudinal_v, 0): continue
                return t, t + minimum_period
    return None

def is_decelerating(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    if not is_driving_forward(element, state): return None
    for progress in get_progress_infos(element, state):
        if len(progress) < minimum_period: continue
        for t in range(len(progress)):
            if t + minimum_period > len(progress): break
            if np.all(np.diff(np.diff(progress[t:t + minimum_period])) < -epsilon_acceleration): return t, t + minimum_period
    return None

def is_standing_still(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    for progress in get_progress_infos(element, state):
        if len(progress) < minimum_period: continue
        for t in range(len(progress)):
            if t + minimum_period > len(progress): break
            if np.all(np.abs(np.diff(progress[t:t + minimum_period])) <= epsilon_progress):
                # Check if the agent is actually standing still or just projected onto the wrong lane
                if np.allclose(progress[t:t + minimum_period], 0) or np.allclose(progress[t:t + minimum_period], 1): continue
                return t, t + minimum_period
    return None

def is_reversing(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    for progress in get_progress_infos(element, state):
        if len(progress) < minimum_period: continue
        for t in range(len(progress)):
            if t + minimum_period > len(progress): break
            if np.all(np.diff(progress[t:t + minimum_period]) < -epsilon_progress): return t, t + minimum_period
    return None
