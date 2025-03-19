from trajdata.data_structures import AgentBatchElement, StateArray
from typing import Optional, Tuple

from utils.get_cross_product import get_cross_product
from utils.get_current_lanes import get_current_lanes
from utils.get_connected_paths import get_connected_paths

minimum_period = 10

cross_product_threshold = 1

def is_turning_left(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    current_lanes = get_current_lanes(element, state)
    connected_paths = get_connected_paths(current_lanes)

    for path in connected_paths:
        i = 0
        current_lane = path[i]

        for t, lanes in enumerate(current_lanes):
            while len(current_lane.next_lanes) < 2 and i < len(path) - 1:
                # Need at least two lanes to turn
                i += 1
                current_lane = path[i]

            for lane in lanes:
                if lane.id not in current_lane.next_lanes: continue
                cross_product = get_cross_product(current_lane.center.xy[0], current_lane.center.xy[-1], lane.center.xy[-1])
                if cross_product > cross_product_threshold:
                    if t < minimum_period: continue
                    return 0, t

    return None

def is_turning_right(element: AgentBatchElement, state: StateArray) -> Optional[Tuple[int, int]]:
    current_lanes = get_current_lanes(element, state)
    connected_paths = get_connected_paths(current_lanes)

    for path in connected_paths:
        i = 0
        current_lane = path[i]

        for t, lanes in enumerate(current_lanes):
            while len(current_lane.next_lanes) < 2 and i < len(path) - 1:
                # Need at least two lanes to turn
                i += 1
                current_lane = path[i]

            for lane in lanes:
                if lane.id not in current_lane.next_lanes: continue
                cross_product = get_cross_product(current_lane.center.xy[0], current_lane.center.xy[-1], lane.center.xy[-1])
                if cross_product < -cross_product_threshold:
                    if t < minimum_period: continue
                    return 0, t

    return None
