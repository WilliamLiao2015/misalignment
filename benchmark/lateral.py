import numpy as np

from trajdata.data_structures import StateArray
from trajdata.maps.vec_map import VectorMap
from typing import Optional, Tuple

# from utils.get_cross_product import get_cross_product
# from utils.get_current_lanes import get_current_lanes
# from utils.get_connected_paths import get_connected_paths

minimum_period = 10

# cross_product_threshold = 0.3

epsilon_straight = np.pi / 12
epsilon_turning = np.pi / 6

def is_turning_left(vector_map: VectorMap, state: StateArray) -> Optional[Tuple[int, int]]:
    initial_angle = np.arctan2(state[1, 1] - state[0, 1], state[1, 0] - state[0, 0])
    for t in range(len(state) - minimum_period):
        angles = np.asarray([np.arctan2(y2 - y1, x2 - x1) for (x1, y1), (x2, y2) in zip(state[t:t + minimum_period], state[t + 1:t + minimum_period + 1])])
        diffs = (angles[1:] - initial_angle + np.pi) % (2 * np.pi) - np.pi
        if np.all(diffs > -epsilon_straight) and diffs[-1] > epsilon_turning: return t, t + minimum_period

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

    # if np.allclose(state[-1] - state[0], 0): return None

    # current_lanes = get_current_lanes(vector_map, state)
    # connected_paths = get_connected_paths(current_lanes)

    # for path in connected_paths:
    #     i = 0
    #     current_lane = path[i]

    #     for t, lanes in enumerate(current_lanes):
    #         while len(current_lane.next_lanes) < 2 and i < len(path) - 1:
    #             # Need at least two lanes to turn
    #             i += 1
    #             current_lane = path[i]

    #         for lane in lanes:
    #             if lane.id not in current_lane.next_lanes: continue
    #             cross_product = get_cross_product(current_lane.center.xy[0], current_lane.center.xy[-1], lane.center.xy[-1])
    #             cross_product /= np.linalg.norm(current_lane.center.xy[-1] - current_lane.center.xy[0]) * np.linalg.norm(lane.center.xy[-1] - current_lane.center.xy[0])
    #             if cross_product > cross_product_threshold:
    #                 print(cross_product)
    #                 if t < minimum_period: continue
    #                 return 0, t

    return None

def is_turning_right(vector_map: VectorMap, state: StateArray) -> Optional[Tuple[int, int]]:
    initial_angle = np.arctan2(state[1, 1] - state[0, 1], state[1, 0] - state[0, 0])
    for t in range(len(state) - minimum_period):
        angles = np.asarray([np.arctan2(y2 - y1, x2 - x1) for (x1, y1), (x2, y2) in zip(state[t:t + minimum_period], state[t + 1:t + minimum_period + 1])])
        diffs = (angles[1:] - initial_angle + np.pi) % (2 * np.pi) - np.pi
        if np.all(diffs < epsilon_straight) and diffs[-1] < -epsilon_turning: return t, t + minimum_period

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

    # if np.allclose(state[-1] - state[0], 0): return None

    # current_lanes = get_current_lanes(vector_map, state)
    # connected_paths = get_connected_paths(current_lanes)

    # for path in connected_paths:
    #     i = 0
    #     current_lane = path[i]

    #     for t, lanes in enumerate(current_lanes):
    #         while len(current_lane.next_lanes) < 2 and i < len(path) - 1:
    #             # Need at least two lanes to turn
    #             i += 1
    #             current_lane = path[i]

    #         for lane in lanes:
    #             if lane.id not in current_lane.next_lanes: continue
    #             cross_product = get_cross_product(current_lane.center.xy[0], current_lane.center.xy[-1], lane.center.xy[-1])
    #             cross_product /= np.linalg.norm(current_lane.center.xy[-1] - current_lane.center.xy[0]) * np.linalg.norm(lane.center.xy[-1] - current_lane.center.xy[0])
    #             if cross_product < -cross_product_threshold:
    #                 print(cross_product)
    #                 if t < minimum_period: continue
    #                 return 0, t

    return None
