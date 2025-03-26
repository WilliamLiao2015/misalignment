import numpy as np

from typing import List

from trajdata.maps.vec_map import VectorMap
from trajdata.data_structures import StateArray

from .get_cross_product import get_cross_product

cross_product_threshold = 0.3

def get_possible_activities(vector_map: VectorMap, agents: List[StateArray]) -> List[str]:
    activities = []

    for i, agent in enumerate(agents):
        xy = agent.as_format("x,y")
        heading = np.arctan2(xy[1, 1] - xy[0, 0], xy[1, 1] - xy[0, 0])
        xyzh = np.asarray([xy[0, 0], xy[0, 1], 1, heading])
        lanes = vector_map.get_current_lane(xyzh, 5)
        # activities.append({"type": "longitudinal:standing-still", "participants": [f"V{i + 1}"]})
        if len(lanes) == 0: continue
        activities.append({"type": "longitudinal:driving-forward:accelerating", "participants": [f"V{i + 1}"]})
        activities.append({"type": "longitudinal:driving-forward:cruising", "participants": [f"V{i + 1}"]})
        activities.append({"type": "longitudinal:driving-forward:decelerating", "participants": [f"V{i + 1}"]})
        if len(lanes) == 1:
            # activities.append({"type": "lateral:going-straight", "participants": [f"V{i + 1}"]})
            pass
        else:
            right = False
            left = False
            for lane in lanes:
                if right and left: break
                try:
                    for next_lane in [vector_map.get_road_lane(lane_id) for lane_id in lane.next_lanes]:
                        if not right and get_cross_product(lane.center.xy[0], lane.center.xy[-1], next_lane.center.xy[-1]) > cross_product_threshold:
                            activities.append({"type": "lateral:turning:right", "participants": [f"V{i + 1}"]})
                            right = True
                        if not left and get_cross_product(lane.center.xy[0], lane.center.xy[-1], next_lane.center.xy[-1]) < -cross_product_threshold:
                            activities.append({"type": "lateral:turning:left", "participants": [f"V{i + 1}"]})
                            left = True
                except: pass

    return activities
