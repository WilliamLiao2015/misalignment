import numpy as np

from typing import List

from trajdata.maps.vec_map import VectorMap
from trajdata.data_structures import StateArray

from benchmark.lateral import epsilon_straight, epsilon_turning

def get_possible_activities(vector_map: VectorMap, agents: List[StateArray]) -> List[str]:
    activities = []

    for i, agent in enumerate(agents):
        xy = agent.as_format("x,y")
        heading = np.arctan2(xy[1, 1] - xy[0, 0], xy[1, 1] - xy[0, 0])
        xyzh = np.asarray([xy[0, 0], xy[0, 1], 1, heading])
        lanes = vector_map.get_current_lane(xyzh, 5)
        activities.append({"type": "longitudinal:standing-still", "participants": [f"V{i + 1}"]})

        if len(lanes) == 0: continue

        activities.append({"type": "longitudinal:driving-forward:accelerating", "participants": [f"V{i + 1}"]})
        activities.append({"type": "longitudinal:driving-forward:cruising", "participants": [f"V{i + 1}"]})
        activities.append({"type": "longitudinal:driving-forward:decelerating", "participants": [f"V{i + 1}"]})

        paths = [[lane] for lane in lanes]
        right = False
        left = False

        for _ in range(3):
            new_paths = []
            for path in paths:
                for lane in path[-1].next_lanes:
                    try: new_paths.append(path + [vector_map.get_road_lane(lane)])
                    except: new_paths.append(path)
            paths = new_paths

        for path in paths:
            initial_angle = np.arctan2(path[0].center.xy[1][1] - path[0].center.xy[0][1], path[0].center.xy[1][0] - path[0].center.xy[0][0])
            points = np.asarray([point for lane in path for point in lane.center.xy])
            angles = np.asarray([np.arctan2(y2 - y1, x2 - x1) for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])])
            diffs = (angles[1:] - initial_angle + np.pi) % (2 * np.pi) - np.pi
            if len(diffs) < 2: continue
            if np.all(diffs > -epsilon_straight) and diffs[-1] > epsilon_turning: left = True
            if np.all(diffs < epsilon_straight) and diffs[-1] < -epsilon_turning: right = True

        if right: activities.append({"type": "lateral:turning:right", "participants": [f"V{i + 1}"]})
        if left: activities.append({"type": "lateral:turning:left", "participants": [f"V{i + 1}"]})

    return activities
