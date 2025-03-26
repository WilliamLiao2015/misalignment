from trajdata.maps.vec_map import RoadLane
from typing import List

def get_connected_paths(lanes: List[List[RoadLane]]) -> List[List[RoadLane]]:
    if len(lanes) == 0: return []

    candidate_paths = [[lane] for lane in lanes[-1]]

    for previous_lanes in reversed(lanes[:-1]):
        new_candidate_paths = []
        for previous_lane in previous_lanes:
            existing_paths = set()
            for candidate_path in candidate_paths:
                if previous_lane.id in candidate_path[-1].prev_lanes or candidate_path[-1].id in previous_lane.next_lanes:
                    new_candidate_paths.append(candidate_path + [previous_lane])
                elif previous_lane.id == candidate_path[-1].id:
                    path = tuple([lane.id for lane in candidate_path])
                    if path in existing_paths: continue
                    new_candidate_paths.append(candidate_path)
                    existing_paths.add(path)
        candidate_paths = new_candidate_paths

    candidate_paths = [candidate_path for candidate_path in candidate_paths if candidate_path[0].id in [lane.id for lane in lanes[0]] and candidate_path[-1].id in [lane.id for lane in lanes[-1]]]

    return candidate_paths
