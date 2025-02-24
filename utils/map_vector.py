import argparse
import sys
import os
import random

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../lctgen"))

from pathlib import Path

from trafficgen.utils.typedef import *
from trajdata.maps import MapAPI

def get_map_description(map_id):
    def id_to_connected_lane(id, connected_lanes):
        for lane in connected_lanes:
            if id in lane:
                return [lane]
        return [id]

    def get_neighbor_lanes(lane_id, connected_lanes, dir):
        same_lane_ids = id_to_connected_lane(lane_id, connected_lanes)
        neighbor_ids = []
        id_info = center_info[lane_id].__dict__
        for adj_lane_ids in id_info[f"adj_lanes_{dir}"]:
            if adj_lane_ids not in neighbor_ids and adj_lane_ids not in same_lane_ids:
                neighbor_ids.append(adj_lane_ids)

        neighbor_lanes = set()
        for id in neighbor_ids:
            neighbor_lanes.add(tuple(id_to_connected_lane(id, connected_lanes)))

        return {"lane": list(neighbor_lanes), "seg": neighbor_ids}

    def get_all_dir_lanes(lane_id, connected_lanes, dir):
        cnt = 0
        dir_lanes = []
        neighbor = get_neighbor_lanes(lane_id, connected_lanes, dir)

        while len(neighbor["seg"]) > 0 and cnt < 10:
            dir_lanes.append(neighbor)
            cnt += 1
            neighbor = get_neighbor_lanes(neighbor["seg"][0], connected_lanes, dir)

        return cnt, dir_lanes

    def get_opposite_neighbor(lane_id, unique_ids, all_same_dir_ids):
        yellow_types = [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW,
            RoadLineType.UNKNOWN
        ]

        id_info = center_info[lane_id].__dict__
        left_yellow_boundaries = [bound for bound in id_info["left_edge"] if bound["type"] in yellow_types]

        if len(left_yellow_boundaries) == 0:
            return []

        left_yellow_boundary_ids = [bound["id"] for bound in left_yellow_boundaries]
        left_opposite_ids = []

        for id in unique_ids:
            if id in all_same_dir_ids:
                continue
            id_info = center_info[id].__dict__
            left_boundary_ids = [bound["id"] for bound in id_info["left_edge"]]
            if len(set(left_boundary_ids) & set(left_yellow_boundary_ids)) > 0:
                left_opposite_ids.append(id)

        return left_opposite_ids

    map_api = MapAPI(Path("~/.unified_data_cache").expanduser())
    vector_map = map_api.get_map(map_id)
    center_info = {lane.id: lane for lane in vector_map.lanes}

    lane_id = "214_0" # random.choice(vector_map.lanes).id
    print(f"Map ID: {map_id}", f"Lane ID: {lane_id}")
    print(get_all_dir_lanes(lane_id, center_info[lane_id].reachable_lanes, "left"))
    print(get_all_dir_lanes(lane_id, center_info[lane_id].reachable_lanes, "right"))
    # print(get_opposite_neighbor(lane_id, vector_map.unique_ids, vector_map.all_same_dir_ids))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--map-id", type=str, default="waymo_val:waymo_val_0")
    args = argparser.parse_args()

    get_map_description(args.map_id)
