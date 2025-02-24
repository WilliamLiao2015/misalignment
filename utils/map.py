from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import Polyline

def flatten_map(map_data: VectorMap):
    map_data.extent[2] = 0
    map_data.extent[5] = 0

    for element in map_data.iter_elems():
        for attribute in element.__dict__:
            if type(element.__dict__[attribute]) == Polyline:
                element.__dict__[attribute].points[:, 2] = 0

    map_data.compute_search_indices()

    return map_data
