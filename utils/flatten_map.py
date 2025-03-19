from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import Polyline

def flatten_map(vector_map: VectorMap):
    vector_map.extent[2] = 0
    vector_map.extent[5] = 0

    for element in vector_map.iter_elems():
        for attribute in element.__dict__:
            if type(element.__dict__[attribute]) == Polyline:
                element.__dict__[attribute].points[:, 2] = 0

    vector_map.compute_search_indices()

    return vector_map
