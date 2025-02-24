import unittest

import numpy as np

from pathlib import Path

from shapely.geometry import LineString
from trajdata.maps import MapAPI

from utils.map import flatten_map

map_api = MapAPI(Path("~/.unified_data_cache").expanduser())

class TestBase(unittest.TestCase):
    def set_scenario(self, scenario):
        self.map = flatten_map(map_api.get_map(scenario["map_id"]))
        self.trajectory = np.asarray(scenario["trajectory"])
        self.activity = scenario["activity"]
