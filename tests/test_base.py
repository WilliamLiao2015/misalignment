import unittest

import numpy as np

class TestBase(unittest.TestCase):
    def set_trajectory(self, trajectory: np.ndarray):
        self.trajectory = trajectory
