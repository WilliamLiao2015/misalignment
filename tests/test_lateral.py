import numpy as np

from .test_base import TestBase

class TestLateral(TestBase):
    def test_going_straight(self):
        start = self.trajectory[0]
        end = self.trajectory[-1]
        self.assertTrue(np.allclose(np.arctan2(end[:, 1]-start[:, 1], end[:, 0]-start[:, 0]), 0))

    def test_turning_right(self):
        start = self.trajectory[0]
        end = self.trajectory[-1]
        self.assertTrue(np.all(np.arctan2(end[:, 1]-start[:, 1], end[:, 0]-start[:, 0]) > 0))

    def test_turning_left(self):
        start = self.trajectory[0]
        end = self.trajectory[-1]
        self.assertTrue(np.all(np.arctan2(end[:, 1]-start[:, 1], end[:, 0]-start[:, 0]) < 0))
