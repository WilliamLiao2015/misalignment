import numpy as np

from .test_base import TestBase

epsilon_straight = np.pi / 8
epsilon_turning = np.pi / 4

class TestLateral(TestBase):
    def test_going_straight(self):
        angles = np.asarray([np.arctan2(y - self.trajectory[0, 0, 1], x - self.trajectory[0, 0, 0]) for x, y in self.trajectory[0, 1:, :]])
        self.assertTrue(np.all(np.abs(angles[1:] - angles[0]) < epsilon_straight))

    def test_turning_right(self):
        angles = np.asarray([np.arctan2(y - self.trajectory[0, 0, 1], x - self.trajectory[0, 0, 0]) for x, y in self.trajectory[0, 1:, :]])
        self.assertTrue(np.all(angles[1:] - angles[0] < epsilon_straight) and angles[-1] - angles[0] < -epsilon_turning)

    def test_turning_left(self):
        angles = np.asarray([np.arctan2(y - self.trajectory[0, 0, 1], x - self.trajectory[0, 0, 0]) for x, y in self.trajectory[0, 1:, :]])
        self.assertTrue(np.all(angles[1:] - angles[0] > -epsilon_straight) and angles[-1] - angles[0] > epsilon_turning)
