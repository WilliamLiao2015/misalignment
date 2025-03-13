import numpy as np

from .test_base import TestBase

epsilon_straight = np.pi / 12
epsilon_turning = np.pi / 6

class TestLateral(TestBase):
    def test_going_straight(self):
        angles = np.asarray([np.arctan2(y - self.trajectory[0, 0, 1], x - self.trajectory[0, 0, 0]) for x, y in self.trajectory[0, 1:, :]])
        diffs = (angles[1:] - angles[0]) % np.pi
        self.assertTrue(np.all(np.abs(diffs) < epsilon_straight), f"Invalid differences: {diffs[np.abs(diffs) >= epsilon_straight]}")

    def test_turning_right(self):
        angles = np.asarray([np.arctan2(y - self.trajectory[0, 0, 1], x - self.trajectory[0, 0, 0]) for x, y in self.trajectory[0, 1:, :]])
        diffs = (angles[1:] - angles[0]) % np.pi
        self.assertTrue(np.all(diffs < epsilon_straight) and diffs[-1] < -epsilon_turning, f"Invalid differences: {diffs[(diffs >= epsilon_straight)]}, {diffs[-1]}")

    def test_turning_left(self):
        angles = np.asarray([np.arctan2(y - self.trajectory[0, 0, 1], x - self.trajectory[0, 0, 0]) for x, y in self.trajectory[0, 1:, :]])
        diffs = (angles[1:] - angles[0]) % np.pi
        self.assertTrue(np.all(diffs > -epsilon_straight) and diffs[-1] > epsilon_turning, f"Invalid differences: {diffs[(diffs <= -epsilon_straight)]}, {diffs[-1]}")
