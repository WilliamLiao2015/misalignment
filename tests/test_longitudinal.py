import numpy as np

from .constants import epsilon_speed, epsilon_acceleration
from .test_base import TestBase

class TestLongitudinal(TestBase):
    def test_cruising(self):
        self.assertTrue(np.all(np.abs(np.diff(self.trajectory[:, 1])) < epsilon_speed))

    def test_accelerating(self):
        self.assertTrue(np.all(np.diff(self.trajectory[:, 1]) > epsilon_acceleration))

    def test_decelerating(self):
        self.assertTrue(np.all(np.diff(self.trajectory[:, 1]) < -epsilon_acceleration))

    def test_standing_still(self):
        self.assertTrue(np.all(np.abs(self.trajectory[:, 1]) < epsilon_speed))

    def test_reversing(self):
        self.assertTrue(np.all(self.trajectory[:, 1] < 0))
