import numpy as np

from .test_base import TestBase

epsilon_speed = 0.1
epsilon_acceleration = 0.1

class TestLongitudinal(TestBase):
    def test_decelerating(self):
        deltas = np.linalg.norm(np.diff(self.trajectory[0], axis=0), axis=1)
        diffs = deltas[1:] - deltas[:-1]
        self.assertTrue(np.all(diffs < epsilon_acceleration), f"Invalid differences: {diffs[diffs >= epsilon_acceleration]}")

    def test_cruising(self):
        deltas = np.linalg.norm(np.diff(self.trajectory[0], axis=0), axis=1)
        diffs = deltas[1:] - deltas[:-1]
        self.assertTrue(np.all(np.abs(diffs) < epsilon_acceleration), f"Invalid differences: {diffs[np.abs(diffs) >= epsilon_acceleration]}")

    def test_accelerating(self):
        deltas = np.linalg.norm(np.diff(self.trajectory[0], axis=0), axis=1)
        diffs = deltas[1:] - deltas[:-1]
        self.assertTrue(np.all(diffs > -epsilon_acceleration), f"Invalid differences: {diffs[diffs <= -epsilon_acceleration]}")

    def test_standing_still(self):
        deltas = np.linalg.norm(np.diff(self.trajectory[0], axis=0), axis=1)
        self.assertTrue(np.all(deltas < epsilon_speed), f"Invalid deltas: {deltas[deltas >= epsilon_speed]}")
