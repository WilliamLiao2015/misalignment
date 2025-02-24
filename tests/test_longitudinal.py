import numpy as np

from .constants import epsilon_speed, epsilon_acceleration
from .test_base import TestBase

class TestLongitudinal(TestBase):
    def get_speed(self, t: int) -> float:
        t = max(1, t)
        return np.linalg.norm(self.trajectory[t] - self.trajectory[t - 1])

    def get_acceleration(self, t: int) -> float:
        t = max(1, t)
        return self.get_speed(t) - self.get_speed(t - 1)

    def test_driving_forward(self):
        # if self.get_x(len(self.trajectory) - 1) <= self.get_x(1):
        #     self.fail(f"Trajectory does not move forward.")

        for t in range(1, len(self.trajectory)):
            delta_x = self.get_speed(t)
            if delta_x <= -epsilon_speed:
                self.fail(f"Trajectory moves backward of {delta_x} at {t}.")

    def test_decelerating(self):
        self.test_driving_forward()

        if self.get_speed(len(self.trajectory) - 1) >= self.get_speed(1):
            self.fail(f"Trajectory does not decelerate.")

        for t in range(1, len(self.trajectory)):
            delta_v = self.get_acceleration(t)
            if delta_v > epsilon_acceleration:
                self.fail(f"Speed increases of {delta_v} at {t}.")

    def test_cruising(self):
        self.test_driving_forward()

        # print(self.get_speed(len(self.trajectory) - 1), self.get_speed(1))
        if abs(self.get_speed(len(self.trajectory) - 1) - self.get_speed(1)) > epsilon_speed:
            self.fail(f"Trajectory does not cruise.")

        for t in range(1, len(self.trajectory)):
            delta_v = self.get_acceleration(t)
            if abs(delta_v) > epsilon_acceleration:
                self.fail(f"Speed changes of {delta_v} at {t}.")

    def test_accelerating(self):
        self.test_driving_forward()

        if self.get_speed(len(self.trajectory) - 1) <= self.get_speed(1):
            self.fail(f"Trajectory does not accelerate.")

        for t in range(1, len(self.trajectory)):
            delta_v = self.get_acceleration(t)
            if delta_v < -epsilon_acceleration:
                self.fail(f"Speed decreases of {delta_v} at {t}.")
