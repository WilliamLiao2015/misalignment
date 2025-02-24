from shapely.geometry import LineString, Point, Polygon

from .test_base import TestBase

class TestLateral(TestBase):
    def set_scenario(self, scenario):
        super().set_scenario(scenario)

        self.line_string = None
        previous_id = None

        for t in range(1, len(self.trajectory)):
            state = [*self.trajectory[t], 0]
            lane = self.map.get_closest_lane(state)

            if lane.id != previous_id:
                line_string = LineString(lane.center.xy)
                if self.line_string is None: self.line_string = line_string
                else: self.line_string = self.line_string.union(line_string)
                previous_id = lane.id

    def get_x(self, t: int) -> float:
        return self.line_string.project(Point(self.trajectory[t]), normalized=True)

    def get_delta_x(self, t: int) -> float:
        t = max(1, t)
        return self.get_x(t) - self.get_x(t - 1)

    def get_delta_v(self, t: int) -> float:
        t = max(1, t)
        print(self.get_delta_x(t), self.get_delta_x(t - 1))
        return self.get_delta_x(t) - self.get_delta_x(t - 1)

    def test_left_turn(self):
        lane = self.map.get_closest_lane(self.trajectory[0])
        polygon = Polygon(lane.center)
