class Activity:
    def __init__(self, action, interval):
        self.action = action
        self.interval = interval

    def __str__(self):
        return f"{self.action} from {self.interval[0]} to {self.interval[1]}"

class Trajectory:
    def __init__(self, data):
        self.points = data["data"]
        self.activities = []

        for activity in data["activities"]:
            action = activity["action"]
            interval = activity["interval"]
            acceleration = activity["acceleration"]

            self.activities.append(Activity(action, interval))

            if acceleration > 0: action = "driving-forward:accelerating"
            elif acceleration < 0: action = "driving-forward:decelerating"
            else: action = "driving-forward:cruising"

            self.activities.append(Activity(action, interval))

    def __get_item__(self, index):
        return self.points[index]

    def __str__(self):
        activities_str = "\n\t".join([str(activity) for activity in self.activities])
        return f"Trajectory with {len(self.points)} points and {len(self.activities)} activities:\n\t{activities_str}"
