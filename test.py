import argparse
import unittest
import json

from tests.test_longitudinal import TestLongitudinal

def add_tests(suite, scenario):
    for i, activities in enumerate(scenario["activities"]):
        for activity in activities:
            action = activity["action"]
            start, end = activity["interval"]
            test = None

            test_scenario = {
                "map_id": scenario["mapId"],
                "trajectory": scenario["trajectories"][i][start:end - 1],
                "activity": activity
            }

            if action == "driving-forward:cruising":
                test = TestLongitudinal("test_cruising")
            elif action == "driving-forward:accelerating":
                test = TestLongitudinal("test_accelerating")
            elif action == "driving-forward:decelerating":
                test = TestLongitudinal("test_decelerating")

            if test:
                test.set_scenario(test_scenario)
                suite.addTest(test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests.")
    parser.add_argument("--scenario-path", type=str, help="Scenario to test.")
    args = parser.parse_args()

    print(f"Testing scenario: {args.scenario_path}")

    with open(args.scenario_path, "r") as fp:
        scenario = json.load(fp)

    suite = unittest.TestSuite()
    add_tests(suite, scenario)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
