import argparse
import unittest

import numpy as np

from benchmark.main import evaluate_method, get_config

from tests.test_longitudinal import TestLongitudinal
from tests.test_lateral import TestLateral

test_generator_map = {
    "longitudinal:driving-forward:accelerating": lambda: TestLongitudinal("test_accelerating"),
    "longitudinal:driving-forward:cruising": lambda: TestLongitudinal("test_cruising"),
    "longitudinal:driving-forward:decelerating": lambda: TestLongitudinal("test_decelerating"),
    "longitudinal:standing-still": lambda: TestLongitudinal("test_standing_still"),
    "lateral:going-straight": lambda: TestLateral("test_going_straight"),
    "lateral:turning:right": lambda: TestLateral("test_turning_right"),
    "lateral:turning:left": lambda: TestLateral("test_turning_left")
}

def add_tests(suite, config: dict, trajectories: np.ndarray):
    for activity in config["activities"]:
        indices = [int(participant[1:]) - 1 for participant in activity["participants"]] # remove the "V" prefix and subtract 1

        action = activity["type"]
        start, end = activity["interval"] if "interval" in activity else (0, len(trajectories[0]))
        test = None

        if action not in test_generator_map: continue

        test = test_generator_map[action]()
        test.set_trajectory(trajectories[indices][start:end])
        suite.addTest(test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tests for the generated scenarios.")
    parser.add_argument("--config", type=str, help="The path to the configuration file.", default="benchmark/configs/turning-right.yaml")
    parser.add_argument("--method", type=str, help="The method to evaluate.", default="lctgen")
    args = parser.parse_args()

    config = get_config(args.config)

    try:
        returncode, stdout, stderr = evaluate_method(args.method, config)
        trajectories = np.asarray(eval(stdout.decode("utf-8")))

        suite = unittest.TestSuite()
        add_tests(suite, config, trajectories)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    except Exception as e:
        raise e
