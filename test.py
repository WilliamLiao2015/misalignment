import unittest

import numpy as np

from benchmark.generate import generate_configs
from benchmark.main import describe_for_lctgen

from methods.lctgen import generate_scene

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
        participants_name_map = {participant: f"V{i + 1}" for i, participant in enumerate(set([participant for activity in config["activities"] for participant in activity["participants"]]))}

        indices = [int(participants_name_map[participant][1:]) - 1 for participant in activity["participants"]] # remove the "V" prefix and subtract 1

        action = activity["type"]
        start, end = activity["interval"] if "interval" in activity else (0, len(trajectories[0]))
        test = None

        if action not in test_generator_map: continue

        try:
            test = test_generator_map[action]()
            test.set_trajectory(trajectories[indices][start:end])
            suite.addTest(test)
        except: pass

if __name__ == "__main__":
    batch, configs = generate_configs(standardize_data=True)
    config = configs[0] if len(configs) > 0 else None

    try:
        if config is None: raise Exception("No valid configuration generated")
        description = describe_for_lctgen(config)
        print(f"Running scenario generation based on text: \"{description}\"")
        scene = generate_scene(config, description)
        trajectories = scene["traj"].swapaxes(0, 1)

        suite = unittest.TestSuite()
        add_tests(suite, config, trajectories)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    except Exception as e:
        raise e
