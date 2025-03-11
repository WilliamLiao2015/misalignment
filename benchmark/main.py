import os
import subprocess
import yaml

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

type_map = {
    "longitudinal:driving-forward:accelerating": "Accelerating",
    "longitudinal:driving-forward:cruising": "Cruising",
    "longitudinal:driving-forward:decelerating": "Decelerating",
    "longitudinal:driving-forward:standing-still": "Standing Still",
    "longitudinal:driving-forward:reverse": "Reversing",
    "lateral:going-straight": "Going Straight",
    "lateral:turning:right": "Turning Right",
    "lateral:turning:left": "Turning Left"
}

def get_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

def describe_config(config: dict, combine=True) -> str:
    description = ""
    if not combine:
        for activity in config["activities"]:
            description += f"{' and '.join(activity['participants'])} {'is' if len(activity['participants']) == 1 else 'are'} {type_map[activity['type']].lower()}, "
        description = description[:-2]
    else:
        single_participant_activities = sorted([activity for activity in config["activities"] if len(activity["participants"]) == 1], key=lambda x: x["participants"][0])
        multi_participant_activities = [activity for activity in config["activities"] if len(activity["participants"]) > 1]

        current_participant = None
        for activity in single_participant_activities:
            if current_participant != activity["participants"][0]:
                if current_participant is not None: description += ", "
                description += f"{activity['participants'][0]} is {type_map[activity['type']].lower()}"
            else: description += f" and {type_map[activity['type']].lower()}"
            current_participant = activity["participants"][0]
        for activity in multi_participant_activities:
            description += f"{' and '.join(activity['participants'])} are {type_map[activity['type']].lower()}, "
        if len(multi_participant_activities): description = description[:-2]
    return description + "."

def evaluate_method(method: str, config: dict) -> float:
    if method == "lctgen":
        conda_env = "lctgen"
        script_path = "-m methods.lctgen"
    else:
        raise ValueError(f"Method {method} not recognized.")

    print(f"Running scenario generation based on text: \"{describe_config(config)}\"")

    command = f"conda run -n {conda_env} python {script_path} --text '{describe_config(config)}' --save_image --llm_base_url {os.environ.get('LLM_BASE_URL')} --llm_model {os.environ.get('LLM_MODEL')}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    return process.returncode, stdout, stderr

if __name__ == "__main__":
    config = get_config("benchmark/configs/turning-right.yaml")

    try:
        returncode, stdout, stderr = evaluate_method("lctgen", config)
        print(stderr.decode("utf-8"))
        trajectories = np.asarray(eval(stdout.decode("utf-8")))
        print(trajectories)
    except Exception as e:
        raise e
