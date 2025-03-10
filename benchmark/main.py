import yaml

type_map = {
    "longitudinal:driving-forward:accelerating": "Accelerating",
    "longitudinal:driving-forward:cruising": "Cruising",
    "longitudinal:driving-forward:decelerating": "Decelerating",
    "longitudinal:driving-forward:standing-still": "Standing Still",
    "longitudinal:driving-forward:reverse": "Reversing",
    "lateral:going-straight": "Going Straight",
    "lateral:turning:right": "Turning Right",
    "lateral:turning-left": "Turning Left"
}

def get_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

def describe_config(config: dict) -> str:
    description = ""
    for activity in config["activities"]:
        description += f"{' and '.join(activity['participants'])} {'is' if len(activity['participants']) == 1 else 'are'} {type_map[activity['type']]}, "
    return f"{description[:-2]}."

if __name__ == "__main__":
    config = get_config("benchmark/configs/turning-right.yaml")

    from pprint import pprint
    pprint(config["activities"])
    print(describe_config(config))
