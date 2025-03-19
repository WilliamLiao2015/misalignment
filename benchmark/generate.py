from random import choices

from torch.utils.data import DataLoader

from .dataset import create_dataset, collate_fn
from .visualize import visualize

def generate_configs():
    dataset = create_dataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    try: batch, configs = next(iter(dataloader))
    except: return generate_configs()

    for config in configs:
        activities = [{
            "type": test_type,
            "participants": [f"V{i + 1}"]
        }  for test_type, results in config["tests"].items() for i, result in enumerate(results) if result]
        config["activities"] = choices(activities, k=min(len(activities), 3))
    return batch, configs

if __name__ == "__main__":
    dataset = create_dataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    default_collate_fn = dataset.get_collate_fn()
    batch, configs = collate_fn([dataset[35000]])
    # visualize(dataset.cache_path, default_collate_fn(batch), 0, interactive=False)

    from pprint import pprint
    pprint([config["activities"] for config in generate_configs()[1]])
