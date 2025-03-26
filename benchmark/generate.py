from random import choices, randint

from torch.utils.data import DataLoader

from .dataset import create_dataset, collate_fn
from .visualize import visualize

def generate_configs(num_configs: int = 1, standardize_data: bool = False):
    dataset = create_dataset(standardize_data=False)
    standardize_dataset = create_dataset(standardize_data=True) # ego-centric
    indices = [randint(0, len(dataset) - 1) for _ in range(num_configs)]

    try: batch, configs = collate_fn([dataset[i] for i in indices])
    except: return None, []

    for config in configs:
        activities = [{
            "type": test_type,
            "participants": [f"V{i + 1}"]
        }  for test_type, results in config["tests"].items() for i, result in enumerate(results) if result is not None and test_type != "longitudinal:standing-still"]
        config["activities"] = choices(activities, k=min(len(activities), 3))

    configs = [config for config in configs if len(config["activities"]) > 0]

    default_collate_fn = dataset.get_collate_fn()
    visualize(dataset.cache_path, default_collate_fn(batch), 0, interactive=False)

    if standardize_data:
        try: standardize_batch, standardize_configs = collate_fn([standardize_dataset[i] for i in indices])
        except: return None, []
        default_collate_fn = standardize_dataset.get_collate_fn()
        visualize(standardize_dataset.cache_path, default_collate_fn(standardize_batch), 0, interactive=False)
        for i, config in enumerate(configs):
            standardize_configs[i]["activities"] = config["activities"]
        return standardize_batch, standardize_configs

    return batch, configs

if __name__ == "__main__":
    dataset = create_dataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    default_collate_fn = dataset.get_collate_fn()
    batch, configs = collate_fn([dataset[35000]])
    # visualize(dataset.cache_path, default_collate_fn(batch), 0, interactive=False)

    from pprint import pprint
    pprint([config["activities"] for config in generate_configs()[1]])
