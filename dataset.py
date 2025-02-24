from trajdata import UnifiedDataset

# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["waymo_val"],
    data_dirs={  # Remember to change this to match your filesystem!
        "waymo_val": "../../data/waymo"
    }
)
