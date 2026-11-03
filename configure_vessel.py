import os
import numpy as np
import yaml
from dataloading.point_vessel import VesselData

vessel_data = VesselData("datasets/PointVessel/dvn")

if os.path.exists("datasets/PointVesselDiffTS"):
    raise Exception("PointVesselDiffTS already exists. If you want to re-generate the dataset, delete this folder and run the script again.")
else:
    os.mkdir("datasets/PointVesselDiffTS")

    # make the subfolders with the splits
    os.mkdir("datasets/PointVesselDiffTS/train")
    os.mkdir("datasets/PointVesselDiffTS/val")
    os.mkdir("datasets/PointVesselDiffTS/test")

    splits = dict()

    splits["train"] = vessel_data.get_train_split()
    splits["val"] = vessel_data.get_val_split()
    splits["test"] = vessel_data.get_test_split()

    for split_name, split_sets in splits.items():
        # save the point cloud into a new folder that is identified using its set id followed by the sample id
        for set_id in split_sets:
            for sample_id in vessel_data.get_samples_per_set():
                point_cloud = vessel_data.load_pcl(set_id, sample_id)
                np.save(f"datasets/PointVesselDiffTS/{split_name}/" + "{:04d}_{:04d}.npy".format(set_id, sample_id), point_cloud)


config = dict()

config["experiment"] = {"id": "point_vessel"}
config["data"] = {
    "data_dir": "Datasets/PointVesselDiffTS",
    "dataloader": "PointVessel",
    "num_nodes": "auto",
    "num_points": 2048, # this number comes directly from the author's script
}