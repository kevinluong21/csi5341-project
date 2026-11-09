import os
import numpy as np
import yaml
import torch
from dataloading.point_vessel import VesselData

# NOTE: at the moment, you can only run the point clouds using the DiffTS model trained on the orchard dataset

vessel_data = VesselData("datasets/PointVessel/dvn")

if os.path.exists("../Datasets/PointVesselDiffTS"):
    pass
else:
    os.mkdir("../Datasets/PointVesselDiffTS")

    # make the subfolders with the splits
    # os.mkdir("datasets/PointVesselDiffTS/train")
    # os.mkdir("datasets/PointVesselDiffTS/val")
    os.mkdir(r"../Datasets/PointVesselDiffTS/test")

    splits = dict()

    # use the demo tree scan from the orchard dataset as a guide for the scale of our vessel point cloud
    demo_tree = torch.load("../Datasets/demo/test/lane1_tree_13_24_07_25_lane1.pt", weights_only=False)

    # splits["train"] = vessel_data.get_train_split()
    # splits["val"] = vessel_data.get_val_split()
    splits["test"] = vessel_data.get_test_split()

    for split_name, split_sets in splits.items():
        # save the point cloud into a new folder that is identified using its set id followed by the sample id
        for set_id in split_sets:
            for sample_id in range(vessel_data.get_samples_per_set()):
                point_cloud = vessel_data.load_pcl(set_id, sample_id)
                skeleton = vessel_data.load_skeleton(set_id, sample_id)
                segments = vessel_data.load_seg(set_id, sample_id).astype(np.float32)

                # classes are 0 for wire/vessel and 1 for bifurcation, so wires/vessels are red and bifurcations are blue (normalized in RGB)
                colors = [
                    [255, 0, 0],
                    [0, 255, 0]
                ]

                # calculate the range for x, y, and z for the demo tree
                max_xyz = np.max(demo_tree["scan_points"], axis=0)
                min_xyz = np.min(demo_tree["scan_points"], axis=0)

                # of the 3 ranges, pick the largest one to use as our scale
                range_xyz = max_xyz - min_xyz
                max_range = np.max(range_xyz)

                # now that i know the max range, i have to scale my vessel point cloud and skeleton nodes to be within the same scale as the demo tree
                pcl_max = np.max(point_cloud, axis=0)
                pcl_min = np.min(point_cloud, axis=0)

                extent = pcl_max - pcl_min
                s = max_range / extent.max()

                # reuse the pcl_max and min for the skeleton so that both are on the same scale
                pcl_scaled = (point_cloud - pcl_min) * s
                skeleton_scaled = (skeleton["nodes"] - pcl_min) * s

                # skeleton contains 2 keys: nodes and adj (adjacency matrix which stores all the edges)
                data = {
                    "scan_points": pcl_scaled,
                    "skeleton_vertices": skeleton_scaled,
                    "skeleton_edges": skeleton["adj"],
                    "scan_point_classes": segments, # classes are 0 for wire/vessel and 1 for bifurcation
                    "scan_point_colors": np.array([colors[int(i)] for i in segments], dtype=np.float32),
                }

                # np.save(f"datasets/PointVesselDiffTS/{split_name}/" + "{:04d}_{:04d}.npy".format(set_id, sample_id), point_cloud)
                torch.save(data, f"../Datasets/PointVesselDiffTS/{split_name}/" + "{:04d}_{:04d}.pt".format(set_id, sample_id))

with open("configs/config_orchard.yaml", "r") as file:
    config = yaml.safe_load(file)

config["experiment"]["id"] = "point_vessel_orchard"

config["data"]["data_dir"] = "Datasets/PointVesselDiffTS"
config["data"]["dataloader"] = "PointVessel"
config["data"]["varieties"] = ["Vessel"]

with open("../pretrained_models/point_vessel_orchard.yaml", "w", encoding="utf-8") as file:
    yaml.dump(config, file)

# with open("configs/config_syntree.yaml", "r") as file:
#     config = yaml.safe_load(file)

# config["experiment"]["id"] = "point_vessel_syntree"

# config["data"]["data_dir"] = "Datasets/PointVesselDiffTS"
# config["data"]["dataloader"] = "PointVessel"
# config["data"]["varieties"] = ["Vessel"]

# with open("../pretrained_models/point_vessel_syntree.yaml", "w", encoding="utf-8") as file:
#     yaml.dump(config, file)

# with open("configs/config_treenet.yaml", "r") as file:
#     config = yaml.safe_load(file)

# config["experiment"]["id"] = "point_vessel_treenet"

# config["data"]["data_dir"] = "Datasets/PointVesselDiffTS"
# config["data"]["dataloader"] = "PointVessel"
# config["data"]["varieties"] = ["Vessel"]

# with open("../pretrained_models/point_vessel_treenet.yaml", "w", encoding="utf-8") as file:
#     yaml.dump(config, file)