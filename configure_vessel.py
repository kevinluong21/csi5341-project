import os
import numpy as np
import yaml
import torch
import pandas as pd
from dataloading.point_vessel import VesselData
import click
import open3d as o3d

vessel_data = VesselData("datasets/PointVessel/dvn")

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--model',
              '-m',
              type=str,
              help='Can be either "orchard", "treenet3d", or "synthetictree"',
              default='orchard')
def main(model: str):
    if (model is None) or (model.lower() not in ["orchard", "treenet3d", "synthetictree"]):
        raise ValueError('Must be either "orchard", "treenet3d", or "synthetictree"')

    print("Generating test set for", model, "...")

    if os.path.exists(f"../Datasets/PointVesselDiffTS{model.capitalize()}"):
        pass
    else:
        os.mkdir(f"../Datasets/PointVesselDiffTS{model.capitalize()}")

        # make the subfolders with the splits
        # os.mkdir("datasets/PointVesselDiffTS/train")
        # os.mkdir("datasets/PointVesselDiffTS/val")
        os.mkdir(f"../Datasets/PointVesselDiffTS{model.capitalize()}/test")

    splits = dict()

    scales = []

    rescale = False

    # use the demo tree scan from the original dataset as a guide for the scale of our vessel point cloud
    if model == "orchard":
        demo_tree = torch.load("../Datasets/demo/test/lane1_tree_13_24_07_25_lane1.pt", weights_only=False)
        rescale = True # their config files do not have normalization on which can affect the scale of trees, so the point vessel has to be rescaled
    elif model == "synthetictree":
        demo_tree = torch.load("", weights_only=False)
        rescale = True # their config files do not have normalization on which can affect the scale of trees, so the point vessel has to be rescaled
    else:
        # for treenet3d, use the merge point cloud which includes both the leaves and the branches as a point cloud
        pcd = o3d.io.read_point_cloud("../Datasets/demo/test/mergePointClouds.ply")
        # convert the tree to numpy
        pcd = np.asarray(pcd.points)

        demo_tree = {"scan_points": pcd}


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

                if rescale:
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

                    scales.append({
                        "set_id": set_id,
                        "sample_id": sample_id,
                        "pcl_max": pcl_max,
                        "pcl_min": pcl_min,
                        "scale_factor": s,
                    })

                    # reuse the pcl_max and min for the skeleton so that both are on the same scale
                    pcl_scaled = (point_cloud - pcl_min) * s
                    skeleton_scaled = (skeleton["nodes"] - pcl_min) * s
                else:
                    pcl_scaled = point_cloud
                    skeleton_scaled = skeleton

                # skeleton contains 2 keys: nodes and adj (adjacency matrix which stores all the edges)
                data = {
                    "scan_points": pcl_scaled,
                    "skeleton_vertices": skeleton_scaled,
                    "skeleton_edges": skeleton["adj"],
                    "scan_point_classes": segments, # classes are 0 for wire/vessel and 1 for bifurcation
                    "scan_point_colors": np.array([colors[int(i)] for i in segments], dtype=np.float32),
                }

                torch.save(data, f"../Datasets/PointVesselDiffTS{model.capitalize()}/{split_name}/" + "{:04d}_{:04d}.pt".format(set_id, sample_id))


    if model == "orchard":
        with open("configs/config_orchard.yaml", "r") as file:
            config = yaml.safe_load(file)

        config["experiment"]["id"] = "point_vessel_orchard"

        config["data"]["data_dir"] = f"Datasets/PointVesselDiffTS{model.capitalize()}"
        config["data"]["dataloader"] = "PointVessel"
        config["data"]["varieties"] = ["Vessel"]
        config["data"]["num_nodes"] = 3000 # default number of nodes (and the same number used in the paper for the orchard dataset)

        with open("../pretrained_models/point_vessel_orchard.yaml", "w", encoding="utf-8") as file:
            yaml.dump(config, file)

        scales = pd.DataFrame(scales)
        scales.to_csv(f"../Datasets/PointVesselDiffTS{model.capitalize()}/point_vessel_orchard_scale_values.csv", index=False)

    elif model == "synthetictree":
        with open("configs/config_syntree.yaml", "r") as file:
            config = yaml.safe_load(file)

        config["experiment"]["id"] = "point_vessel_syntree"

        config["data"]["data_dir"] = f"Datasets/PointVesselDiffTS{model.capitalize()}"
        config["data"]["dataloader"] = "PointVessel"
        config["data"]["varieties"] = ["Vessel"]
        config["data"]["num_nodes"] = 3000 # default number of nodes

        with open("../pretrained_models/point_vessel_syntree.yaml", "w", encoding="utf-8") as file:
            yaml.dump(config, file)

        scales = pd.DataFrame(scales)
        scales.to_csv(f"../Datasets/PointVesselDiffTS{model.capitalize()}/point_vessel_synthetictree_scale_values.csv", index=False)

    else:
        with open("configs/config_treenet.yaml", "r") as file:
            config = yaml.safe_load(file)

        config["experiment"]["id"] = "point_vessel_treenet"

        config["data"]["data_dir"] = f"Datasets/PointVesselDiffTS{model.capitalize()}"
        config["data"]["dataloader"] = "PointVessel"
        config["data"]["varieties"] = ["Vessel"]
        # the number of nodes is already defined in the original config file, so no need to overwrite

        with open("../pretrained_models/point_vessel_treenet.yaml", "w", encoding="utf-8") as file:
            yaml.dump(config, file)

        scales = pd.DataFrame(scales)
        scales.to_csv(f"../Datasets/PointVesselDiffTS{model.capitalize()}/point_vessel_treenet3d_scale_values.csv", index=False)

if __name__ == "__main__":
    main(auto_envvar_prefix='PLS')