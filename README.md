# csi5341-project

**NOTE**: At the moment, the files only generate the point clouds for running on the DiffTS model trained on the Orchard dataset.
You can not expect good results if you run the PointVessel point clouds on the DiffTS model trained on other datasets at this time.

### Instructions:
1) Download the PointVessel dataset and place the folder in the `datasets` folder as `datasets/PointVessel/dvn`.
2) Place the entirety of the repositor, `csi5431-project`, inside of `DiffTS/DiffTS `.
3) Move the `PointVessel.py` file into the `DiffTS/DiffTS/Datasets/dataloader` folder.
4) Move the `datasets.py` file into the `DiffTS/DiffTS/Datasets/dataloader` folder.
5) Replace the Makefile, Dockerfile, and requirements.txt file with the one in this repo.
6) To run the test, run `make test WEIGHTS=pretrained_models/orchard_model.ckpt CONFIG=pretrained_models/point_vessel_orchard.yaml`.
7) The output point clouds are stored in `DiffTS/logs` (the parent folder of the entire repo) under the ID defined in the config of your test run.
If you want to see the edges, use `open3d.io.read_line_set()`, pass the file ending in `pred_pp_edges.ply`, then run `o3d.visualization.draw_geometries()`.