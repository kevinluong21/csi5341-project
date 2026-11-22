import json
from pathlib import Path
import numpy as np
import csv
import torch

from evaluation import evaluate
from load_inference import load_prediction_ply

PROJECT_ROOT = Path(r"C:\Users\xhn20\dev\diffts\DiffTS\DiffTS\csi5341-project")
DATASETS_ROOT = PROJECT_ROOT / "datasets"
VESSEL_DATA_ROOT = DATASETS_ROOT / "dvn"
WIRE_PT_ROOT = PROJECT_ROOT.parent / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"

PATHS = [
    (
        "vessel_orchard",
        "vessel",
        "orchard_vessel_inference",
        "point_vessel_orchard_scale_values.csv",
    ),
    (
        "vessel_synthetic",
        "vessel",
        "synthetic_vessel_inference",
        "point_vessel_synthetictree_scale_values.csv",
    ),
    (
        "wire_orchard",
        "wire",
        "orchard_wire_inference",
        "PointWireDiffTSOrchard",
    ),
    (
        "wire_synthetic",
        "wire",
        "synthetic_wire_inference",
        "PointWireDiffTSSynthetictree",
    ),
    (
        "wire_treenet",
        "wire",
        "treenet_wire_inference",
        "PointWireDiffTSTreenet3d",
    ),
]

def runner_vessel(pred_dir, data_root, scale_csv):
    samples = []
    predictions = {}

    with scale_csv.open(newline="", encoding="utf-8") as file:
        scale_factors = {
            (int(row["set_id"]), int(row["sample_id"])): float(row["scale_factor"])
            for row in csv.DictReader(file)
        }

    for ply in sorted(pred_dir.glob("*_0_pred_pp_edges.ply")):
        # 0118_0000_0_pred_pp_edges.ply -> sample id: 0118_0000
        set_id, frame_id = ply.name.split("_", 2)[:2]
        sample_id = f"{set_id}_{frame_id}"

        folder = data_root / set_id 
        points = np.load(folder / "pcl_2048" / f"pcl_{frame_id}.npy")
        labels = np.load(folder / "seg_2048" / f"seg_{frame_id}.npy")
        
        # scale 
        scale = scale_factors[(int(set_id), int(frame_id))]
        points = (points - points.min(axis=0)) * scale

        # print(f"scale for {sample_id}: {scale}")

        samples.append({
            "sample_id": sample_id, 
            "points": points, 
            "true_labels": labels
        })
        predictions[sample_id] = load_prediction_ply(ply)

    assert len(samples) > 0, f"no samples"

    return evaluate(samples, predictions, "vessel")

def runner_wire(pred_dir, pt_dir):
    samples = []
    predictions = {}

    for ply in sorted(pred_dir.glob("*_0_pred_pp_edges.ply")):
        set_id, frame_id = ply.name.split("_", 2)[:2]
        sample_id = f"{set_id}_{frame_id}"

        data = torch.load(pt_dir / f"{sample_id}.pt", weights_only=False)
        points = np.asarray(data["scan_points"], dtype=np.float64)
        labels = np.asarray(data["gt_scan_point_classes"], dtype=np.int64).reshape(-1)

        # 0 wire, 1 endpoint, 2 bifurcation
        keep = np.isin(labels, [0, 1, 2])

        samples.append({
            "sample_id": sample_id,
            "points": points[keep],
            "true_labels": labels[keep],
        })
        predictions[sample_id] = load_prediction_ply(ply)

    assert len(samples) > 0, f"no samples"

    return evaluate(samples, predictions, "wire")

def main():
    for name, kind, folder, reference in PATHS:
            print(f"Evaluating {name}...", flush=True)

            pred_dir = DATASETS_ROOT / folder / "generated_pcd"
    
            if kind == "vessel":
                result = runner_vessel(
                    pred_dir,
                    VESSEL_DATA_ROOT,
                    DATASETS_ROOT / folder / reference,
                )
            else:
                result = runner_wire(
                    pred_dir,
                    WIRE_PT_ROOT / reference / "test",
                )
    
            output_file = OUTPUT_DIR / f"{name}_results.json"

            output_file.write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
    
            scores = result["dataset_metrics"]
            
            print(
                f"{name}: {result['number_of_samples']} samples, "
                f"mIoU={scores['mIoU']:.4f}, "
                f"OA={scores['OA']:.4f}; "
                f"saved {output_file}",
                flush=True,
            )


if __name__ == "__main__":
    main()