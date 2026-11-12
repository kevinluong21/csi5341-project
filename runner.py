import json
from pathlib import Path
import numpy as np
import csv

from evaluation import evaluate
from load_inference import load_prediction_ply

def runner(pred_dir, data_root, scale_csv):
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

        print(f"scale for {sample_id}: {scale}")

        samples.append({"sample_id": sample_id, "points": points, "true_labels": labels})
        predictions[sample_id] = load_prediction_ply(ply)

    assert len(samples) > 0, f"no samples"

    return evaluate(samples, predictions, "vessel")

def main():
    repo = Path("C:/Users/xhn20/dev/csi5341-project")
    pred_dir = repo / "output"
    data_root = repo / "datasets" / "dvn"
    scale_csv = repo / "datasets" / "point_vessel_orchard_scale_values(in).csv"
    output = repo / "results.json"

    result = runner(
        pred_dir=pred_dir,
        data_root=data_root,
        scale_csv=scale_csv
    )

    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    
    print(f"{result['number_of_samples']} samples, {result['number_of_points']} points")
    print(json.dumps(result["dataset_metrics"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()