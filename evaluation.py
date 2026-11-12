import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    jaccard_score,
    recall_score,
    accuracy_score,
)


# get classification radius
def get_radius(points):
    max_side = points.max(axis=0) - points.min(axis=0)
    max_diag = np.linalg.norm(max_side)
    return 0.05 * max_diag


# turn graph into points with predicted labels: [samples: [(point id, predicted label)]]
def mark_labels(points, nodes, edges, kind):
    points = np.asarray(points, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(edges)

    degrees = np.zeros(len(nodes))
    unique_edges = set()

    for edge in edges:
        node_a = int(edge[0])
        node_b = int(edge[1])

        normalized_edge = (min(node_a, node_b), max(node_a, node_b))
        if normalized_edge in unique_edges:
            continue

        degrees[node_a] += 1
        degrees[node_b] += 1

        unique_edges.add(normalized_edge)

    special_indices = []
    special_classes = []

    for index in range(len(nodes)):
        degree = degrees[index]

        if degree >= 3:
            special_indices.append(index)

            if kind == "vessel":
                special_classes.append(1)
            elif kind == "wire":
                special_classes.append(2)
            else:
                raise ValueError(f"invalid kind")

        #  wire endpoints
        if degree == 1 and kind == "wire":
            special_indices.append(index)
            special_classes.append(1)

    if len(special_indices) == 0:
        return np.zeros(len(points))

    special_indices = np.asarray(special_indices, dtype=np.int64)
    special_classes = np.asarray(special_classes, dtype=np.int64)

    predicted_labels = np.zeros(len(points))
    special_nodes = nodes[special_indices]
    radius = get_radius(points)

    # get point: special points: coordinate difference
    coordinate_differences = points[:, np.newaxis, :] - special_nodes[np.newaxis, :, :]

    # convert to point: special point: distance
    distances = np.linalg.norm(coordinate_differences, axis=2)

    # convert to point: nearest distance index
    nearest_special_index = np.argmin(distances, axis=1)

    # get point: nearest distance value
    point_indices = np.arange(len(points))
    nearest_distance = distances[point_indices, nearest_special_index]

    # get point: nearest special point class
    nearest_class = special_classes[nearest_special_index]

    # check if within radius
    points_inside_radius = nearest_distance <= radius

    # get point: final class (non special class defaults to 0)
    predicted_labels[points_inside_radius] = nearest_class[points_inside_radius]

    return predicted_labels


# calculate metrics
def get_metrics(true_labels, predicted_labels, class_ids):
    confusion = confusion_matrix(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=class_ids,
    )

    iou_per_class = jaccard_score(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=class_ids,
        average=None,
        zero_division=0,
    )

    accuracy_per_class = recall_score(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=class_ids,
        average=None,
        zero_division=0,
    )

    mean_iou = jaccard_score(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    mean_accuracy = recall_score(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    overall_accuracy = accuracy_score(
        y_true=true_labels,
        y_pred=predicted_labels,
    )

    return {
        "confusion_matrix": confusion.tolist(),
        "IoU": iou_per_class.tolist(),
        "accuracy_per_class": accuracy_per_class.tolist(),
        "mIoU": float(mean_iou),
        "mAcc": float(mean_accuracy),
        "OA": float(overall_accuracy),
    }


def evaluate(samples, predictions, kind):
    all_true_labels = []
    all_predicted_labels = []

    sample_results = []

    if kind == "wire":
        class_ids = [0, 1, 2]
    elif kind == "vessel":
        class_ids = [0, 1]
    else:
        raise ValueError("Invalid dataset kind")

    for sample in samples:
        sample_id = sample["sample_id"]
        points = np.asarray(sample["points"], dtype=np.float64)
        true_labels = np.asarray(sample["true_labels"], dtype=np.int64)

        assert sample_id  in predictions, f"Missing prediction for sample: {sample_id}"

        prediction = predictions[sample_id]

        nodes = prediction["nodes"]
        edges = prediction["edges"]

        predicted_labels = mark_labels(
            points=points,
            nodes=nodes,
            edges=edges,
            kind=kind,
        )

        assert len(true_labels) == len(predicted_labels), f"prediction labels do not match ground truth"

        all_true_labels.append(true_labels)
        all_predicted_labels.append(predicted_labels)

        sample_metrics = get_metrics(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            class_ids=class_ids,
        )

        sample_results.append(
            {
                "sample_id": sample_id,
                "number_of_points": len(points),
                "predicted_labels": predicted_labels.tolist(),
                "metrics": sample_metrics,
            }
        )

    assert len(all_true_labels) != 0, f"no samples"

    all_true_labels = np.concatenate(all_true_labels)
    all_predicted_labels = np.concatenate(all_predicted_labels)

    dataset_metrics = get_metrics(
        true_labels=all_true_labels,
        predicted_labels=all_predicted_labels,
        class_ids=class_ids,
    )

    return {
        "kind": kind,
        "number_of_samples": len(samples),
        "number_of_points": len(all_true_labels),
        "dataset_metrics": dataset_metrics,
        "sample_results": sample_results,
    }