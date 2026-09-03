from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import AnnotationSegment, load_annotations


TARGET_LABELS = ("forehand", "backhand", "other")
GROUNDSTROKES = {"forehand", "backhand"}
FEATURE_NAMES = (
    "dominant_wrist_x",
    "dominant_wrist_y",
    "dominant_wrist_vx",
    "dominant_wrist_vy",
    "dominant_wrist_speed",
    "dominant_arm_extension",
    "dominant_elbow_angle_rad",
    "wrist_distance",
    "shoulder_hip_angle_rad",
    "mean_keypoint_score",
)
DOMINANT_SHOULDER = 6
DOMINANT_ELBOW = 8
DOMINANT_WRIST = 10
NON_DOMINANT_WRIST = 9
RESAMPLE_POINTS = 21
PHASE_POINTS = 7
MOTION_PHASE_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)


@dataclass(frozen=True)
class StrokeEvent:
    video: str
    start: float
    end: float
    source_label: str
    target_label: str
    vector: np.ndarray
    sample_count: int


def target_label(source_label: str) -> str:
    return source_label if source_label in GROUNDSTROKES else "other"


def _finite_values(records: list[dict[str, Any]], name: str) -> np.ndarray:
    values = [record["features"].get(name) for record in records]
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )


def aggregate_event_vector(
    records: list[dict[str, Any]],
    segment: AnnotationSegment,
) -> np.ndarray:
    """Create a robust fixed-length description of one annotated event."""
    result: list[float] = [segment.end - segment.start]
    for name in FEATURE_NAMES:
        values = _finite_values(records, name)
        if values.size == 0:
            result.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        clipped = np.clip(values, np.percentile(values, 5), np.percentile(values, 95))
        result.extend(
            (
                float(np.median(clipped)),
                float(np.percentile(clipped, 10)),
                float(np.percentile(clipped, 90)),
                float(np.max(clipped) - np.min(clipped)),
                float(clipped[-1] - clipped[0]),
            )
        )

    wrist_speeds = _finite_values(records, "dominant_wrist_speed")
    peak_index = int(np.argmax(wrist_speeds)) if wrist_speeds.size else 0
    result.append(peak_index / max(wrist_speeds.size - 1, 1))
    return np.asarray(result, dtype=np.float64)


def _median_smooth(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values
    padded = np.pad(values, (1, 1), mode="edge")
    median = np.asarray(
        [np.median(padded[index : index + 3]) for index in range(values.size)],
        dtype=np.float64,
    )
    return np.convolve(np.pad(median, (1, 1), mode="edge"), (0.25, 0.5, 0.25), mode="valid")


def _resample_joint(
    timestamps: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    joint: int,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    coordinates = keypoints[:, joint]
    confidence = scores[:, joint]
    valid = (
        (confidence >= 0.35)
        & np.all(np.isfinite(coordinates), axis=1)
        & np.all(np.abs(coordinates) <= 6.0, axis=1)
    )
    valid_fraction = float(np.mean(valid))
    if np.count_nonzero(valid) < 2:
        valid = np.all(np.isfinite(coordinates), axis=1)
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(grid), np.zeros_like(grid), valid_fraction
    x = np.interp(grid, timestamps[valid], coordinates[valid, 0])
    y = np.interp(grid, timestamps[valid], coordinates[valid, 1])
    return _median_smooth(x), _median_smooth(y), valid_fraction


def _joint_angle_series(
    shoulder_x: np.ndarray,
    shoulder_y: np.ndarray,
    elbow_x: np.ndarray,
    elbow_y: np.ndarray,
    wrist_x: np.ndarray,
    wrist_y: np.ndarray,
) -> np.ndarray:
    first = np.column_stack((shoulder_x - elbow_x, shoulder_y - elbow_y))
    second = np.column_stack((wrist_x - elbow_x, wrist_y - elbow_y))
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-6,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def smoothed_event_vector(
    records: list[dict[str, Any]],
    segment: AnnotationSegment,
) -> np.ndarray:
    """Preserve swing phases after confidence-aware trajectory repair."""
    timestamps = np.asarray([float(record["time_seconds"]) for record in records])
    keypoints = np.asarray([record["normalized_keypoints"] for record in records], dtype=np.float64)
    scores = np.asarray([record["scores"] for record in records], dtype=np.float64)
    duration = max(float(timestamps[-1] - timestamps[0]), 1e-3)
    grid = np.linspace(float(timestamps[0]), float(timestamps[-1]), RESAMPLE_POINTS)

    wrist_x, wrist_y, wrist_valid = _resample_joint(
        timestamps, keypoints, scores, DOMINANT_WRIST, grid
    )
    other_wrist_x, other_wrist_y, other_wrist_valid = _resample_joint(
        timestamps, keypoints, scores, NON_DOMINANT_WRIST, grid
    )
    elbow_x, elbow_y, elbow_valid = _resample_joint(
        timestamps, keypoints, scores, DOMINANT_ELBOW, grid
    )
    shoulder_x, shoulder_y, shoulder_valid = _resample_joint(
        timestamps, keypoints, scores, DOMINANT_SHOULDER, grid
    )
    wrist_vx = np.gradient(wrist_x, grid)
    wrist_vy = np.gradient(wrist_y, grid)
    wrist_speed = np.hypot(wrist_vx, wrist_vy)
    extension = np.hypot(wrist_x - shoulder_x, wrist_y - shoulder_y)
    elbow_angle = _joint_angle_series(
        shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y
    )
    wrist_distance = np.hypot(wrist_x - other_wrist_x, wrist_y - other_wrist_y)

    result: list[float] = [
        segment.end - segment.start,
        duration,
        wrist_valid,
        other_wrist_valid,
        elbow_valid,
        shoulder_valid,
        float(np.mean([record["features"]["mean_keypoint_score"] for record in records])),
    ]
    series = (
        wrist_x,
        wrist_y,
        wrist_vx,
        wrist_vy,
        wrist_speed,
        extension,
        elbow_angle,
        wrist_distance,
    )
    for values in series:
        result.extend(
            (
                float(np.median(values)),
                float(np.percentile(values, 10)),
                float(np.percentile(values, 90)),
                float(np.max(values) - np.min(values)),
                float(values[-1] - values[0]),
            )
        )

    phase_indices = np.linspace(0, RESAMPLE_POINTS - 1, PHASE_POINTS).round().astype(int)
    for values in (wrist_x, wrist_y, wrist_vx, wrist_vy, extension, elbow_angle):
        result.extend(float(value) for value in values[phase_indices])
    peak_index = int(np.argmax(wrist_speed))
    result.extend(
        (
            peak_index / max(RESAMPLE_POINTS - 1, 1),
            float(wrist_x[peak_index]),
            float(wrist_y[peak_index]),
            float(wrist_x[-1] - wrist_x[0]),
            float(wrist_y[-1] - wrist_y[0]),
        )
    )
    return np.asarray(result, dtype=np.float64)


def motion_phase_event_vector(
    records: list[dict[str, Any]],
    segment: AnnotationSegment,
) -> np.ndarray:
    """Describe a swing at multiple cumulative-motion anchors, not one speed peak."""
    timestamps = np.asarray([float(record["time_seconds"]) for record in records])
    keypoints = np.asarray(
        [record["normalized_keypoints"] for record in records],
        dtype=np.float64,
    )
    scores = np.asarray([record["scores"] for record in records], dtype=np.float64)
    duration = max(float(timestamps[-1] - timestamps[0]), 1e-3)
    grid = np.linspace(float(timestamps[0]), float(timestamps[-1]), RESAMPLE_POINTS)

    wrist_x, wrist_y, wrist_valid = _resample_joint(
        timestamps, keypoints, scores, DOMINANT_WRIST, grid
    )
    other_wrist_x, other_wrist_y, other_wrist_valid = _resample_joint(
        timestamps, keypoints, scores, NON_DOMINANT_WRIST, grid
    )
    elbow_x, elbow_y, elbow_valid = _resample_joint(
        timestamps, keypoints, scores, DOMINANT_ELBOW, grid
    )
    shoulder_x, shoulder_y, shoulder_valid = _resample_joint(
        timestamps, keypoints, scores, DOMINANT_SHOULDER, grid
    )
    wrist_vx = np.gradient(wrist_x, grid)
    wrist_vy = np.gradient(wrist_y, grid)
    wrist_speed = np.hypot(wrist_vx, wrist_vy)
    extension = np.hypot(wrist_x - shoulder_x, wrist_y - shoulder_y)
    elbow_angle = _joint_angle_series(
        shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y
    )
    wrist_distance = np.hypot(wrist_x - other_wrist_x, wrist_y - other_wrist_y)

    increments = 0.5 * (wrist_speed[:-1] + wrist_speed[1:]) * np.diff(grid)
    cumulative_motion = np.concatenate(([0.0], np.cumsum(increments)))
    total_motion = float(cumulative_motion[-1])
    if total_motion > 1e-6:
        phase_indices = np.asarray(
            [
                min(
                    int(np.searchsorted(cumulative_motion, total_motion * quantile)),
                    RESAMPLE_POINTS - 1,
                )
                for quantile in MOTION_PHASE_QUANTILES
            ],
            dtype=int,
        )
    else:
        phase_indices = np.linspace(
            0,
            RESAMPLE_POINTS - 1,
            len(MOTION_PHASE_QUANTILES),
        ).round().astype(int)

    result: list[float] = [
        segment.end - segment.start,
        duration,
        wrist_valid,
        other_wrist_valid,
        elbow_valid,
        shoulder_valid,
        float(np.mean([record["features"]["mean_keypoint_score"] for record in records])),
    ]
    result.extend(
        float(index) / max(RESAMPLE_POINTS - 1, 1) for index in phase_indices
    )
    phase_series = (
        wrist_x,
        wrist_y,
        wrist_vx,
        wrist_vy,
        wrist_speed,
        extension,
        elbow_angle,
        wrist_distance,
        other_wrist_x,
        other_wrist_y,
    )
    for values in phase_series:
        result.extend(float(values[index]) for index in phase_indices)

    for values in (wrist_x, wrist_y, other_wrist_x, other_wrist_y, extension):
        anchored = values[phase_indices]
        result.extend(float(value) for value in np.diff(anchored))

    return np.asarray(result, dtype=np.float64)


def load_events(
    pose_log: Path,
    annotation_path: Path,
    feature_mode: str = "smoothed",
) -> list[StrokeEvent]:
    metadata, segments = load_annotations(annotation_path)
    records = [json.loads(line) for line in pose_log.open() if line.strip()]
    events: list[StrokeEvent] = []
    for segment in segments:
        if segment.label == "unknown":
            continue
        selected = [
            record
            for record in records
            if segment.start <= float(record["time_seconds"]) < segment.end
        ]
        if len(selected) < 3:
            continue
        events.append(
            StrokeEvent(
                video=str(metadata.get("video", annotation_path.stem)),
                start=segment.start,
                end=segment.end,
                source_label=segment.label,
                target_label=target_label(segment.label),
                vector=(
                    motion_phase_event_vector(selected, segment)
                    if feature_mode == "motion-phases"
                    else smoothed_event_vector(selected, segment)
                    if feature_mode == "smoothed"
                    else aggregate_event_vector(selected, segment)
                ),
                sample_count=len(selected),
            )
        )
    return events


def discover_events(
    pose_dir: Path,
    annotation_dir: Path,
    feature_mode: str = "smoothed",
) -> list[StrokeEvent]:
    events: list[StrokeEvent] = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        pose_log = pose_dir / f"{annotation_path.stem}-pose.jsonl"
        if not pose_log.exists():
            print(f"Skipping {annotation_path.name}: missing {pose_log}")
            continue
        events.extend(load_events(pose_log, annotation_path, feature_mode))
    return events


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(train, axis=0)
    q25, q75 = np.percentile(train, (25, 75), axis=0)
    scale = q75 - q25
    standard_deviation = np.std(train, axis=0)
    scale = np.where(scale > 1e-6, scale, standard_deviation)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def predict_knn(
    train_events: list[StrokeEvent],
    test_events: list[StrokeEvent],
    labels: tuple[str, ...] = TARGET_LABELS,
    k: int = 3,
    binary_groundstroke: bool = False,
) -> list[str]:
    train = np.stack([event.vector for event in train_events])
    test = np.stack([event.vector for event in test_events])
    train, test = _standardize(train, test)
    predictions: list[str] = []
    k = min(k, len(train_events))
    for vector in test:
        distances = np.linalg.norm(train - vector, axis=1)
        nearest = np.argsort(distances)[:k]
        votes = {label: 0.0 for label in labels}
        for index in nearest:
            event_label = train_events[int(index)].target_label
            label = (
                "groundstroke"
                if binary_groundstroke and event_label in GROUNDSTROKES
                else event_label
            )
            votes[label] += 1.0 / max(float(distances[index]), 1e-6)
        predictions.append(max(labels, key=lambda label: votes[label]))
    return predictions


def predict_knn_diagnostics(
    train_events: list[StrokeEvent],
    test_events: list[StrokeEvent],
    labels: tuple[str, ...] = TARGET_LABELS,
    k: int = 3,
    binary_groundstroke: bool = False,
) -> list[dict[str, Any]]:
    """Return k-NN labels plus a nearest-class distance margin."""
    train = np.stack([event.vector for event in train_events])
    test = np.stack([event.vector for event in test_events])
    train, test = _standardize(train, test)
    mapped_labels = [
        "groundstroke"
        if binary_groundstroke and event.target_label in GROUNDSTROKES
        else event.target_label
        for event in train_events
    ]
    k = min(k, len(train_events))
    diagnostics: list[dict[str, Any]] = []
    for vector in test:
        distances = np.linalg.norm(train - vector, axis=1)
        nearest = np.argsort(distances)[:k]
        votes = {label: 0.0 for label in labels}
        for index in nearest:
            votes[mapped_labels[int(index)]] += 1.0 / max(
                float(distances[index]),
                1e-6,
            )
        prediction = max(labels, key=lambda label: votes[label])
        class_distances = {
            label: min(
                (
                    float(distances[index])
                    for index, event_label in enumerate(mapped_labels)
                    if event_label == label
                ),
                default=float("inf"),
            )
            for label in labels
        }
        predicted_distance = class_distances[prediction]
        alternatives = {
            label: distance
            for label, distance in class_distances.items()
            if label != prediction
        }
        runner_up = min(alternatives, key=alternatives.get)
        runner_up_distance = alternatives[runner_up]
        denominator = predicted_distance + runner_up_distance
        confidence = (
            runner_up_distance / denominator
            if np.isfinite(denominator) and denominator > 1e-9
            else 1.0
        )
        diagnostics.append(
            {
                "prediction": prediction,
                "confidence": float(confidence),
                "nearest_distance": float(predicted_distance),
                "runner_up": runner_up,
                "runner_up_distance": float(runner_up_distance),
                "votes": {label: float(votes[label]) for label in labels},
            }
        )
    return diagnostics


def predict_hierarchical(
    aggregate_train: list[StrokeEvent],
    aggregate_test: list[StrokeEvent],
    smoothed_train: list[StrokeEvent],
    smoothed_test: list[StrokeEvent],
    gate_k: int,
    side_k: int,
) -> list[str]:
    gate = predict_knn(
        aggregate_train,
        aggregate_test,
        labels=("groundstroke", "other"),
        k=gate_k,
        binary_groundstroke=True,
    )
    groundstroke_train = [
        event for event in smoothed_train if event.target_label in GROUNDSTROKES
    ]
    side = predict_knn(
        groundstroke_train,
        smoothed_test,
        labels=("forehand", "backhand"),
        k=side_k,
    )
    return [
        side_label if gate_label == "groundstroke" else "other"
        for gate_label, side_label in zip(gate, side)
    ]


def predict_hierarchical_diagnostics(
    aggregate_train: list[StrokeEvent],
    aggregate_test: list[StrokeEvent],
    temporal_train: list[StrokeEvent],
    temporal_test: list[StrokeEvent],
    gate_k: int,
    side_k: int,
) -> list[dict[str, Any]]:
    gate = predict_knn_diagnostics(
        aggregate_train,
        aggregate_test,
        labels=("groundstroke", "other"),
        k=gate_k,
        binary_groundstroke=True,
    )
    groundstroke_train = [
        event for event in temporal_train if event.target_label in GROUNDSTROKES
    ]
    side = predict_knn_diagnostics(
        groundstroke_train,
        temporal_test,
        labels=("forehand", "backhand"),
        k=side_k,
    )
    result: list[dict[str, Any]] = []
    for gate_result, side_result in zip(gate, side):
        prediction = (
            side_result["prediction"]
            if gate_result["prediction"] == "groundstroke"
            else "other"
        )
        confidence = (
            min(gate_result["confidence"], side_result["confidence"])
            if gate_result["prediction"] == "groundstroke"
            else gate_result["confidence"]
        )
        result.append(
            {
                "prediction": prediction,
                "confidence": float(confidence),
                "gate": gate_result,
                "side": side_result,
            }
        )
    return result


def evaluate(events: list[StrokeEvent]) -> dict[str, Any]:
    videos = sorted({event.video for event in events})
    confusion = {actual: Counter() for actual in TARGET_LABELS}
    mistakes: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    total = 0
    correct = 0
    for video in videos:
        test_events = [event for event in events if event.video == video]
        train_events = [event for event in events if event.video != video]
        if not test_events or not train_events:
            continue
        predictions = predict_knn(train_events, test_events)
        fold_correct = 0
        for event, prediction in zip(test_events, predictions):
            confusion[event.target_label][prediction] += 1
            total += 1
            if prediction == event.target_label:
                correct += 1
                fold_correct += 1
            else:
                mistakes.append(
                    {
                        "video": event.video,
                        "start": event.start,
                        "end": event.end,
                        "actual": event.target_label,
                        "source_label": event.source_label,
                        "predicted": prediction,
                    }
                )
        folds.append(
            {
                "video": video,
                "events": len(test_events),
                "correct": fold_correct,
                "accuracy": fold_correct / len(test_events),
            }
        )
    return {
        "method": "event-level 3-nearest-neighbors",
        "split": "leave-one-video-out",
        "events": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion": {label: dict(confusion[label]) for label in TARGET_LABELS},
        "folds": folds,
        "mistakes": mistakes,
    }


def evaluate_hierarchical(
    aggregate_events: list[StrokeEvent],
    smoothed_events: list[StrokeEvent],
) -> dict[str, Any]:
    aggregate_keys = [
        (event.video, event.start, event.end, event.source_label)
        for event in aggregate_events
    ]
    smoothed_keys = [
        (event.video, event.start, event.end, event.source_label)
        for event in smoothed_events
    ]
    if aggregate_keys != smoothed_keys:
        raise ValueError("aggregate and smoothed event sets do not align")

    videos = sorted({event.video for event in aggregate_events})
    confusion = {actual: Counter() for actual in TARGET_LABELS}
    mistakes: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    candidates = ((gate_k, side_k) for gate_k in (1, 3, 5, 7) for side_k in (1, 3, 5, 7))
    candidate_pairs = tuple(candidates)

    for held_out_video in videos:
        training_videos = [video for video in videos if video != held_out_video]
        scored_candidates: list[tuple[float, int, int]] = []
        for gate_k, side_k in candidate_pairs:
            inner_correct = 0
            inner_total = 0
            for validation_video in training_videos:
                inner_training = {
                    video for video in training_videos if video != validation_video
                }
                aggregate_train = [
                    event for event in aggregate_events if event.video in inner_training
                ]
                aggregate_test = [
                    event for event in aggregate_events if event.video == validation_video
                ]
                smoothed_train = [
                    event for event in smoothed_events if event.video in inner_training
                ]
                smoothed_test = [
                    event for event in smoothed_events if event.video == validation_video
                ]
                predictions = predict_hierarchical(
                    aggregate_train,
                    aggregate_test,
                    smoothed_train,
                    smoothed_test,
                    gate_k,
                    side_k,
                )
                inner_correct += sum(
                    prediction == event.target_label
                    for prediction, event in zip(predictions, smoothed_test)
                )
                inner_total += len(smoothed_test)
            score = inner_correct / inner_total if inner_total else 0.0
            scored_candidates.append((score, -gate_k, -side_k))

        _, negative_gate_k, negative_side_k = max(scored_candidates)
        gate_k, side_k = -negative_gate_k, -negative_side_k
        aggregate_train = [
            event for event in aggregate_events if event.video != held_out_video
        ]
        aggregate_test = [
            event for event in aggregate_events if event.video == held_out_video
        ]
        smoothed_train = [
            event for event in smoothed_events if event.video != held_out_video
        ]
        smoothed_test = [
            event for event in smoothed_events if event.video == held_out_video
        ]
        predictions = predict_hierarchical(
            aggregate_train,
            aggregate_test,
            smoothed_train,
            smoothed_test,
            gate_k,
            side_k,
        )
        fold_correct = 0
        for event, prediction in zip(smoothed_test, predictions):
            confusion[event.target_label][prediction] += 1
            if prediction == event.target_label:
                fold_correct += 1
            else:
                mistakes.append(
                    {
                        "video": event.video,
                        "start": event.start,
                        "end": event.end,
                        "actual": event.target_label,
                        "source_label": event.source_label,
                        "predicted": prediction,
                    }
                )
        folds.append(
            {
                "video": held_out_video,
                "events": len(smoothed_test),
                "correct": fold_correct,
                "accuracy": fold_correct / len(smoothed_test),
                "gate_k": gate_k,
                "side_k": side_k,
            }
        )

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in TARGET_LABELS)
    return {
        "method": "nested hierarchical event-level nearest-neighbors",
        "split": "outer leave-one-video-out; inner leave-one-video-out model selection",
        "events": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion": {label: dict(confusion[label]) for label in TARGET_LABELS},
        "folds": folds,
        "mistakes": mistakes,
    }


def print_report(events: list[StrokeEvent], report: dict[str, Any]) -> None:
    source_counts = Counter(event.source_label for event in events)
    target_counts = Counter(event.target_label for event in events)
    print("Event inventory: " + ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())))
    print("Evaluation classes: " + ", ".join(f"{k}={target_counts[k]}" for k in TARGET_LABELS))
    print(
        f"Leave-one-video-out accuracy: {report['correct']}/{report['events']} "
        f"({report['accuracy']:.1%})"
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("actual\\pred " + " ".join(f"{label:>10}" for label in TARGET_LABELS))
    for actual in TARGET_LABELS:
        row = report["confusion"][actual]
        print(f"{actual:>11} " + " ".join(f"{row.get(label, 0):>10}" for label in TARGET_LABELS))
    print("Per-video folds:")
    for fold in report["folds"]:
        parameters = (
            f", gate k={fold['gate_k']}, side k={fold['side_k']}"
            if "gate_k" in fold
            else ""
        )
        print(
            f"  {fold['video']}: {fold['correct']}/{fold['events']} "
            f"({fold['accuracy']:.1%}){parameters}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an offline tennis-stroke baseline")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument(
        "--method",
        choices=(
            "hierarchical",
            "hierarchical-phases",
            "aggregate",
            "smoothed",
            "motion-phases",
        ),
        default="hierarchical",
        help="Evaluation feature/classifier design (default: hierarchical)",
    )
    parser.add_argument("--report", type=Path, help="Optionally write the full JSON report")
    args = parser.parse_args()

    feature_mode = (
        "aggregate"
        if args.method == "aggregate"
        else "motion-phases"
        if args.method in {"hierarchical-phases", "motion-phases"}
        else "smoothed"
    )
    events = discover_events(args.pose_dir, args.annotations, feature_mode)
    if len({event.video for event in events}) < 2:
        raise SystemExit("At least two videos with pose logs and annotations are required")
    if args.method in {"hierarchical", "hierarchical-phases"}:
        aggregate_events = discover_events(args.pose_dir, args.annotations, "aggregate")
        report = evaluate_hierarchical(aggregate_events, events)
    else:
        report = evaluate(events)
        report["method"] = f"event-level 3-nearest-neighbors ({args.method} features)"
    print_report(events, report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
