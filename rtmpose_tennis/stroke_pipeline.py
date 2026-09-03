from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .annotations import AnnotationSegment, load_annotations
from .stroke_analysis import (
    GROUNDSTROKES,
    TARGET_LABELS,
    StrokeEvent,
    aggregate_event_vector,
    motion_phase_event_vector,
    predict_hierarchical,
    predict_hierarchical_diagnostics,
    smoothed_event_vector,
    target_label,
)
from .stroke_segmentation import (
    VideoTimeline,
    discover_timelines,
    evaluate as evaluate_segmentation,
    match_intervals,
    propose,
)


CLASSIFIER_K = (1, 3, 5, 7)
CONFIDENCE_THRESHOLDS = tuple(index / 100 for index in range(48, 61))
PEAK_WINDOWS = tuple(
    (before, after)
    for before in (0.45, 0.60, 0.75, 0.90)
    for after in (0.45, 0.60, 0.75, 0.90)
)
SEGMENTATION_PARAMETERS = tuple(
    (threshold, separation, before, after, boundary_quantile)
    for threshold in (0.75, 0.80, 0.85, 0.90)
    for separation in (0.75, 0.90, 1.05, 1.20)
    for before in (0.90, 1.20)
    for after in (0.90, 1.20)
    for boundary_quantile in (0.35, 0.50, 0.65)
)


def select_segmentation_parameters(
    training: list[VideoTimeline],
    minimum_iou: float,
) -> tuple[float, float, float, float, float]:
    scored: list[tuple[Any, ...]] = []
    for threshold, separation, before, after, boundary_quantile in SEGMENTATION_PARAMETERS:
        report = evaluate_segmentation(
            training,
            threshold,
            separation,
            before,
            after,
            minimum_iou,
            boundary_quantile,
        )
        boundary_preference = -abs(before - 0.75) - abs(after - 0.75)
        scored.append(
            (
                report["f1"],
                report["precision"],
                report["recall"],
                boundary_preference,
                threshold,
                separation,
                before,
                after,
                boundary_quantile,
            )
        )
    _, _, _, _, threshold, separation, before, after, boundary_quantile = max(scored)
    return threshold, separation, before, after, boundary_quantile


def _classifier_score(
    aggregate_events: list[StrokeEvent],
    smoothed_events: list[StrokeEvent],
    training_videos: set[str],
    validation_video: str,
    gate_k: int,
    side_k: int,
) -> tuple[int, int]:
    aggregate_train = [event for event in aggregate_events if event.video in training_videos]
    aggregate_test = [event for event in aggregate_events if event.video == validation_video]
    smoothed_train = [event for event in smoothed_events if event.video in training_videos]
    smoothed_test = [event for event in smoothed_events if event.video == validation_video]
    predictions = predict_hierarchical(
        aggregate_train,
        aggregate_test,
        smoothed_train,
        smoothed_test,
        gate_k,
        side_k,
    )
    correct = sum(
        prediction == event.target_label
        for prediction, event in zip(predictions, smoothed_test)
    )
    return correct, len(smoothed_test)


def select_classifier_parameters(
    aggregate_events: list[StrokeEvent],
    smoothed_events: list[StrokeEvent],
    training_videos: list[str],
) -> tuple[int, int, float]:
    scored: list[tuple[float, int, int]] = []
    for gate_k in CLASSIFIER_K:
        for side_k in CLASSIFIER_K:
            correct = 0
            total = 0
            for validation_video in training_videos:
                inner_training = {
                    video for video in training_videos if video != validation_video
                }
                fold_correct, fold_total = _classifier_score(
                    aggregate_events,
                    smoothed_events,
                    inner_training,
                    validation_video,
                    gate_k,
                    side_k,
                )
                correct += fold_correct
                total += fold_total
            scored.append((correct / total if total else 0.0, -gate_k, -side_k))
    score, negative_gate_k, negative_side_k = max(scored)
    return -negative_gate_k, -negative_side_k, score


def select_ensemble_confidence_threshold(
    aggregate_events: list[StrokeEvent],
    primary_events: list[StrokeEvent],
    auxiliary_events: list[StrokeEvent],
    training_videos: list[str],
    primary_gate_k: int,
    primary_side_k: int,
    auxiliary_gate_k: int,
    auxiliary_side_k: int,
) -> tuple[float, list[dict[str, Any]]]:
    outcomes: list[tuple[bool, float, bool]] = []
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
        primary_train = [
            event for event in primary_events if event.video in inner_training
        ]
        primary_test = [
            event for event in primary_events if event.video == validation_video
        ]
        auxiliary_train = [
            event for event in auxiliary_events if event.video in inner_training
        ]
        auxiliary_test = [
            event for event in auxiliary_events if event.video == validation_video
        ]
        primary_diagnostics = predict_hierarchical_diagnostics(
            aggregate_train,
            aggregate_test,
            primary_train,
            primary_test,
            primary_gate_k,
            primary_side_k,
        )
        auxiliary_diagnostics = predict_hierarchical_diagnostics(
            aggregate_train,
            aggregate_test,
            auxiliary_train,
            auxiliary_test,
            auxiliary_gate_k,
            auxiliary_side_k,
        )
        outcomes.extend(
            (
                primary["prediction"] == auxiliary["prediction"],
                min(primary["confidence"], auxiliary["confidence"]),
                primary["prediction"] == event.target_label,
            )
            for primary, auxiliary, event in zip(
                primary_diagnostics,
                auxiliary_diagnostics,
                primary_test,
            )
        )

    curve: list[dict[str, Any]] = []
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = [
            correct
            for agrees, confidence, correct in outcomes
            if agrees and confidence >= threshold
        ]
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / len(outcomes) if outcomes else 0.0,
                "accepted": len(accepted),
                "correct": sum(accepted),
                "accuracy": sum(accepted) / len(accepted) if accepted else 0.0,
            }
        )
    eligible = [item for item in curve if item["coverage"] >= 0.50]
    target_accuracy = [item for item in eligible if item["accuracy"] >= 0.90]
    if target_accuracy:
        selected = max(
            target_accuracy,
            key=lambda item: (item["coverage"], item["accuracy"], -item["threshold"]),
        )
    elif eligible:
        selected = max(
            eligible,
            key=lambda item: (item["accuracy"], item["coverage"], item["threshold"]),
        )
    else:
        selected = curve[0]
    return float(selected["threshold"]), curve


def _failure_stage(actual_class: str, diagnostic: dict[str, Any]) -> str | None:
    if diagnostic["prediction"] == actual_class:
        return None
    if actual_class in GROUNDSTROKES:
        if diagnostic["gate"]["prediction"] != "groundstroke":
            return "groundstroke_gate"
        return "forehand_backhand"
    return "groundstroke_gate"


def _feature_bounds(
    proposal: Any,
    known_regions: tuple[Any, ...],
    peak_before: float | None,
    peak_after: float | None,
) -> tuple[float, float]:
    if peak_before is None or peak_after is None or proposal.peak is None:
        return proposal.start, proposal.end
    start = proposal.peak - peak_before
    end = proposal.peak + peak_after
    for region in known_regions:
        if region.start <= proposal.peak < region.end:
            return max(start, region.start), min(end, region.end)
    return start, end


def proposal_events(
    video: str,
    records: list[dict[str, Any]],
    proposals: list[Any],
    labels_by_interval: dict[tuple[float, float], str] | None = None,
    known_regions: tuple[Any, ...] = (),
    peak_before: float | None = None,
    peak_after: float | None = None,
    temporal_features: str = "smoothed",
) -> tuple[list[StrokeEvent], list[StrokeEvent]]:
    aggregate: list[StrokeEvent] = []
    smoothed: list[StrokeEvent] = []
    for proposal in proposals:
        feature_start, feature_end = _feature_bounds(
            proposal,
            known_regions,
            peak_before,
            peak_after,
        )
        aggregate_selected = [
            record
            for record in records
            if proposal.start <= float(record["time_seconds"]) < proposal.end
        ]
        smoothed_selected = [
            record
            for record in records
            if feature_start <= float(record["time_seconds"]) < feature_end
        ]
        if len(aggregate_selected) < 3 or len(smoothed_selected) < 3:
            continue
        aggregate_segment = AnnotationSegment(proposal.start, proposal.end, "unknown")
        smoothed_segment = AnnotationSegment(feature_start, feature_end, "unknown")
        source_label = (
            labels_by_interval.get((proposal.start, proposal.end), "recovery")
            if labels_by_interval is not None
            else "unknown"
        )
        common = {
            "video": video,
            "start": proposal.start,
            "end": proposal.end,
            "source_label": source_label,
            "target_label": target_label(source_label),
            "sample_count": len(smoothed_selected),
        }
        aggregate.append(
            StrokeEvent(
                vector=aggregate_event_vector(aggregate_selected, aggregate_segment),
                **common,
            )
        )
        smoothed.append(
            StrokeEvent(
                vector=(
                    motion_phase_event_vector(smoothed_selected, smoothed_segment)
                    if temporal_features == "motion-phases"
                    else smoothed_event_vector(smoothed_selected, smoothed_segment)
                ),
                **common,
            )
        )
    return aggregate, smoothed


def build_training_events(
    training_timelines: list[VideoTimeline],
    records_by_video: dict[str, list[dict[str, Any]]],
    proposals_by_video: dict[str, list[Any]],
    labels_by_video: dict[str, dict[tuple[float, float], str]],
    peak_before: float | None,
    peak_after: float | None,
    temporal_features: str,
) -> tuple[list[StrokeEvent], list[StrokeEvent]]:
    aggregate: list[StrokeEvent] = []
    smoothed: list[StrokeEvent] = []
    for timeline in training_timelines:
        proposal_aggregate, proposal_smoothed = proposal_events(
            timeline.video,
            records_by_video[timeline.video],
            proposals_by_video[timeline.video],
            labels_by_video[timeline.video],
            timeline.known_regions,
            peak_before,
            peak_after,
            temporal_features,
        )
        aggregate.extend(proposal_aggregate)
        smoothed.extend(proposal_smoothed)
    return aggregate, smoothed


def select_peak_window(
    training_timelines: list[VideoTimeline],
    records_by_video: dict[str, list[dict[str, Any]]],
    proposals_by_video: dict[str, list[Any]],
    labels_by_video: dict[str, dict[tuple[float, float], str]],
    temporal_features: str,
) -> tuple[float, float, int, int, list[StrokeEvent], list[StrokeEvent], float]:
    training_videos = [timeline.video for timeline in training_timelines]
    candidates: list[tuple[Any, ...]] = []
    for peak_before, peak_after in PEAK_WINDOWS:
        aggregate, smoothed = build_training_events(
            training_timelines,
            records_by_video,
            proposals_by_video,
            labels_by_video,
            peak_before,
            peak_after,
            temporal_features,
        )
        gate_k, side_k, score = select_classifier_parameters(
            aggregate,
            smoothed,
            training_videos,
        )
        symmetry = -abs(peak_before - peak_after)
        center_preference = -abs(peak_before - 0.70) - abs(peak_after - 0.70)
        candidates.append(
            (
                score,
                symmetry,
                center_preference,
                -gate_k,
                -side_k,
                peak_before,
                peak_after,
                aggregate,
                smoothed,
            )
        )
    (
        score,
        _,
        _,
        negative_gate_k,
        negative_side_k,
        peak_before,
        peak_after,
        aggregate,
        smoothed,
    ) = max(candidates, key=lambda item: item[:7])
    return (
        peak_before,
        peak_after,
        -negative_gate_k,
        -negative_side_k,
        aggregate,
        smoothed,
        score,
    )


def evaluate_pipeline(
    pose_dir: Path,
    annotation_dir: Path,
    minimum_iou: float,
    classifier_alignment: str = "interval",
    classifier_features: str = "motion-phases",
) -> dict[str, Any]:
    timelines = discover_timelines(pose_dir, annotation_dir)
    records_by_video: dict[str, list[dict[str, Any]]] = {}
    for annotation_path in annotation_dir.glob("*.json"):
        metadata, _ = load_annotations(annotation_path)
        video = str(metadata.get("video", annotation_path.stem))
        pose_log = pose_dir / f"{annotation_path.stem}-pose.jsonl"
        if pose_log.exists():
            records_by_video[video] = [
                json.loads(line) for line in pose_log.open() if line.strip()
            ]
    confusion = {actual: Counter() for actual in TARGET_LABELS}
    folds: list[dict[str, Any]] = []
    total_truth = 0
    total_proposals = 0
    total_matches = 0
    total_correct = 0
    total_false_proposals = 0
    total_accepted_matches = 0
    total_accepted_correct = 0
    total_accepted_false_proposals = 0
    total_known_seconds = 0.0

    for timeline in timelines:
        training_timelines = [other for other in timelines if other.video != timeline.video]
        threshold, separation, before, after, boundary_quantile = select_segmentation_parameters(
            training_timelines, minimum_iou
        )
        training_proposals_by_video: dict[str, list[Any]] = {}
        training_labels_by_video: dict[str, dict[tuple[float, float], str]] = {}
        for training_timeline in training_timelines:
            training_proposals = propose(
                training_timeline,
                threshold,
                separation,
                before,
                after,
                boundary_quantile,
            )
            training_matches, _, _ = match_intervals(
                training_proposals,
                training_timeline.truth,
                minimum_iou,
            )
            training_proposals_by_video[training_timeline.video] = training_proposals
            training_labels_by_video[training_timeline.video] = {
                (proposal.start, proposal.end): actual.label or "unknown"
                for proposal, actual, _ in training_matches
            }
        if classifier_alignment == "peak":
            (
                peak_before,
                peak_after,
                gate_k,
                side_k,
                training_aggregate,
                training_smoothed,
                classifier_selection_score,
            ) = select_peak_window(
                training_timelines,
                records_by_video,
                training_proposals_by_video,
                training_labels_by_video,
                classifier_features,
            )
        else:
            peak_before = None
            peak_after = None
            training_aggregate, training_smoothed = build_training_events(
                training_timelines,
                records_by_video,
                training_proposals_by_video,
                training_labels_by_video,
                peak_before,
                peak_after,
                classifier_features,
            )
            gate_k, side_k, classifier_selection_score = select_classifier_parameters(
                training_aggregate,
                training_smoothed,
                [training_timeline.video for training_timeline in training_timelines],
            )
        auxiliary_features = (
            "smoothed" if classifier_features == "motion-phases" else "motion-phases"
        )
        auxiliary_aggregate, training_auxiliary = build_training_events(
            training_timelines,
            records_by_video,
            training_proposals_by_video,
            training_labels_by_video,
            peak_before,
            peak_after,
            auxiliary_features,
        )
        auxiliary_gate_k, auxiliary_side_k, _ = select_classifier_parameters(
            auxiliary_aggregate,
            training_auxiliary,
            [training_timeline.video for training_timeline in training_timelines],
        )
        confidence_threshold, training_confidence_curve = select_ensemble_confidence_threshold(
            training_aggregate,
            training_smoothed,
            training_auxiliary,
            [training_timeline.video for training_timeline in training_timelines],
            gate_k,
            side_k,
            auxiliary_gate_k,
            auxiliary_side_k,
        )
        proposals = propose(
            timeline,
            threshold,
            separation,
            before,
            after,
            boundary_quantile,
        )
        proposal_aggregate, proposal_smoothed = proposal_events(
            timeline.video,
            records_by_video[timeline.video],
            proposals,
            known_regions=timeline.known_regions,
            peak_before=peak_before,
            peak_after=peak_after,
            temporal_features=classifier_features,
        )
        _, proposal_auxiliary = proposal_events(
            timeline.video,
            records_by_video[timeline.video],
            proposals,
            known_regions=timeline.known_regions,
            peak_before=peak_before,
            peak_after=peak_after,
            temporal_features=auxiliary_features,
        )
        prediction_diagnostics = predict_hierarchical_diagnostics(
            training_aggregate,
            proposal_aggregate,
            training_smoothed,
            proposal_smoothed,
            gate_k,
            side_k,
        )
        auxiliary_diagnostics = predict_hierarchical_diagnostics(
            auxiliary_aggregate,
            proposal_aggregate,
            training_auxiliary,
            proposal_auxiliary,
            auxiliary_gate_k,
            auxiliary_side_k,
        )
        diagnostic_by_interval = {
            (event.start, event.end): diagnostic
            for event, diagnostic in zip(proposal_smoothed, prediction_diagnostics)
        }
        auxiliary_by_interval = {
            (event.start, event.end): diagnostic
            for event, diagnostic in zip(proposal_auxiliary, auxiliary_diagnostics)
        }
        matches, false_positives, misses = match_intervals(
            proposals, timeline.truth, minimum_iou
        )
        correct = 0
        accepted_matches = 0
        accepted_correct = 0
        audit: list[dict[str, Any]] = []
        for proposal, actual, iou in matches:
            diagnostic = diagnostic_by_interval[(proposal.start, proposal.end)]
            auxiliary = auxiliary_by_interval[(proposal.start, proposal.end)]
            prediction = diagnostic["prediction"]
            actual_class = target_label(actual.label or "unknown")
            confusion[actual_class][prediction] += 1
            if prediction == actual_class:
                correct += 1
            ensemble_confidence = min(
                diagnostic["confidence"],
                auxiliary["confidence"],
            )
            classifiers_agree = prediction == auxiliary["prediction"]
            accepted = classifiers_agree and ensemble_confidence >= confidence_threshold
            if accepted:
                accepted_matches += 1
                if prediction == actual_class:
                    accepted_correct += 1
            feature_start, feature_end = _feature_bounds(
                proposal,
                timeline.known_regions,
                peak_before,
                peak_after,
            )
            event = next(
                event
                for event in proposal_smoothed
                if event.start == proposal.start and event.end == proposal.end
            )
            phase_anchors = (
                [
                    feature_start + float(position) * (feature_end - feature_start)
                    for position in event.vector[7:12]
                ]
                if classifier_features == "motion-phases"
                else []
            )
            audit.append(
                {
                    "proposal_start": proposal.start,
                    "proposal_end": proposal.end,
                    "motion_peak": proposal.peak,
                    "actual_start": actual.start,
                    "actual_end": actual.end,
                    "actual_source_label": actual.label,
                    "actual_class": actual_class,
                    "predicted_class": prediction,
                    "iou": iou,
                    "confidence": ensemble_confidence,
                    "primary_confidence": diagnostic["confidence"],
                    "auxiliary_prediction": auxiliary["prediction"],
                    "auxiliary_confidence": auxiliary["confidence"],
                    "classifiers_agree": classifiers_agree,
                    "accepted": accepted,
                    "failure_stage": _failure_stage(actual_class, diagnostic),
                    "pose_confidence": float(event.vector[6]),
                    "phase_anchor_times": phase_anchors,
                    "gate": diagnostic["gate"],
                    "side": diagnostic["side"],
                }
            )
        accepted_false_proposals = 0
        for proposal in false_positives:
            diagnostic = diagnostic_by_interval[(proposal.start, proposal.end)]
            auxiliary = auxiliary_by_interval[(proposal.start, proposal.end)]
            ensemble_confidence = min(
                diagnostic["confidence"],
                auxiliary["confidence"],
            )
            classifiers_agree = diagnostic["prediction"] == auxiliary["prediction"]
            accepted = classifiers_agree and ensemble_confidence >= confidence_threshold
            accepted_false_proposals += int(accepted)
            audit.append(
                {
                    "proposal_start": proposal.start,
                    "proposal_end": proposal.end,
                    "motion_peak": proposal.peak,
                    "actual_class": None,
                    "predicted_class": diagnostic["prediction"],
                    "iou": 0.0,
                    "confidence": ensemble_confidence,
                    "primary_confidence": diagnostic["confidence"],
                    "auxiliary_prediction": auxiliary["prediction"],
                    "auxiliary_confidence": auxiliary["confidence"],
                    "classifiers_agree": classifiers_agree,
                    "accepted": accepted,
                    "failure_stage": "unmatched_proposal",
                    "gate": diagnostic["gate"],
                    "side": diagnostic["side"],
                }
            )
        for actual in misses:
            audit.append(
                {
                    "actual_start": actual.start,
                    "actual_end": actual.end,
                    "actual_source_label": actual.label,
                    "actual_class": target_label(actual.label or "unknown"),
                    "predicted_class": None,
                    "accepted": False,
                    "failure_stage": "missed_detection",
                }
            )
        known_seconds = sum(region.end - region.start for region in timeline.known_regions)
        total_known_seconds += known_seconds
        total_truth += len(timeline.truth)
        total_proposals += len(proposals)
        total_matches += len(matches)
        total_correct += correct
        total_false_proposals += len(false_positives)
        total_accepted_matches += accepted_matches
        total_accepted_correct += accepted_correct
        total_accepted_false_proposals += accepted_false_proposals
        folds.append(
            {
                "video": timeline.video,
                "truth": len(timeline.truth),
                "proposals": len(proposals),
                "matches": len(matches),
                "correctly_classified_matches": correct,
                "false_proposals": len(false_positives),
                "misses": len(misses),
                "matched_classification_accuracy": correct / len(matches) if matches else 0.0,
                "confidence_threshold": confidence_threshold,
                "accepted_matches": accepted_matches,
                "accepted_correct": accepted_correct,
                "accepted_accuracy": (
                    accepted_correct / accepted_matches if accepted_matches else 0.0
                ),
                "accepted_coverage": accepted_matches / len(matches) if matches else 0.0,
                "accepted_false_proposals": accepted_false_proposals,
                "end_to_end_recall": correct / len(timeline.truth) if timeline.truth else 0.0,
                "training_confidence_curve": training_confidence_curve,
                "audit": sorted(
                    audit,
                    key=lambda item: item.get(
                        "proposal_start",
                        item.get("actual_start", 0.0),
                    ),
                ),
                "parameters": {
                    "threshold_quantile": threshold,
                    "minimum_separation": separation,
                    "before_peak": before,
                    "after_peak": after,
                    "boundary_quantile": boundary_quantile,
                    "classifier_peak_before": peak_before,
                    "classifier_peak_after": peak_after,
                    "classifier_selection_accuracy": classifier_selection_score,
                    "classifier_features": classifier_features,
                    "confidence_method": "temporal-representation agreement plus distance margin",
                    "auxiliary_features": auxiliary_features,
                    "auxiliary_gate_k": auxiliary_gate_k,
                    "auxiliary_side_k": auxiliary_side_k,
                    "gate_k": gate_k,
                    "side_k": side_k,
                },
            }
        )

    precision = total_correct / total_proposals if total_proposals else 0.0
    recall = total_correct / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": "outer leave-one-video-out with training-video parameter selection",
        "classifier_alignment": classifier_alignment,
        "classifier_features": classifier_features,
        "minimum_iou": minimum_iou,
        "truth": total_truth,
        "proposals": total_proposals,
        "detected": total_matches,
        "correctly_detected_and_classified": total_correct,
        "unmatched_proposals": total_false_proposals,
        "detection_recall": total_matches / total_truth if total_truth else 0.0,
        "matched_classification_accuracy": total_correct / total_matches if total_matches else 0.0,
        "selective_classification": {
            "accepted": total_accepted_matches,
            "correct": total_accepted_correct,
            "abstained": total_matches - total_accepted_matches,
            "coverage": total_accepted_matches / total_matches if total_matches else 0.0,
            "accuracy": (
                total_accepted_correct / total_accepted_matches
                if total_accepted_matches
                else 0.0
            ),
            "accepted_false_proposals": total_accepted_false_proposals,
            "false_alerts_per_minute": (
                total_accepted_false_proposals / (total_known_seconds / 60.0)
                if total_known_seconds
                else 0.0
            ),
        },
        "end_to_end_precision": precision,
        "end_to_end_recall": recall,
        "end_to_end_f1": f1,
        "false_alerts_per_minute": (
            total_false_proposals / (total_known_seconds / 60.0)
            if total_known_seconds
            else 0.0
        ),
        "confusion_on_matched_proposals": {
            label: dict(confusion[label]) for label in TARGET_LABELS
        },
        "videos": folds,
    }


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Classifier: alignment={report['classifier_alignment']}, "
        f"features={report['classifier_features']}"
    )
    print(
        f"End-to-end: correct={report['correctly_detected_and_classified']}/"
        f"{report['truth']}, proposals={report['proposals']}, "
        f"precision={report['end_to_end_precision']:.1%}, "
        f"recall={report['end_to_end_recall']:.1%}, "
        f"F1={report['end_to_end_f1']:.1%}"
    )
    print(
        f"Stage diagnostics: detection recall={report['detection_recall']:.1%}, "
        f"classification on matched proposals={report['matched_classification_accuracy']:.1%}, "
        f"false alerts={report['false_alerts_per_minute']:.2f}/minute"
    )
    selective = report["selective_classification"]
    print(
        f"Confidence filter: accuracy={selective['accuracy']:.1%}, "
        f"coverage={selective['coverage']:.1%} "
        f"({selective['accepted']}/{report['detected']} accepted), "
        f"confident false alerts={selective['false_alerts_per_minute']:.2f}/minute"
    )
    print("Per-video results:")
    for video in report["videos"]:
        print(
            f"  {video['video']}: correct={video['correctly_classified_matches']}/"
            f"{video['truth']}, detected={video['matches']}, proposals={video['proposals']}, "
            f"matched classification={video['matched_classification_accuracy']:.1%}, "
            f"confident={video['accepted_accuracy']:.1%} at "
            f"{video['accepted_coverage']:.1%} coverage "
            f"(threshold={video['confidence_threshold']:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end stroke detection and recognition")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--minimum-iou", type=float, default=0.30)
    parser.add_argument(
        "--classifier-alignment",
        choices=("interval", "peak"),
        default="interval",
        help="Classify adaptive intervals or training-selected peak windows (default: interval)",
    )
    parser.add_argument(
        "--classifier-features",
        choices=("smoothed", "motion-phases"),
        default="motion-phases",
        help="Temporal representation for forehand/backhand recognition (default: motion-phases)",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate_pipeline(
        args.pose_dir,
        args.annotations,
        args.minimum_iou,
        args.classifier_alignment,
        args.classifier_features,
    )
    print_report(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
