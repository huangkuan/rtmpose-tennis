from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import AnnotationSegment, load_annotations
from .stroke_analysis import DOMINANT_WRIST, GROUNDSTROKES

STROKE_LABELS = GROUNDSTROKES | {
    "serve",
    "overhead",
    "forehand_volley",
    "backhand_volley",
}

@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    score: float = 0.0
    label: str | None = None
    peak: float | None = None


@dataclass(frozen=True)
class VideoTimeline:
    video: str
    timestamps: np.ndarray
    motion: np.ndarray
    known_regions: tuple[Interval, ...]
    truth: tuple[Interval, ...]


def _odd_window(timestamps: np.ndarray, seconds: float) -> int:
    if timestamps.size < 3:
        return 1
    median_step = float(np.median(np.diff(timestamps)))
    size = max(1, int(round(seconds / max(median_step, 1e-3))))
    if size % 2 == 0:
        size += 1
    return min(size, timestamps.size if timestamps.size % 2 else timestamps.size - 1)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + window]) for index in range(values.size)],
        dtype=np.float64,
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.full(window, 1.0 / window)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _known_regions(segments: tuple[AnnotationSegment, ...]) -> tuple[Interval, ...]:
    known = [segment for segment in segments if segment.label != "unknown"]
    if not known:
        return ()
    regions: list[Interval] = []
    start, end = known[0].start, known[0].end
    for segment in known[1:]:
        if segment.start - end <= 0.05:
            end = max(end, segment.end)
        else:
            regions.append(Interval(start, end))
            start, end = segment.start, segment.end
    regions.append(Interval(start, end))
    return tuple(regions)


def build_timeline(pose_log: Path, annotation_path: Path) -> VideoTimeline:
    metadata, segments = load_annotations(annotation_path)
    records = [json.loads(line) for line in pose_log.open() if line.strip()]
    timestamps = np.asarray([float(record["time_seconds"]) for record in records])
    keypoints = np.asarray([record["normalized_keypoints"] for record in records], dtype=np.float64)
    scores = np.asarray([record["scores"] for record in records], dtype=np.float64)
    wrist = keypoints[:, DOMINANT_WRIST]
    valid = (
        (scores[:, DOMINANT_WRIST] >= 0.35)
        & np.all(np.isfinite(wrist), axis=1)
        & np.all(np.abs(wrist) <= 6.0, axis=1)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"not enough valid dominant-wrist samples in {pose_log}")
    wrist_x = np.interp(timestamps, timestamps[valid], wrist[valid, 0])
    wrist_y = np.interp(timestamps, timestamps[valid], wrist[valid, 1])
    median_window = _odd_window(timestamps, 0.10)
    wrist_x = _rolling_median(wrist_x, median_window)
    wrist_y = _rolling_median(wrist_y, median_window)
    wrist_x = _rolling_mean(wrist_x, _odd_window(timestamps, 0.08))
    wrist_y = _rolling_mean(wrist_y, _odd_window(timestamps, 0.08))
    velocity_x = np.gradient(wrist_x, timestamps)
    velocity_y = np.gradient(wrist_y, timestamps)
    motion = np.hypot(velocity_x, velocity_y)
    motion = np.log1p(motion)
    motion = _rolling_mean(motion, _odd_window(timestamps, 0.16))
    truth = tuple(
        Interval(segment.start, segment.end, label=segment.label)
        for segment in segments
        if segment.label in STROKE_LABELS
    )
    return VideoTimeline(
        video=str(metadata.get("video", annotation_path.stem)),
        timestamps=timestamps,
        motion=motion,
        known_regions=_known_regions(segments),
        truth=truth,
    )


def discover_timelines(pose_dir: Path, annotation_dir: Path) -> list[VideoTimeline]:
    timelines: list[VideoTimeline] = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        pose_log = pose_dir / f"{annotation_path.stem}-pose.jsonl"
        if pose_log.exists():
            timelines.append(build_timeline(pose_log, annotation_path))
    return timelines


def propose(
    timeline: VideoTimeline,
    threshold_quantile: float = 0.80,
    minimum_separation: float = 0.75,
    before_peak: float = 0.75,
    after_peak: float = 0.75,
    boundary_quantile: float | None = None,
    boundary_padding: float = 0.12,
) -> list[Interval]:
    proposals: list[Interval] = []
    for region in timeline.known_regions:
        mask = (timeline.timestamps >= region.start) & (timeline.timestamps < region.end)
        indices = np.flatnonzero(mask)
        if indices.size < 3:
            continue
        regional_motion = timeline.motion[indices]
        threshold = float(np.quantile(regional_motion, threshold_quantile))
        local = indices[
            (regional_motion >= np.roll(regional_motion, 1))
            & (regional_motion > np.roll(regional_motion, -1))
            & (regional_motion >= threshold)
        ]
        local = local[(local != indices[0]) & (local != indices[-1])]
        selected: list[int] = []
        for index in sorted(local, key=lambda item: timeline.motion[item], reverse=True):
            timestamp = float(timeline.timestamps[index])
            if all(
                abs(timestamp - float(timeline.timestamps[kept])) >= minimum_separation
                for kept in selected
            ):
                selected.append(int(index))
        for index in sorted(selected):
            peak = float(timeline.timestamps[index])
            start = max(region.start, peak - before_peak)
            end = min(region.end, peak + after_peak)
            if boundary_quantile is not None:
                boundary_level = float(np.quantile(regional_motion, boundary_quantile))
                left_limit = int(np.searchsorted(timeline.timestamps, start, side="left"))
                right_limit = int(np.searchsorted(timeline.timestamps, end, side="right")) - 1
                onset_index = left_limit
                for candidate in range(index - 1, left_limit - 1, -1):
                    if timeline.motion[candidate] <= boundary_level:
                        onset_index = candidate
                        break
                deceleration_index = right_limit
                for candidate in range(index + 1, right_limit + 1):
                    if timeline.motion[candidate] <= boundary_level:
                        deceleration_index = candidate
                        break
                start = max(
                    region.start,
                    float(timeline.timestamps[onset_index]) - boundary_padding,
                )
                end = min(
                    region.end,
                    float(timeline.timestamps[deceleration_index]) + boundary_padding,
                )
            proposals.append(
                Interval(
                    start=start,
                    end=end,
                    score=float(timeline.motion[index]),
                    peak=peak,
                )
            )
    return proposals


def interval_iou(first: Interval, second: Interval) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return intersection / union if union > 0 else 0.0


def match_intervals(
    proposals: list[Interval],
    truth: tuple[Interval, ...],
    minimum_iou: float = 0.30,
) -> tuple[list[tuple[Interval, Interval, float]], list[Interval], list[Interval]]:
    candidates = sorted(
        (
            (interval_iou(proposal, actual), proposal_index, truth_index)
            for proposal_index, proposal in enumerate(proposals)
            for truth_index, actual in enumerate(truth)
        ),
        reverse=True,
    )
    used_proposals: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[Interval, Interval, float]] = []
    for iou, proposal_index, truth_index in candidates:
        if iou < minimum_iou:
            break
        if proposal_index in used_proposals or truth_index in used_truth:
            continue
        used_proposals.add(proposal_index)
        used_truth.add(truth_index)
        matches.append((proposals[proposal_index], truth[truth_index], iou))
    false_positives = [
        proposal for index, proposal in enumerate(proposals) if index not in used_proposals
    ]
    misses = [actual for index, actual in enumerate(truth) if index not in used_truth]
    return matches, false_positives, misses


def evaluate(
    timelines: list[VideoTimeline],
    threshold_quantile: float,
    minimum_separation: float,
    before_peak: float,
    after_peak: float,
    minimum_iou: float,
    boundary_quantile: float | None = None,
    boundary_padding: float = 0.12,
) -> dict[str, Any]:
    per_video: list[dict[str, Any]] = []
    all_matches: list[tuple[Interval, Interval, float]] = []
    false_positive_count = 0
    miss_count = 0
    for timeline in timelines:
        proposals = propose(
            timeline,
            threshold_quantile,
            minimum_separation,
            before_peak,
            after_peak,
            boundary_quantile,
            boundary_padding,
        )
        matches, false_positives, misses = match_intervals(
            proposals, timeline.truth, minimum_iou
        )
        all_matches.extend(matches)
        false_positive_count += len(false_positives)
        miss_count += len(misses)
        precision = len(matches) / len(proposals) if proposals else 0.0
        recall = len(matches) / len(timeline.truth) if timeline.truth else 0.0
        per_video.append(
            {
                "video": timeline.video,
                "truth": len(timeline.truth),
                "proposals": len(proposals),
                "matches": len(matches),
                "precision": precision,
                "recall": recall,
                "false_positives": len(false_positives),
                "misses": len(misses),
            }
        )
    true_positive_count = len(all_matches)
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + miss_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "parameters": {
            "threshold_quantile": threshold_quantile,
            "minimum_separation": minimum_separation,
            "before_peak": before_peak,
            "after_peak": after_peak,
            "minimum_iou": minimum_iou,
            "boundary_quantile": boundary_quantile,
            "boundary_padding": boundary_padding,
        },
        "truth": true_positive_count + miss_count,
        "proposals": true_positive_count + false_positive_count,
        "matches": true_positive_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean([match[2] for match in all_matches])) if all_matches else 0.0,
        "start_error_ms": float(
            np.mean([abs(match[0].start - match[1].start) for match in all_matches]) * 1000
        ) if all_matches else 0.0,
        "end_error_ms": float(
            np.mean([abs(match[0].end - match[1].end) for match in all_matches]) * 1000
        ) if all_matches else 0.0,
        "videos": per_video,
    }


def evaluate_nested(
    timelines: list[VideoTimeline],
    minimum_iou: float,
    adaptive: bool = True,
) -> dict[str, Any]:
    candidate_parameters = tuple(
        (threshold, separation, before, after, boundary_quantile)
        for threshold in (0.75, 0.80, 0.85, 0.90)
        for separation in (0.75, 0.90, 1.05, 1.20)
        for before in ((0.90, 1.20) if adaptive else (0.60, 0.75, 0.90))
        for after in ((0.90, 1.20) if adaptive else (0.60, 0.75, 0.90))
        for boundary_quantile in ((0.35, 0.50, 0.65) if adaptive else (None,))
    )
    per_video: list[dict[str, Any]] = []
    total_truth = 0
    total_proposals = 0
    total_matches = 0
    weighted_iou = 0.0
    weighted_start_error = 0.0
    weighted_end_error = 0.0
    for held_out in timelines:
        training = [timeline for timeline in timelines if timeline.video != held_out.video]
        scored: list[tuple[Any, ...]] = []
        for threshold, separation, before, after, boundary_quantile in candidate_parameters:
            result = evaluate(
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
                    result["f1"],
                    result["precision"],
                    result["recall"],
                    boundary_preference,
                    threshold,
                    separation,
                    before,
                    after,
                    boundary_quantile if boundary_quantile is not None else -1.0,
                )
            )
        _, _, _, _, threshold, separation, before, after, selected_quantile = max(scored)
        boundary_quantile = selected_quantile if selected_quantile >= 0 else None
        result = evaluate(
            [held_out],
            threshold,
            separation,
            before,
            after,
            minimum_iou,
            boundary_quantile,
        )
        video = result["videos"][0]
        video["parameters"] = {
            "threshold_quantile": threshold,
            "minimum_separation": separation,
            "before_peak": before,
            "after_peak": after,
            "boundary_quantile": boundary_quantile,
        }
        per_video.append(video)
        total_truth += result["truth"]
        total_proposals += result["proposals"]
        total_matches += result["matches"]
        weighted_iou += result["mean_iou"] * result["matches"]
        weighted_start_error += result["start_error_ms"] * result["matches"]
        weighted_end_error += result["end_error_ms"] * result["matches"]

    precision = total_matches / total_proposals if total_proposals else 0.0
    recall = total_matches / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "selection": "outer leave-one-video-out with training-video parameter selection",
        "boundary_mode": "adaptive" if adaptive else "fixed",
        "minimum_iou": minimum_iou,
        "truth": total_truth,
        "proposals": total_proposals,
        "matches": total_matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": weighted_iou / total_matches if total_matches else 0.0,
        "start_error_ms": weighted_start_error / total_matches if total_matches else 0.0,
        "end_error_ms": weighted_end_error / total_matches if total_matches else 0.0,
        "videos": per_video,
    }


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Boundary detection: matches={report['matches']}/{report['truth']}, "
        f"proposals={report['proposals']}, precision={report['precision']:.1%}, "
        f"recall={report['recall']:.1%}, F1={report['f1']:.1%}"
    )
    print(
        f"Matched boundaries: mean IoU={report['mean_iou']:.3f}, "
        f"start error={report['start_error_ms']:.0f} ms, "
        f"end error={report['end_error_ms']:.0f} ms"
    )
    print("Per-video results:")
    for video in report["videos"]:
        parameters = video.get("parameters")
        parameter_text = (
            f", q={parameters['threshold_quantile']:.2f}, "
            f"separation={parameters['minimum_separation']:.2f}s, "
            f"window=-{parameters['before_peak']:.2f}/+{parameters['after_peak']:.2f}s"
            + (
                f", boundary q={parameters['boundary_quantile']:.2f}"
                if parameters.get("boundary_quantile") is not None
                else ""
            )
            if parameters is not None
            else ""
        )
        print(
            f"  {video['video']}: matches={video['matches']}/{video['truth']}, "
            f"proposals={video['proposals']}, precision={video['precision']:.1%}, "
            f"recall={video['recall']:.1%}{parameter_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate automatic stroke-boundary proposals")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--threshold-quantile", type=float, default=0.80)
    parser.add_argument("--minimum-separation", type=float, default=0.75)
    parser.add_argument("--before-peak", type=float, default=0.75)
    parser.add_argument("--after-peak", type=float, default=0.75)
    parser.add_argument("--minimum-iou", type=float, default=0.30)
    parser.add_argument(
        "--boundary-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="Use motion onset/deceleration or fixed peak windows (default: adaptive)",
    )
    parser.add_argument("--boundary-quantile", type=float, default=0.50)
    parser.add_argument("--boundary-padding", type=float, default=0.12)
    parser.add_argument(
        "--selection",
        choices=("nested", "fixed"),
        default="nested",
        help="Select parameters on training videos or use fixed CLI values (default: nested)",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    timelines = discover_timelines(args.pose_dir, args.annotations)
    if args.selection == "nested":
        report = evaluate_nested(
            timelines,
            args.minimum_iou,
            adaptive=args.boundary_mode == "adaptive",
        )
    else:
        report = evaluate(
            timelines,
            args.threshold_quantile,
            args.minimum_separation,
            args.before_peak,
            args.after_peak,
            args.minimum_iou,
            args.boundary_quantile if args.boundary_mode == "adaptive" else None,
            args.boundary_padding,
        )
    print_report(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
