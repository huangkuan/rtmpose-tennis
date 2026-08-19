from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from mmdet.apis import DetInferencer
from mmpose.apis import MMPoseInferencer

POSE_MODEL_PRESETS = {
    "tiny": "rtmpose-t_8xb256-420e_coco-256x192",
    "small": "rtmpose-s_8xb256-420e_coco-256x192",
    "medium": "human",
}
DEFAULT_POSE_MODEL = "small"

COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)


@dataclass
class PlayerPose:
    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray | None


def _instances(result: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = result.get("predictions", [])
    return predictions[0] if predictions else []


def _bbox(instance: dict[str, Any]) -> np.ndarray | None:
    raw = instance.get("bbox")
    if raw is None:
        raw = instance.get("bboxes")
    if raw is None:
        return None
    box = np.asarray(raw, dtype=float).reshape(-1, 4)[0]
    return box


def select_player(result: dict[str, Any], frame_shape: tuple[int, ...]) -> PlayerPose | None:
    """Choose the most prominent person, biased toward the frame centre.

    This keeps bystanders from displacing the athlete in typical court framing.
    A tracker can replace this function later without changing downstream code.
    """
    height, width = frame_shape[:2]
    centre = np.array([width / 2, height / 2])
    best: tuple[float, dict[str, Any], np.ndarray | None] | None = None
    for instance in _instances(result):
        box = _bbox(instance)
        points = np.asarray(instance.get("keypoints", []), dtype=float)
        scores = np.asarray(instance.get("keypoint_scores", []), dtype=float)
        if points.size == 0:
            continue
        if box is not None:
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            box_centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        else:
            visible = points[scores > 0.2] if scores.size else points
            if not len(visible):
                continue
            lo, hi = visible.min(axis=0), visible.max(axis=0)
            area = float(np.prod(hi - lo))
            box_centre = (lo + hi) / 2
        distance_penalty = 1.0 + np.linalg.norm(box_centre - centre) / max(width, height)
        rank = area / distance_penalty
        if best is None or rank > best[0]:
            best = (rank, instance, box)
    if best is None:
        return None
    instance, box = best[1], best[2]
    return PlayerPose(
        np.asarray(instance["keypoints"], dtype=np.float32),
        np.asarray(instance["keypoint_scores"], dtype=np.float32),
        box,
    )


def select_player_bbox(
    result: dict[str, Any],
    frame_shape: tuple[int, ...],
    score_threshold: float,
) -> np.ndarray | None:
    """Select the large, central COCO person box from a DetInferencer result."""
    predictions = result.get("predictions", [])
    if not predictions:
        return None
    prediction = predictions[0]
    boxes = np.asarray(prediction.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(prediction.get("scores", []), dtype=np.float32).reshape(-1)
    labels = np.asarray(prediction.get("labels", []), dtype=np.int64).reshape(-1)
    height, width = frame_shape[:2]
    frame_centre = np.asarray([width / 2, height / 2], dtype=np.float32)
    best: tuple[float, np.ndarray] | None = None
    for box, score, label in zip(boxes, scores, labels):
        if label != 0 or score < score_threshold:  # COCO class 0 is person
            continue
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        box_centre = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        distance_penalty = 1.0 + np.linalg.norm(box_centre - frame_centre) / max(width, height)
        rank = area * float(score) / distance_penalty
        if best is None or rank > best[0]:
            best = (rank, box)
    return best[1].copy() if best is not None else None


def draw_player(
    frame: np.ndarray,
    player: PlayerPose | None,
    threshold: float,
    visual_scale: float = 1.0,
) -> np.ndarray:
    canvas = frame.copy()
    if player is None:
        return canvas
    visible = player.scores >= threshold
    for start, end in COCO_SKELETON:
        if start < len(visible) and end < len(visible) and visible[start] and visible[end]:
            p1 = tuple(np.rint(player.keypoints[start]).astype(int))
            p2 = tuple(np.rint(player.keypoints[end]).astype(int))
            cv2.line(canvas, p1, p2, (0, 220, 255), max(1, round(3 * visual_scale)), cv2.LINE_AA)
    for point, is_visible in zip(player.keypoints, visible):
        if is_visible:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), max(2, round(5 * visual_scale)),
                       (40, 40, 255), -1, cv2.LINE_AA)
    if player.bbox is not None:
        x1, y1, x2, y2 = np.rint(player.bbox).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 230, 80),
                      max(1, round(2 * visual_scale)))
    return canvas


def scale_player_pose(player: PlayerPose | None, scale: float) -> PlayerPose | None:
    """Create a visualization-only pose without modifying full-resolution data."""
    if player is None:
        return None
    return PlayerPose(
        keypoints=player.keypoints * scale,
        scores=player.scores.copy(),
        bbox=player.bbox * scale if player.bbox is not None else None,
    )


def crop_around_player(
    player: PlayerPose,
    frame_shape: tuple[int, ...],
    margin: float,
) -> tuple[int, int, int, int] | None:
    """Return a clamped player crop as (x1, y1, x2, y2)."""
    if player.bbox is not None:
        return crop_around_bbox(player.bbox, frame_shape, margin)
    else:
        visible = player.keypoints[player.scores >= 0.2]
        if len(visible) < 4:
            return None
        x1, y1 = visible.min(axis=0)
        x2, y2 = visible.max(axis=0)
    return crop_around_bbox(np.asarray([x1, y1, x2, y2]), frame_shape, margin)


def crop_around_bbox(
    bbox: np.ndarray,
    frame_shape: tuple[int, ...],
    margin: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    x1 -= width * margin
    x2 += width * margin
    y1 -= height * margin
    y2 += height * margin
    frame_height, frame_width = frame_shape[:2]
    bounds = (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(frame_width, int(np.ceil(x2))),
        min(frame_height, int(np.ceil(y2))),
    )
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def move_pose_to_frame(
    player: PlayerPose | None,
    crop_bounds: tuple[int, int, int, int],
) -> PlayerPose | None:
    """Translate crop-local pose coordinates back to the source frame."""
    if player is None:
        return None
    x1, y1, x2, y2 = crop_bounds
    keypoints = player.keypoints.copy()
    keypoints[:, 0] += x1
    keypoints[:, 1] += y1
    return PlayerPose(
        keypoints=keypoints,
        scores=player.scores.copy(),
        bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
    )


def pose_needs_redetection(
    player: PlayerPose | None,
    crop_bounds: tuple[int, int, int, int],
    score_threshold: float,
    min_keypoints: int,
    edge_fraction: float,
) -> bool:
    if player is None:
        return True
    visible = player.keypoints[player.scores >= score_threshold]
    if len(visible) < min_keypoints:
        return True
    x1, y1, x2, y2 = crop_bounds
    edge_x = (x2 - x1) * edge_fraction
    edge_y = (y2 - y1) * edge_fraction
    return bool(
        np.any(visible[:, 0] <= x1 + edge_x)
        or np.any(visible[:, 0] >= x2 - edge_x)
        or np.any(visible[:, 1] <= y1 + edge_y)
        or np.any(visible[:, 1] >= y2 - edge_y)
    )


def update_crop_from_pose(
    player: PlayerPose,
    current: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    score_threshold: float,
    margin: float,
    alpha: float,
) -> tuple[int, int, int, int]:
    """Smoothly follow confident full-frame keypoints without abrupt crop resizing."""
    visible = player.keypoints[player.scores >= score_threshold]
    if len(visible) < 4:
        return current
    lo, hi = visible.min(axis=0), visible.max(axis=0)
    target = crop_around_bbox(np.concatenate([lo, hi]), frame_shape, margin)
    if target is None:
        return current
    cx1, cy1, cx2, cy2 = map(float, current)
    tx1, ty1, tx2, ty2 = map(float, target)
    current_w, current_h = cx2 - cx1, cy2 - cy1
    target_w = np.clip(tx2 - tx1, current_w * 0.85, current_w * 1.20)
    target_h = np.clip(ty2 - ty1, current_h * 0.85, current_h * 1.20)
    current_cx, current_cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    target_cx, target_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    centre_x = (1 - alpha) * current_cx + alpha * target_cx
    centre_y = (1 - alpha) * current_cy + alpha * target_cy
    width = (1 - alpha) * current_w + alpha * target_w
    height = (1 - alpha) * current_h + alpha * target_h
    smoothed = np.asarray([
        centre_x - width / 2,
        centre_y - height / 2,
        centre_x + width / 2,
        centre_y + height / 2,
    ])
    return crop_around_bbox(smoothed, frame_shape, margin=0.0) or current


def disable_flip_test(inferencer: MMPoseInferencer) -> None:
    pose_inferencer = getattr(inferencer, "inferencer", None)
    pose_model = getattr(pose_inferencer, "model", None)
    if pose_model is not None and hasattr(pose_model, "test_cfg"):
        pose_model.test_cfg["flip_test"] = False


def average_ms(timings: deque[float]) -> str:
    return f"{1000.0 * sum(timings) / len(timings):.1f}" if timings else "--"


def frames(
    camera: int | None,
    video: Path | None,
    width: int,
    height: int,
) -> Iterator[tuple[np.ndarray, float]]:
    source: int | str = str(video) if video is not None else (camera if camera is not None else 0)
    capture = cv2.VideoCapture(source)
    if video is None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        kind = "video" if video is not None else "camera"
        raise RuntimeError(f"Could not open {kind}: {source}")
    try:
        while True:
            decode_started = time.perf_counter()
            ok, frame = capture.read()
            decode_seconds = time.perf_counter() - decode_started
            if not ok:
                if video is not None:
                    break
                raise RuntimeError("Camera stopped returning frames")
            yield frame, decode_seconds
    finally:
        capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a tennis player's body with RTMPose")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Path to a video file")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default=None, help="cuda:0, cpu, or mps; default: auto")
    parser.add_argument(
        "--detector-device",
        default=None,
        help="Device for periodic detection in hybrid mode (default: CPU when --device mps)",
    )
    parser.add_argument(
        "--detector-model",
        default="rtmdet_tiny_8xb32-300e_coco",
        help="MMDetection model used for hybrid refreshes (default: RTMDet-tiny)",
    )
    parser.add_argument(
        "--detector-score-threshold",
        type=float,
        default=0.3,
        help="Minimum person detection confidence in hybrid mode (default: 0.3)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_POSE_MODEL,
        help="Pose preset (tiny, small, medium) or MMPose model/config (default: small)",
    )
    parser.add_argument("--det-model", default=None, help="Optional detector alias/config")
    parser.add_argument(
        "--whole-image",
        action="store_true",
        help="Skip person detection and pose the full frame (fastest; use for one-player framing)",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        metavar="N",
        help="Run inference every Nth frame and reuse the last pose between frames",
    )
    parser.add_argument(
        "--detector-interval",
        type=int,
        default=0,
        metavar="N",
        help="Detect the player every N frames and run pose on the tracked crop between (0: disabled)",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.2,
        help="Fractional padding around the detected player in hybrid mode (default: 0.2)",
    )
    parser.add_argument(
        "--tracking-alpha",
        type=float,
        default=0.35,
        help="Pose-guided crop response from 0 (fixed) to 1 (immediate; default: 0.35)",
    )
    parser.add_argument(
        "--tracking-min-keypoints",
        type=int,
        default=8,
        help="Confident keypoints required before requesting redetection (default: 8)",
    )
    parser.add_argument(
        "--redetect-edge",
        type=float,
        default=0.06,
        help="Crop-edge fraction that triggers detector recovery (default: 0.06)",
    )
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Display scale from 0 to 1; inference remains full resolution (default: 1.0)",
    )
    args = parser.parse_args()
    if args.infer_every < 1:
        parser.error("--infer-every must be at least 1")
    if args.whole_image and args.det_model:
        parser.error("--whole-image cannot be combined with --det-model")
    if args.detector_interval < 0:
        parser.error("--detector-interval cannot be negative")
    if args.detector_interval and args.whole_image:
        parser.error("--detector-interval cannot be combined with --whole-image")
    if args.detector_interval and args.det_model:
        parser.error("--detector-interval cannot currently be combined with --det-model")
    if args.detector_device and not args.detector_interval:
        parser.error("--detector-device requires --detector-interval")
    if not 0 <= args.detector_score_threshold <= 1:
        parser.error("--detector-score-threshold must be between 0 and 1")
    if args.crop_margin < 0:
        parser.error("--crop-margin cannot be negative")
    if not 0 <= args.tracking_alpha <= 1:
        parser.error("--tracking-alpha must be between 0 and 1")
    if args.tracking_min_keypoints < 1:
        parser.error("--tracking-min-keypoints must be at least 1")
    if not 0 <= args.redetect_edge < 0.5:
        parser.error("--redetect-edge must be between 0 and 0.5")
    if not 0 < args.preview_scale <= 1:
        parser.error("--preview-scale must be greater than 0 and at most 1")
    return args


def main() -> None:
    args = parse_args()
    pose_model_name = POSE_MODEL_PRESETS.get(args.model, args.model)
    base_kwargs: dict[str, Any] = {"pose2d": pose_model_name, "device": args.device}
    if args.detector_interval:
        detector_device = args.detector_device or ("cpu" if args.device == "mps" else args.device)
        mode = (
            f"hybrid tracking (pose={args.device or 'auto'}, "
            f"detector={detector_device or 'auto'}, "
            f"detect every {args.detector_interval} frames)"
        )
    elif args.whole_image:
        mode = "whole image (detector bypassed)"
    else:
        mode = "person detector + pose"
    print(
        f"Initializing RTMPose model '{args.model}': {mode}; "
        f"infer every {args.infer_every} frame(s)...",
        flush=True,
    )
    if args.detector_interval:
        print(f"Loading detector-only model '{args.detector_model}'...", flush=True)
        detector_inferencer = DetInferencer(
            model=args.detector_model,
            device=detector_device,
        )
        crop_inferencer = MMPoseInferencer(**base_kwargs, det_model="whole_image")
        disable_flip_test(crop_inferencer)
        inferencer = None
    else:
        kwargs = dict(base_kwargs)
        if args.whole_image:
            kwargs["det_model"] = "whole_image"
        elif args.det_model:
            kwargs["det_model"] = args.det_model
        inferencer = MMPoseInferencer(**kwargs)
        disable_flip_test(inferencer)
        detector_inferencer = None
        crop_inferencer = None
    print("Models ready. Opening input...", flush=True)
    inference_timings: deque[float] = deque(maxlen=30)
    decode_timings: deque[float] = deque(maxlen=60)
    detector_timings: deque[float] = deque(maxlen=20)
    pose_timings: deque[float] = deque(maxlen=60)
    display_timings: deque[float] = deque(maxlen=60)
    loop_timings: deque[float] = deque(maxlen=60)
    player: PlayerPose | None = None
    player_crop: tuple[int, int, int, int] | None = None
    last_detection_frame = -args.detector_interval
    redetection_requested = False
    cv2.namedWindow("RTMPose Tennis", cv2.WINDOW_NORMAL)
    try:
        for frame_index, (frame, decode_seconds) in enumerate(
            frames(args.camera, args.video, args.width, args.height)
        ):
            loop_started = time.perf_counter()
            decode_timings.append(decode_seconds)
            if frame_index % args.infer_every == 0:
                inference_started = time.perf_counter()
                if args.detector_interval:
                    needs_detection = (
                        player_crop is None
                        or redetection_requested
                        or frame_index - last_detection_frame >= args.detector_interval
                    )
                    if needs_detection:
                        assert detector_inferencer is not None
                        detector_started = time.perf_counter()
                        detection_result = detector_inferencer(frame, return_vis=False)
                        detector_timings.append(time.perf_counter() - detector_started)
                        player_bbox = select_player_bbox(
                            detection_result,
                            frame.shape,
                            args.detector_score_threshold,
                        )
                        detected_crop = (
                            crop_around_bbox(player_bbox, frame.shape, args.crop_margin)
                            if player_bbox is not None
                            else None
                        )
                        if detected_crop is not None:
                            player_crop = detected_crop
                            redetection_requested = False
                        last_detection_frame = frame_index
                    if player_crop is not None:
                        assert crop_inferencer is not None
                        x1, y1, x2, y2 = player_crop
                        crop = frame[y1:y2, x1:x2]
                        pose_started = time.perf_counter()
                        result = next(crop_inferencer(crop, return_vis=False))
                        pose_timings.append(time.perf_counter() - pose_started)
                        crop_player = select_player(result, crop.shape)
                        player = move_pose_to_frame(crop_player, player_crop)
                        redetection_requested = pose_needs_redetection(
                            player,
                            player_crop,
                            args.score_threshold,
                            args.tracking_min_keypoints,
                            args.redetect_edge,
                        )
                        if player is not None:
                            player_crop = update_crop_from_pose(
                                player,
                                player_crop,
                                frame.shape,
                                args.score_threshold,
                                args.crop_margin,
                                args.tracking_alpha,
                            )
                            player.bbox = np.asarray(player_crop, dtype=np.float32)
                    else:
                        player = None
                        redetection_requested = True
                else:
                    assert inferencer is not None
                    stage_started = time.perf_counter()
                    result = next(inferencer(frame, return_vis=False))
                    stage_seconds = time.perf_counter() - stage_started
                    if args.whole_image:
                        pose_timings.append(stage_seconds)
                    else:
                        detector_timings.append(stage_seconds)
                    player = select_player(result, frame.shape)
                inference_timings.append(time.perf_counter() - inference_started)
            display_started = time.perf_counter()
            if args.preview_scale < 1.0:
                preview_frame = cv2.resize(
                    frame, None, fx=args.preview_scale, fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
                preview_player = scale_player_pose(player, args.preview_scale)
            else:
                preview_frame = frame
                preview_player = player
            canvas = draw_player(
                preview_frame,
                preview_player,
                args.score_threshold,
                visual_scale=args.preview_scale,
            )
            model_fps = len(inference_timings) / max(sum(inference_timings), 1e-6)
            display_fps = len(loop_timings) / max(sum(loop_timings), 1e-6) if loop_timings else 0.0
            status = "player" if player is not None else "no player"
            text_x = max(5, round(20 * args.preview_scale))
            main_y = max(18, round(35 * args.preview_scale))
            input_y = max(main_y + 14, round(65 * args.preview_scale))
            output_y = max(input_y + 14, round(92 * args.preview_scale))
            main_font = max(0.4, 0.8 * args.preview_scale)
            detail_font = max(0.3, 0.55 * args.preview_scale)
            text_thickness = max(1, round(2 * args.preview_scale))
            cv2.putText(
                canvas,
                f"{status} | model {model_fps:.1f} | output {display_fps:.1f} FPS | Q",
                (text_x, main_y), cv2.FONT_HERSHEY_SIMPLEX, main_font,
                (40, 240, 40), text_thickness, cv2.LINE_AA,
            )
            input_metrics = (
                f"decode {average_ms(decode_timings)} ms | "
                f"detect {average_ms(detector_timings)} ms"
            )
            output_metrics = (
                f"crop pose {average_ms(pose_timings)} ms | "
                f"draw+show {average_ms(display_timings)} ms"
            )
            cv2.putText(
                canvas, input_metrics, (text_x, input_y), cv2.FONT_HERSHEY_SIMPLEX,
                detail_font, (40, 240, 40), text_thickness, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, output_metrics, (text_x, output_y), cv2.FONT_HERSHEY_SIMPLEX,
                detail_font, (40, 240, 40), text_thickness, cv2.LINE_AA,
            )
            # `player.keypoints` and `player.scores` are the stable hand-off to the
            # future temporal gesture classifier and feedback engine.
            cv2.imshow("RTMPose Tennis", canvas)
            key = cv2.waitKey(1) & 0xFF
            display_timings.append(time.perf_counter() - display_started)
            if key in (ord("q"), 27):
                break
            loop_timings.append(time.perf_counter() - loop_started)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
