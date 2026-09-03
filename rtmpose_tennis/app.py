from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
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
COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
BILATERAL_KEYPOINT_PAIRS = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
)


@dataclass
class PlayerPose:
    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray | None


@dataclass
class FramePacket:
    frame: np.ndarray
    decode_seconds: float
    captured_at: float
    sequence: int
    dropped: int = 0


@dataclass(frozen=True)
class CaptureStats:
    count: int
    first_captured_at: float | None
    latest_captured_at: float | None


@dataclass(frozen=True)
class DetectorRequest:
    frame: np.ndarray
    frame_shape: tuple[int, ...]
    frame_index: int
    sequence: int
    requested_at: float
    reason: str


@dataclass(frozen=True)
class DetectorResult:
    request: DetectorRequest
    prediction: Any | None
    inference_seconds: float
    completed_at: float
    error: BaseException | None = None


@dataclass(frozen=True)
class EdgeHit:
    joint: str
    edge: str
    confidence: float


@dataclass(frozen=True)
class RedetectionAssessment:
    reason: str | None
    edge_hits: tuple[EdgeHit, ...] = ()
    clamped_edges: tuple[str, ...] = ()
    clamped_only_suppressed: bool = False


@dataclass(frozen=True)
class HandednessSnapshot:
    label: str
    confidence: float
    locked: bool
    left_evidence: float
    right_evidence: float
    motion_observations: int


class HandednessEstimator:
    """Accumulate conservative wrist-motion handedness evidence."""

    def __init__(self, mode: str = "auto") -> None:
        self._mode = mode
        self._evidence = {"left": 0.0, "right": 0.0}
        self._locked_label: str | None = mode if mode in ("left", "right") else None
        self._previous_keypoints: np.ndarray | None = None
        self._previous_scores: np.ndarray | None = None
        self._previous_at: float | None = None
        self._last_motion_vote_at = float("-inf")
        self.motion_observations = 0

    def _add_evidence(self, side: str, weight: float) -> None:
        if self._mode != "auto" or self._locked_label is not None or weight <= 0:
            return
        self._evidence[side] += weight
        total = self._evidence["left"] + self._evidence["right"]
        winner = max(self._evidence, key=self._evidence.get)
        share = self._evidence[winner] / max(total, 1e-6)
        if total >= 10.0 and share >= 0.82:
            self._locked_label = winner

    def observe_pose(
        self,
        player: PlayerPose | None,
        captured_at: float,
        score_threshold: float,
    ) -> None:
        if self._mode != "auto" or player is None:
            self._previous_keypoints = None
            self._previous_scores = None
            self._previous_at = None
            return
        keypoints = player.keypoints
        scores = player.scores
        previous_keypoints = self._previous_keypoints
        previous_scores = self._previous_scores
        previous_at = self._previous_at
        self._previous_keypoints = keypoints.copy()
        self._previous_scores = scores.copy()
        self._previous_at = captured_at
        if previous_keypoints is None or previous_scores is None or previous_at is None:
            return
        dt = captured_at - previous_at
        required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST)
        if dt <= 0 or dt > 0.25 or any(
            min(float(scores[index]), float(previous_scores[index])) < score_threshold
            for index in required
        ):
            return
        shoulder_width = float(
            np.linalg.norm(keypoints[LEFT_SHOULDER] - keypoints[RIGHT_SHOULDER])
        )
        if shoulder_width < 5.0:
            return
        left_speed = float(
            np.linalg.norm(keypoints[LEFT_WRIST] - previous_keypoints[LEFT_WRIST])
        ) / (dt * shoulder_width)
        right_speed = float(
            np.linalg.norm(keypoints[RIGHT_WRIST] - previous_keypoints[RIGHT_WRIST])
        ) / (dt * shoulder_width)
        faster = max(left_speed, right_speed)
        slower = min(left_speed, right_speed)
        ratio = faster / max(slower, 0.25)
        if (
            faster < 1.5
            or ratio < 1.4
            or captured_at - self._last_motion_vote_at < 0.25
        ):
            return
        side = "left" if left_speed > right_speed else "right"
        confidence = min(float(scores[LEFT_WRIST]), float(scores[RIGHT_WRIST]))
        weight = confidence * min(0.15, 0.05 + 0.05 * (ratio - 1.4))
        self.motion_observations += 1
        self._last_motion_vote_at = captured_at
        self._add_evidence(side, weight)

    @property
    def snapshot(self) -> HandednessSnapshot:
        if self._mode in ("left", "right"):
            return HandednessSnapshot(
                self._mode, 1.0, True,
                1.0 if self._mode == "left" else 0.0,
                1.0 if self._mode == "right" else 0.0,
                self.motion_observations,
            )
        total = self._evidence["left"] + self._evidence["right"]
        winner = max(self._evidence, key=self._evidence.get)
        evidence_share = self._evidence[winner] / total if total > 0 else 0.0
        label = self._locked_label or (
            winner if total >= 4.0 and evidence_share >= 0.67 else "unknown"
        )
        confidence = (
            evidence_share
            if label != "unknown"
            else evidence_share * min(1.0, total / 4.0)
        )
        return HandednessSnapshot(
            label=label,
            confidence=confidence,
            locked=self._locked_label is not None,
            left_evidence=self._evidence["left"],
            right_evidence=self._evidence["right"],
            motion_observations=self.motion_observations,
        )


@dataclass(frozen=True)
class PoseFeatures:
    canonical_handedness: str
    torso_scale_px: float
    shoulder_angle_rad: float
    hip_angle_rad: float
    shoulder_hip_angle_rad: float
    left_wrist_speed: float | None
    right_wrist_speed: float | None
    dominant_wrist_x: float | None
    dominant_wrist_y: float | None
    dominant_wrist_vx: float | None
    dominant_wrist_vy: float | None
    dominant_wrist_speed: float | None
    dominant_wrist_acceleration: float | None
    dominant_elbow_angle_rad: float | None
    dominant_arm_extension: float | None
    wrist_distance: float
    dominant_crossed_midline: bool | None
    mean_keypoint_score: float


@dataclass(frozen=True)
class TemporalPoseFrame:
    sequence: int
    captured_at: float
    normalized_keypoints: np.ndarray
    scores: np.ndarray
    features: PoseFeatures


class PoseFeatureExtractor:
    """Normalize poses into a body frame and derive temporal stroke features."""

    def __init__(self) -> None:
        self._previous: TemporalPoseFrame | None = None
        self._previous_velocity: np.ndarray | None = None

    @staticmethod
    def _wrapped_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
        first = a - b
        second = c - b
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1e-6:
            return None
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        return float(np.arccos(cosine))

    @staticmethod
    def _canonicalize_left(
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        canonical_keypoints = keypoints.copy()
        canonical_scores = scores.copy()
        canonical_keypoints[:, 0] *= -1.0
        for left, right in BILATERAL_KEYPOINT_PAIRS:
            canonical_keypoints[[left, right]] = canonical_keypoints[[right, left]]
            canonical_scores[[left, right]] = canonical_scores[[right, left]]
        return canonical_keypoints, canonical_scores

    def extract(
        self,
        player: PlayerPose | None,
        sequence: int,
        captured_at: float,
        handedness: str,
        score_threshold: float,
    ) -> TemporalPoseFrame | None:
        if player is None or len(player.keypoints) < len(COCO_KEYPOINT_NAMES):
            self._previous = None
            self._previous_velocity = None
            return None
        required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if any(float(player.scores[index]) < score_threshold for index in required):
            self._previous = None
            self._previous_velocity = None
            return None
        points = player.keypoints.astype(np.float32, copy=False)
        scores = player.scores.astype(np.float32, copy=True)
        shoulder_vector = points[RIGHT_SHOULDER] - points[LEFT_SHOULDER]
        shoulder_width = float(np.linalg.norm(shoulder_vector))
        if shoulder_width < 5.0:
            self._previous = None
            self._previous_velocity = None
            return None
        hip_center = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
        shoulder_center = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
        x_axis = shoulder_vector / shoulder_width
        y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float32)
        if float(np.dot(y_axis, hip_center - shoulder_center)) < 0:
            y_axis *= -1.0
        relative = points - hip_center
        normalized = np.column_stack((relative @ x_axis, relative @ y_axis)) / shoulder_width
        canonical_handedness = handedness if handedness in ("left", "right") else "unknown"
        if canonical_handedness == "left":
            normalized, scores = self._canonicalize_left(normalized, scores)

        shoulder_angle = float(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
        hip_vector = points[RIGHT_HIP] - points[LEFT_HIP]
        hip_angle = float(np.arctan2(hip_vector[1], hip_vector[0]))
        dt: float | None = None
        velocity: np.ndarray | None = None
        acceleration: np.ndarray | None = None
        previous = self._previous
        if (
            previous is not None
            and previous.features.canonical_handedness == canonical_handedness
        ):
            dt = captured_at - previous.captured_at
            if 0 < dt <= 0.25:
                velocity = (normalized - previous.normalized_keypoints) / dt
                if self._previous_velocity is not None:
                    acceleration = (velocity - self._previous_velocity) / dt

        left_speed = float(np.linalg.norm(velocity[LEFT_WRIST])) if velocity is not None else None
        right_speed = float(np.linalg.norm(velocity[RIGHT_WRIST])) if velocity is not None else None
        known_hand = canonical_handedness in ("left", "right")
        dominant_wrist = RIGHT_WRIST if known_hand else None
        dominant_elbow = RIGHT_ELBOW if known_hand else None
        dominant_shoulder = RIGHT_SHOULDER if known_hand else None
        dominant_velocity = velocity[dominant_wrist] if velocity is not None and dominant_wrist is not None else None
        dominant_acceleration = (
            acceleration[dominant_wrist]
            if acceleration is not None and dominant_wrist is not None
            else None
        )
        dominant_elbow_angle = (
            self._joint_angle(
                normalized[dominant_shoulder],
                normalized[dominant_elbow],
                normalized[dominant_wrist],
            )
            if dominant_wrist is not None
            and dominant_elbow is not None
            and dominant_shoulder is not None
            and min(
                float(scores[dominant_shoulder]),
                float(scores[dominant_elbow]),
                float(scores[dominant_wrist]),
            ) >= score_threshold
            else None
        )
        features = PoseFeatures(
            canonical_handedness=canonical_handedness,
            torso_scale_px=shoulder_width,
            shoulder_angle_rad=shoulder_angle,
            hip_angle_rad=hip_angle,
            shoulder_hip_angle_rad=self._wrapped_angle(shoulder_angle - hip_angle),
            left_wrist_speed=left_speed,
            right_wrist_speed=right_speed,
            dominant_wrist_x=(float(normalized[dominant_wrist, 0]) if dominant_wrist is not None else None),
            dominant_wrist_y=(float(normalized[dominant_wrist, 1]) if dominant_wrist is not None else None),
            dominant_wrist_vx=(float(dominant_velocity[0]) if dominant_velocity is not None else None),
            dominant_wrist_vy=(float(dominant_velocity[1]) if dominant_velocity is not None else None),
            dominant_wrist_speed=(float(np.linalg.norm(dominant_velocity)) if dominant_velocity is not None else None),
            dominant_wrist_acceleration=(float(np.linalg.norm(dominant_acceleration)) if dominant_acceleration is not None else None),
            dominant_elbow_angle_rad=dominant_elbow_angle,
            dominant_arm_extension=(
                float(np.linalg.norm(normalized[dominant_wrist] - normalized[dominant_shoulder]))
                if dominant_wrist is not None and dominant_shoulder is not None
                else None
            ),
            wrist_distance=float(np.linalg.norm(normalized[LEFT_WRIST] - normalized[RIGHT_WRIST])),
            dominant_crossed_midline=(
                bool(normalized[dominant_wrist, 0] < 0.0)
                if dominant_wrist is not None
                else None
            ),
            mean_keypoint_score=float(np.mean(scores)),
        )
        temporal_frame = TemporalPoseFrame(
            sequence=sequence,
            captured_at=captured_at,
            normalized_keypoints=normalized.astype(np.float32),
            scores=scores,
            features=features,
        )
        self._previous = temporal_frame
        self._previous_velocity = velocity
        return temporal_frame


class TemporalPoseBuffer:
    """Keep only the latest time-bounded window of normalized poses."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self._frames: deque[TemporalPoseFrame] = deque()
        self._canonical_handedness: str | None = None

    def add(self, frame: TemporalPoseFrame) -> None:
        handedness = frame.features.canonical_handedness
        if self._canonical_handedness is not None and handedness != self._canonical_handedness:
            self._frames.clear()
        self._canonical_handedness = handedness
        self._frames.append(frame)
        cutoff = frame.captured_at - self.duration_seconds
        while self._frames and self._frames[0].captured_at < cutoff:
            self._frames.popleft()

    @property
    def frames(self) -> tuple[TemporalPoseFrame, ...]:
        return tuple(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


class PoseFeatureLogger:
    """Write normalized pose samples as newline-delimited JSON for analysis."""

    def __init__(self, path: Path, session_started: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._session_started = session_started
        self._file = path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, frame: TemporalPoseFrame) -> None:
        features = {
            name: getattr(frame.features, name)
            for name in PoseFeatures.__dataclass_fields__
        }
        record = {
            "sequence": frame.sequence,
            "time_seconds": frame.captured_at - self._session_started,
            "normalized_keypoints": frame.normalized_keypoints.tolist(),
            "scores": frame.scores.tolist(),
            "features": features,
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self._file.close()


@dataclass
class EdgeDiagnosticStats:
    events: int = 0
    sides: Counter[str] = field(default_factory=Counter)
    joints: Counter[str] = field(default_factory=Counter)
    landmark_counts: Counter[int] = field(default_factory=Counter)
    clamped: Counter[str] = field(default_factory=Counter)
    confidence_samples: list[float] = field(default_factory=list)
    clamped_only_suppressed: int = 0
    repeated_same_hit_events: int = 0
    max_same_hit_streak: int = 0
    _previous_hits: set[tuple[str, str]] = field(default_factory=set)
    _same_hit_streak: int = 0

    def observe(self, assessment: RedetectionAssessment) -> None:
        if not assessment.edge_hits:
            self._previous_hits.clear()
            self._same_hit_streak = 0
            return
        self.events += 1
        if assessment.clamped_only_suppressed:
            self.clamped_only_suppressed += 1
        current_hits = {(hit.joint, hit.edge) for hit in assessment.edge_hits}
        repeated = bool(current_hits & self._previous_hits)
        if repeated:
            self.repeated_same_hit_events += 1
            self._same_hit_streak += 1
        else:
            self._same_hit_streak = 1
        self.max_same_hit_streak = max(self.max_same_hit_streak, self._same_hit_streak)
        self._previous_hits = current_hits
        unique_joints = {hit.joint for hit in assessment.edge_hits}
        self.landmark_counts[len(unique_joints)] += 1
        hit_edges = {hit.edge for hit in assessment.edge_hits}
        for hit in assessment.edge_hits:
            self.sides[hit.edge] += 1
            self.joints[hit.joint] += 1
            self.confidence_samples.append(hit.confidence)
        clamped_hits = hit_edges & set(assessment.clamped_edges)
        if clamped_hits:
            self.clamped.update(clamped_hits)
        else:
            self.clamped["not_clamped"] += 1


class LatestFrameCapture:
    """Capture continuously and expose only the newest live-timeline frame."""

    def __init__(
        self,
        source: int | str,
        width: int,
        height: int,
        realtime_video: bool = False,
    ) -> None:
        self._capture = cv2.VideoCapture(source)
        if not realtime_video:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            kind = "video" if realtime_video else "camera"
            raise RuntimeError(f"Could not open {kind}: {source}")
        self._realtime_fps = float(self._capture.get(cv2.CAP_PROP_FPS)) if realtime_video else 0.0
        if realtime_video and self._realtime_fps <= 0:
            self._capture.release()
            raise RuntimeError("Video does not report a valid FPS for real-time playback")
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._sequence = -1
        self._first_captured_at: float | None = None
        self._latest_captured_at: float | None = None
        self._stopped = False
        self._ended = False
        self._error: RuntimeError | None = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()

    @property
    def latest_sequence(self) -> int:
        with self._condition:
            return self._sequence

    @property
    def stats(self) -> CaptureStats:
        with self._condition:
            return CaptureStats(
                count=self._sequence + 1,
                first_captured_at=self._first_captured_at,
                latest_captured_at=self._latest_captured_at,
            )

    def _capture_loop(self) -> None:
        timeline_started = time.perf_counter()
        next_sequence = 0
        try:
            while True:
                with self._condition:
                    if self._stopped:
                        break
                decode_started = time.perf_counter()
                ok, frame = self._capture.read()
                decoded_at = time.perf_counter()
                if not ok:
                    with self._condition:
                        if self._realtime_fps > 0:
                            self._ended = True
                        elif not self._stopped:
                            self._error = RuntimeError("Camera stopped returning frames")
                        self._condition.notify_all()
                    break
                if self._realtime_fps > 0:
                    release_at = timeline_started + next_sequence / self._realtime_fps
                    with self._condition:
                        while not self._stopped:
                            remaining = release_at - time.perf_counter()
                            if remaining <= 0:
                                break
                            self._condition.wait(timeout=remaining)
                        if self._stopped:
                            break
                    captured_at = time.perf_counter()
                else:
                    captured_at = decoded_at
                with self._condition:
                    self._sequence = next_sequence
                    if self._first_captured_at is None:
                        self._first_captured_at = captured_at
                    self._latest_captured_at = captured_at
                    self._latest = FramePacket(
                        frame=frame,
                        decode_seconds=decoded_at - decode_started,
                        captured_at=captured_at,
                        sequence=self._sequence,
                    )
                    self._condition.notify_all()
                next_sequence += 1
        finally:
            self._capture.release()

    def __iter__(self) -> Iterator[FramePacket]:
        last_sequence = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._sequence > last_sequence
                        or self._error is not None
                        or self._ended
                        or self._stopped
                    )
                )
                if self._error is not None:
                    raise self._error
                if self._sequence <= last_sequence and (self._ended or self._stopped):
                    return
                assert self._latest is not None
                packet = self._latest
            dropped = max(0, packet.sequence - last_sequence - 1)
            last_sequence = packet.sequence
            yield FramePacket(
                frame=packet.frame,
                decode_seconds=packet.decode_seconds,
                captured_at=packet.captured_at,
                sequence=packet.sequence,
                dropped=dropped,
            )

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            # Some camera backends block inside read(); releasing here unblocks them.
            self._capture.release()
            self._thread.join(timeout=1.0)


class LatestDetectorWorker:
    """Run at most one detector refresh at a time on a background thread."""

    def __init__(self, inferencer: Any) -> None:
        self._inferencer = inferencer
        self._condition = threading.Condition()
        self._pending: DetectorRequest | None = None
        self._result: DetectorResult | None = None
        self._busy = False
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="rtmdet-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: DetectorRequest) -> bool:
        """Submit a request only when no older request or result is outstanding."""
        with self._condition:
            if self._stopped or self._busy or self._pending is not None or self._result is not None:
                return False
            self._pending = request
            self._condition.notify_all()
            return True

    def poll(self) -> DetectorResult | None:
        with self._condition:
            result = self._result
            self._result = None
            return result

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._stopped)
                if self._stopped and self._pending is None:
                    return
                request = self._pending
                self._pending = None
                self._busy = True
            assert request is not None
            started = time.perf_counter()
            prediction: Any | None = None
            error: BaseException | None = None
            try:
                prediction = self._inferencer(request.frame, return_vis=False)
            except BaseException as exc:
                error = exc
            completed_at = time.perf_counter()
            with self._condition:
                self._busy = False
                self._result = DetectorResult(
                    request=request,
                    prediction=prediction,
                    inference_seconds=completed_at - started,
                    completed_at=completed_at,
                    error=error,
                )
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join()


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
    dominant_hand: str = "unknown",
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
    dominant_wrist = {
        "left": LEFT_WRIST,
        "right": RIGHT_WRIST,
    }.get(dominant_hand)
    for index, (point, is_visible) in enumerate(zip(player.keypoints, visible)):
        if is_visible:
            color = (40, 230, 40) if index == dominant_wrist else (40, 40, 255)
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), max(2, round(5 * visual_scale)),
                       color, -1, cv2.LINE_AA)
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


def assess_pose_redetection(
    player: PlayerPose | None,
    crop_bounds: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    score_threshold: float,
    min_keypoints: int,
    edge_fraction: float,
) -> RedetectionAssessment:
    if player is None:
        return RedetectionAssessment("missing_pose")
    visible_indices = np.flatnonzero(player.scores >= score_threshold)
    if len(visible_indices) < min_keypoints:
        return RedetectionAssessment("low_keypoints")
    x1, y1, x2, y2 = crop_bounds
    edge_x = (x2 - x1) * edge_fraction
    edge_y = (y2 - y1) * edge_fraction
    hits: list[EdgeHit] = []
    for index in visible_indices:
        point = player.keypoints[index]
        joint = COCO_KEYPOINT_NAMES[index] if index < len(COCO_KEYPOINT_NAMES) else str(index)
        confidence = float(player.scores[index])
        if point[0] <= x1 + edge_x:
            hits.append(EdgeHit(joint, "left", confidence))
        if point[0] >= x2 - edge_x:
            hits.append(EdgeHit(joint, "right", confidence))
        if point[1] <= y1 + edge_y:
            hits.append(EdgeHit(joint, "top", confidence))
        if point[1] >= y2 - edge_y:
            hits.append(EdgeHit(joint, "bottom", confidence))
    frame_height, frame_width = frame_shape[:2]
    clamped_edges: list[str] = []
    if x1 <= 0:
        clamped_edges.append("left")
    if x2 >= frame_width:
        clamped_edges.append("right")
    if y1 <= 0:
        clamped_edges.append("top")
    if y2 >= frame_height:
        clamped_edges.append("bottom")
    clamped_set = set(clamped_edges)
    actionable_hits = [hit for hit in hits if hit.edge not in clamped_set]
    return RedetectionAssessment(
        reason="crop_edge" if actionable_hits else None,
        edge_hits=tuple(hits),
        clamped_edges=tuple(clamped_edges),
        clamped_only_suppressed=bool(hits and not actionable_hits),
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


def percentile_ms(timings: deque[float] | list[float], percentile: float) -> str:
    if not timings:
        return "--"
    return f"{1000.0 * float(np.percentile(np.asarray(timings), percentile)):.1f}"


def maximum_ms(timings: list[float]) -> str:
    return f"{1000.0 * max(timings):.1f}" if timings else "--"


def rate(count: int, duration: float) -> float:
    return count / duration if duration > 0 else 0.0


def format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in counts.most_common())


def format_top_counts(counts: Counter[Any], limit: int = 8) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in counts.most_common(limit))


def percentile_value(samples: list[float], percentile: float) -> str:
    if not samples:
        return "--"
    return f"{float(np.percentile(np.asarray(samples), percentile)):.2f}"


def frames(
    camera: int | None,
    video: Path | None,
    width: int,
    height: int,
) -> Iterator[FramePacket]:
    source: int | str = str(video) if video is not None else (camera if camera is not None else 0)
    capture = cv2.VideoCapture(source)
    if video is None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        kind = "video" if video is not None else "camera"
        raise RuntimeError(f"Could not open {kind}: {source}")
    try:
        sequence = 0
        while True:
            decode_started = time.perf_counter()
            ok, frame = capture.read()
            captured_at = time.perf_counter()
            decode_seconds = captured_at - decode_started
            if not ok:
                if video is not None:
                    break
                raise RuntimeError("Camera stopped returning frames")
            yield FramePacket(frame, decode_seconds, captured_at, sequence)
            sequence += 1
    finally:
        capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a tennis player's body with RTMPose")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--async-camera",
        action="store_true",
        help="Capture continuously and process only the newest live-camera frame",
    )
    parser.add_argument(
        "--realtime-video",
        action="store_true",
        help="Play a video at its recorded FPS through the asynchronous latest-frame buffer",
    )
    parser.add_argument(
        "--metrics-warmup-seconds",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Exclude initial seconds from steady-state metrics (default: 3)",
    )
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
        "--detector-interval-seconds",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Detect periodically by elapsed time instead of processed frames (0: disabled)",
    )
    parser.add_argument(
        "--async-detector",
        action="store_true",
        help="Run periodic RTMDet refreshes in a single-flight background worker",
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
        "--handedness",
        choices=("auto", "left", "right"),
        default="auto",
        help="Player handedness or automatic temporal inference (default: auto)",
    )
    parser.add_argument(
        "--pose-buffer-seconds",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Duration of the in-memory normalized pose window (default: 2)",
    )
    parser.add_argument(
        "--pose-log",
        type=Path,
        metavar="PATH",
        help="Optionally write normalized poses and temporal features as JSONL",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Display scale from 0 to 1; inference remains full resolution (default: 1.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the graphical preview while retaining live and final performance metrics",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Headless live-metrics reporting interval (default: 2)",
    )
    args = parser.parse_args()
    if args.infer_every < 1:
        parser.error("--infer-every must be at least 1")
    if args.async_camera and args.video is not None:
        parser.error("--async-camera is only available with a live camera")
    if args.realtime_video and args.video is None:
        parser.error("--realtime-video requires --video")
    if args.async_camera and args.realtime_video:
        parser.error("--async-camera and --realtime-video cannot be combined")
    if args.metrics_warmup_seconds < 0:
        parser.error("--metrics-warmup-seconds cannot be negative")
    if args.whole_image and args.det_model:
        parser.error("--whole-image cannot be combined with --det-model")
    if args.detector_interval < 0:
        parser.error("--detector-interval cannot be negative")
    if args.detector_interval_seconds < 0:
        parser.error("--detector-interval-seconds cannot be negative")
    if args.detector_interval and args.detector_interval_seconds:
        parser.error(
            "--detector-interval and --detector-interval-seconds cannot be combined"
        )
    hybrid_enabled = bool(args.detector_interval or args.detector_interval_seconds)
    if hybrid_enabled and args.whole_image:
        parser.error("detector intervals cannot be combined with --whole-image")
    if hybrid_enabled and args.det_model:
        parser.error("detector intervals cannot currently be combined with --det-model")
    if args.detector_device and not hybrid_enabled:
        parser.error("--detector-device requires a detector interval")
    if args.async_detector and not hybrid_enabled:
        parser.error("--async-detector requires a detector interval")
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
    if args.status_interval <= 0:
        parser.error("--status-interval must be greater than 0")
    if args.pose_buffer_seconds <= 0:
        parser.error("--pose-buffer-seconds must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    hybrid_enabled = bool(args.detector_interval or args.detector_interval_seconds)
    handedness_estimator = HandednessEstimator(args.handedness)
    pose_feature_extractor = PoseFeatureExtractor()
    temporal_pose_buffer = TemporalPoseBuffer(args.pose_buffer_seconds)
    pose_model_name = POSE_MODEL_PRESETS.get(args.model, args.model)
    base_kwargs: dict[str, Any] = {"pose2d": pose_model_name, "device": args.device}
    if hybrid_enabled:
        detector_device = args.detector_device or ("cpu" if args.device == "mps" else args.device)
        schedule = (
            f"detect every {args.detector_interval_seconds:g} seconds"
            if args.detector_interval_seconds
            else f"detect every {args.detector_interval} processed frames"
        )
        mode = (
            f"hybrid tracking (pose={args.device or 'auto'}, "
            f"detector={detector_device or 'auto'}, "
            f"{schedule}"
            f"{' asynchronously' if args.async_detector else ''})"
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
    if hybrid_enabled:
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
    session_pose_timings: list[float] = []
    handedness_timings: list[float] = []
    pose_feature_timings: list[float] = []
    pose_log_timings: list[float] = []
    temporal_pose_frames = 0
    display_timings: deque[float] = deque(maxlen=60)
    loop_timings: deque[float] = deque(maxlen=60)
    inference_wait_timings: deque[float] = deque(maxlen=120)
    frame_age_timings: deque[float] = deque(maxlen=120)
    sequence_lags: deque[int] = deque(maxlen=120)
    session_inference_waits: list[float] = []
    session_frame_ages: list[float] = []
    steady_inference_waits: list[float] = []
    steady_frame_ages: list[float] = []
    steady_detector_ages: list[float] = []
    steady_pose_ages: list[float] = []
    max_sequence_lag = 0
    steady_max_sequence_lag = 0
    dropped_frames = 0
    processed_frames = 0
    inferred_frames = 0
    pose_output_frames = 0
    steady_pose_output_frames = 0
    steady_processed_frames = 0
    detector_refreshes = 0
    steady_detector_refreshes = 0
    detector_reason_counts: Counter[str] = Counter()
    steady_detector_reason_counts: Counter[str] = Counter()
    detector_result_latencies: list[float] = []
    detector_result_lags: list[int] = []
    steady_detector_result_latencies: list[float] = []
    steady_detector_result_lags: list[int] = []
    detector_results_completed = 0
    steady_detector_results_completed = 0
    edge_diagnostics = EdgeDiagnosticStats()
    steady_edge_diagnostics = EdgeDiagnosticStats()
    sync_first_captured_at: float | None = None
    sync_latest_captured_at: float | None = None
    player: PlayerPose | None = None
    player_crop: tuple[int, int, int, int] | None = None
    last_detection_frame = -args.detector_interval
    last_detection_at = float("-inf")
    redetection_reason: str | None = None
    detector_worker = (
        LatestDetectorWorker(detector_inferencer)
        if args.async_detector and detector_inferencer is not None
        else None
    )
    if not args.headless:
        cv2.namedWindow("RTMPose Tennis", cv2.WINDOW_NORMAL)
    else:
        print(
            "Headless mode enabled: preview rendering is disabled; "
            "press Ctrl+C to stop a live session.",
            flush=True,
        )
    async_capture: LatestFrameCapture | None = None
    if args.async_camera or args.realtime_video:
        if args.realtime_video:
            assert args.video is not None
            async_capture = LatestFrameCapture(
                str(args.video), args.width, args.height, realtime_video=True
            )
        else:
            async_capture = LatestFrameCapture(
                args.camera if args.camera is not None else 0, args.width, args.height
            )
        frame_source: Iterator[FramePacket] = iter(async_capture)
        source_mode = "Real-time video simulation" if args.realtime_video else "Async camera capture"
        print(f"{source_mode} enabled: stale frames will be replaced.", flush=True)
    else:
        frame_source = frames(args.camera, args.video, args.width, args.height)
    session_started = time.perf_counter()
    pose_feature_logger = (
        PoseFeatureLogger(args.pose_log, session_started)
        if args.pose_log is not None
        else None
    )
    warmup_ends = session_started + args.metrics_warmup_seconds
    session_ended = session_started
    status_last_at = session_started
    status_last_pose_outputs = 0
    try:
        for frame_index, packet in enumerate(frame_source):
            loop_started = time.perf_counter()
            processed_frames += 1
            if sync_first_captured_at is None:
                sync_first_captured_at = packet.captured_at
            sync_latest_captured_at = packet.captured_at
            frame = packet.frame
            if detector_worker is not None:
                completed_detection = detector_worker.poll()
                if completed_detection is not None:
                    if completed_detection.error is not None:
                        raise RuntimeError("Background detector failed") from completed_detection.error
                    detector_results_completed += 1
                    detector_timings.append(completed_detection.inference_seconds)
                    result_latency = time.perf_counter() - completed_detection.request.requested_at
                    result_lag = max(0, packet.sequence - completed_detection.request.sequence)
                    detector_result_latencies.append(result_latency)
                    detector_result_lags.append(result_lag)
                    if time.perf_counter() >= warmup_ends:
                        steady_detector_results_completed += 1
                        steady_detector_result_latencies.append(result_latency)
                        steady_detector_result_lags.append(result_lag)
                    player_bbox = select_player_bbox(
                        completed_detection.prediction,
                        completed_detection.request.frame_shape,
                        args.detector_score_threshold,
                    )
                    detected_crop = (
                        crop_around_bbox(
                            player_bbox,
                            completed_detection.request.frame_shape,
                            args.crop_margin,
                        )
                        if player_bbox is not None
                        else None
                    )
                    if detected_crop is not None:
                        player_crop = detected_crop
                        redetection_reason = None
                    elif player_crop is None:
                        redetection_reason = "missing_crop"
            decode_timings.append(packet.decode_seconds)
            dropped_frames += packet.dropped
            frame_stage = "reuse"
            if frame_index % args.infer_every == 0:
                inferred_frames += 1
                inference_started = time.perf_counter()
                inference_wait = inference_started - packet.captured_at
                inference_wait_timings.append(inference_wait)
                session_inference_waits.append(inference_wait)
                if inference_started >= warmup_ends:
                    steady_inference_waits.append(inference_wait)
                if hybrid_enabled:
                    detection_now = time.perf_counter()
                    if player_crop is None:
                        detection_reason = "missing_crop"
                    elif redetection_reason is not None:
                        detection_reason = redetection_reason
                    elif (
                        args.detector_interval_seconds
                        and detection_now - last_detection_at
                        >= args.detector_interval_seconds
                    ):
                        detection_reason = "scheduled_interval"
                    elif (
                        args.detector_interval
                        and frame_index - last_detection_frame >= args.detector_interval
                    ):
                        detection_reason = "scheduled_interval"
                    else:
                        detection_reason = None
                    if detection_reason is not None:
                        assert detector_inferencer is not None
                        if detector_worker is not None:
                            accepted = detector_worker.submit(DetectorRequest(
                                frame=frame.copy(),
                                frame_shape=frame.shape,
                                frame_index=frame_index,
                                sequence=packet.sequence,
                                requested_at=time.perf_counter(),
                                reason=detection_reason,
                            ))
                            if accepted:
                                detector_refreshes += 1
                                detector_reason_counts[detection_reason] += 1
                                if inference_started >= warmup_ends:
                                    steady_detector_refreshes += 1
                                    steady_detector_reason_counts[detection_reason] += 1
                                last_detection_frame = frame_index
                                last_detection_at = detection_now
                        else:
                            frame_stage = "detector"
                            detector_refreshes += 1
                            detector_reason_counts[detection_reason] += 1
                            if inference_started >= warmup_ends:
                                steady_detector_refreshes += 1
                                steady_detector_reason_counts[detection_reason] += 1
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
                                redetection_reason = None
                            last_detection_frame = frame_index
                            last_detection_at = detection_now
                    if player_crop is not None:
                        if frame_stage != "detector":
                            frame_stage = "pose"
                        assert crop_inferencer is not None
                        x1, y1, x2, y2 = player_crop
                        crop = frame[y1:y2, x1:x2]
                        pose_started = time.perf_counter()
                        result = next(crop_inferencer(crop, return_vis=False))
                        pose_seconds = time.perf_counter() - pose_started
                        pose_timings.append(pose_seconds)
                        session_pose_timings.append(pose_seconds)
                        crop_player = select_player(result, crop.shape)
                        player = move_pose_to_frame(crop_player, player_crop)
                        if player is not None:
                            # Let the lightweight pose tracker correct the crop before
                            # deciding whether an expensive detector refresh is needed.
                            player_crop = update_crop_from_pose(
                                player,
                                player_crop,
                                frame.shape,
                                args.score_threshold,
                                args.crop_margin,
                                args.tracking_alpha,
                            )
                            player.bbox = np.asarray(player_crop, dtype=np.float32)
                        assessment = assess_pose_redetection(
                            player,
                            player_crop,
                            frame.shape,
                            args.score_threshold,
                            args.tracking_min_keypoints,
                            args.redetect_edge,
                        )
                        redetection_reason = assessment.reason
                        edge_diagnostics.observe(assessment)
                        if time.perf_counter() >= warmup_ends:
                            steady_edge_diagnostics.observe(assessment)
                    else:
                        player = None
                        redetection_reason = "missing_crop"
                        missing_crop_assessment = RedetectionAssessment("missing_crop")
                        edge_diagnostics.observe(missing_crop_assessment)
                        if time.perf_counter() >= warmup_ends:
                            steady_edge_diagnostics.observe(missing_crop_assessment)
                else:
                    frame_stage = "pose" if args.whole_image else "detector"
                    if frame_stage == "detector":
                        detector_refreshes += 1
                        if inference_started >= warmup_ends:
                            steady_detector_refreshes += 1
                    assert inferencer is not None
                    stage_started = time.perf_counter()
                    result = next(inferencer(frame, return_vis=False))
                    stage_seconds = time.perf_counter() - stage_started
                    if args.whole_image:
                        pose_timings.append(stage_seconds)
                        session_pose_timings.append(stage_seconds)
                    else:
                        detector_timings.append(stage_seconds)
                    player = select_player(result, frame.shape)
                handedness_started = time.perf_counter()
                handedness_estimator.observe_pose(
                    player,
                    packet.captured_at,
                    args.score_threshold,
                )
                handedness_timings.append(time.perf_counter() - handedness_started)
                feature_started = time.perf_counter()
                temporal_pose = pose_feature_extractor.extract(
                    player=player,
                    sequence=packet.sequence,
                    captured_at=packet.captured_at,
                    handedness=handedness_estimator.snapshot.label,
                    score_threshold=args.score_threshold,
                )
                if temporal_pose is not None:
                    temporal_pose_buffer.add(temporal_pose)
                    temporal_pose_frames += 1
                pose_feature_timings.append(time.perf_counter() - feature_started)
                if temporal_pose is not None and pose_feature_logger is not None:
                    log_started = time.perf_counter()
                    pose_feature_logger.write(temporal_pose)
                    pose_log_timings.append(time.perf_counter() - log_started)
                inference_timings.append(time.perf_counter() - inference_started)
                if player is not None:
                    pose_output_frames += 1
                    if time.perf_counter() >= warmup_ends:
                        steady_pose_output_frames += 1
            if args.headless:
                completed_at = time.perf_counter()
                frame_age = completed_at - packet.captured_at
                frame_age_timings.append(frame_age)
                session_frame_ages.append(frame_age)
                latest_sequence = (
                    async_capture.latest_sequence
                    if async_capture is not None
                    else packet.sequence
                )
                sequence_lag = max(0, latest_sequence - packet.sequence)
                sequence_lags.append(sequence_lag)
                max_sequence_lag = max(max_sequence_lag, sequence_lag)
                session_ended = completed_at
                if completed_at >= warmup_ends:
                    steady_processed_frames += 1
                    steady_frame_ages.append(frame_age)
                    steady_max_sequence_lag = max(steady_max_sequence_lag, sequence_lag)
                    if frame_stage == "detector":
                        steady_detector_ages.append(frame_age)
                    elif frame_stage == "pose":
                        steady_pose_ages.append(frame_age)
                if completed_at - status_last_at >= args.status_interval:
                    interval_duration = completed_at - status_last_at
                    interval_pose_outputs = pose_output_frames - status_last_pose_outputs
                    captured_so_far = (
                        async_capture.latest_sequence + 1
                        if async_capture is not None
                        else packet.sequence + 1
                    )
                    skipped_so_far = max(
                        dropped_frames,
                        captured_so_far - processed_frames,
                    )
                    status_handedness = handedness_estimator.snapshot
                    print(
                        "Headless status: "
                        f"pose output={rate(interval_pose_outputs, interval_duration):.1f} FPS, "
                        f"processed={rate(processed_frames, completed_at - session_started):.1f} FPS, "
                        f"age p50/p95={percentile_ms(frame_age_timings, 50)}/"
                        f"{percentile_ms(frame_age_timings, 95)} ms, "
                        f"pose={average_ms(pose_timings)} ms, "
                        f"hand={status_handedness.label} "
                        f"({status_handedness.confidence:.0%}), "
                        f"pose window={len(temporal_pose_buffer)}, "
                        f"dropped={100.0 * skipped_so_far / max(captured_so_far, 1):.1f}%",
                        flush=True,
                    )
                    status_last_at = completed_at
                    status_last_pose_outputs = pose_output_frames
                loop_timings.append(completed_at - loop_started)
                continue
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
            handedness = handedness_estimator.snapshot
            canvas = draw_player(
                preview_frame,
                preview_player,
                args.score_threshold,
                visual_scale=args.preview_scale,
                dominant_hand=handedness.label,
            )
            model_fps = len(inference_timings) / max(sum(inference_timings), 1e-6)
            display_fps = len(loop_timings) / max(sum(loop_timings), 1e-6) if loop_timings else 0.0
            status = "player" if player is not None else "no player"
            text_x = max(5, round(20 * args.preview_scale))
            main_y = max(18, round(35 * args.preview_scale))
            input_y = max(main_y + 14, round(65 * args.preview_scale))
            output_y = max(input_y + 14, round(92 * args.preview_scale))
            latency_y = max(output_y + 14, round(119 * args.preview_scale))
            main_font = max(0.4, 0.8 * args.preview_scale)
            detail_font = max(0.3, 0.55 * args.preview_scale)
            text_thickness = max(1, round(2 * args.preview_scale))
            cv2.putText(
                canvas,
                f"{status} | hand {handedness.label} {handedness.confidence:.0%} | "
                f"model {model_fps:.1f} | "
                f"output {display_fps:.1f} FPS | Q",
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
            captured_so_far = (
                async_capture.latest_sequence + 1
                if async_capture is not None
                else packet.sequence + 1
            )
            skipped_so_far = max(dropped_frames, captured_so_far - processed_frames)
            drop_percentage = 100.0 * skipped_so_far / max(captured_so_far, 1)
            latency_metrics = (
                f"age p50/p95 {percentile_ms(frame_age_timings, 50)}/"
                f"{percentile_ms(frame_age_timings, 95)} ms | "
                f"wait p95 {percentile_ms(inference_wait_timings, 95)} ms | "
                f"lag {max(sequence_lags, default=0)} | drop {drop_percentage:.1f}%"
            )
            cv2.putText(
                canvas, input_metrics, (text_x, input_y), cv2.FONT_HERSHEY_SIMPLEX,
                detail_font, (40, 240, 40), text_thickness, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, output_metrics, (text_x, output_y), cv2.FONT_HERSHEY_SIMPLEX,
                detail_font, (40, 240, 40), text_thickness, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, latency_metrics, (text_x, latency_y), cv2.FONT_HERSHEY_SIMPLEX,
                detail_font, (40, 240, 40), text_thickness, cv2.LINE_AA,
            )
            # `player.keypoints` and `player.scores` are the stable hand-off to the
            # future temporal gesture classifier and feedback engine.
            cv2.imshow("RTMPose Tennis", canvas)
            key = cv2.waitKey(1) & 0xFF
            displayed_at = time.perf_counter()
            display_timings.append(displayed_at - display_started)
            frame_age = displayed_at - packet.captured_at
            frame_age_timings.append(frame_age)
            session_frame_ages.append(frame_age)
            latest_sequence = async_capture.latest_sequence if async_capture is not None else packet.sequence
            sequence_lag = max(0, latest_sequence - packet.sequence)
            sequence_lags.append(sequence_lag)
            max_sequence_lag = max(max_sequence_lag, sequence_lag)
            session_ended = displayed_at
            if displayed_at >= warmup_ends:
                steady_processed_frames += 1
                steady_frame_ages.append(frame_age)
                steady_max_sequence_lag = max(steady_max_sequence_lag, sequence_lag)
                if frame_stage == "detector":
                    steady_detector_ages.append(frame_age)
                elif frame_stage == "pose":
                    steady_pose_ages.append(frame_age)
            if key in (ord("q"), 27):
                break
            loop_timings.append(time.perf_counter() - loop_started)
    except KeyboardInterrupt:
        session_ended = time.perf_counter()
        print("Stopping on Ctrl+C...", flush=True)
    finally:
        capture_stats = async_capture.stats if async_capture is not None else CaptureStats(
            count=processed_frames,
            first_captured_at=sync_first_captured_at,
            latest_captured_at=sync_latest_captured_at,
        )
        if async_capture is not None:
            async_capture.close()
        if detector_worker is not None:
            detector_worker.close()
        if pose_feature_logger is not None:
            pose_feature_logger.close()
        if not args.headless:
            cv2.destroyAllWindows()
    session_duration = max(0.0, session_ended - session_started)
    capture_duration = (
        capture_stats.latest_captured_at - capture_stats.first_captured_at
        if capture_stats.first_captured_at is not None
        and capture_stats.latest_captured_at is not None
        else 0.0
    )
    capture_fps = rate(max(0, capture_stats.count - 1), capture_duration)
    output_fps = rate(processed_frames, session_duration)
    skipped_frames = max(dropped_frames, capture_stats.count - processed_frames)
    drop_percentage = 100.0 * skipped_frames / max(capture_stats.count, 1)
    steady_duration = max(0.0, session_ended - max(session_started, warmup_ends))
    print(
        "Session throughput: "
        f"duration={session_duration:.1f} s, "
        f"captured={capture_stats.count} ({capture_fps:.1f} FPS), "
        f"processed={processed_frames} ({output_fps:.1f} FPS), "
        f"inferred={inferred_frames}, detector refreshes={detector_refreshes}, "
        f"dropped={skipped_frames} ({drop_percentage:.1f}%)",
        flush=True,
    )
    if args.headless:
        print(
            "Headless pose output: "
            f"all={rate(pose_output_frames, session_duration):.1f} FPS "
            f"(n={pose_output_frames}), "
            f"steady={rate(steady_pose_output_frames, steady_duration):.1f} FPS "
            f"(n={steady_pose_output_frames}), "
            f"crop pose latency p50/p95={percentile_ms(session_pose_timings, 50)}/"
            f"{percentile_ms(session_pose_timings, 95)} ms",
            flush=True,
        )
    print(
        "Session freshness (all): "
        f"age p50={percentile_ms(session_frame_ages, 50)} ms, "
        f"p95={percentile_ms(session_frame_ages, 95)} ms, "
        f"max={maximum_ms(session_frame_ages)} ms, "
        f"wait p95={percentile_ms(session_inference_waits, 95)} ms, "
        f"max lag={max_sequence_lag} frame(s)",
        flush=True,
    )
    print(
        f"Steady state (after {args.metrics_warmup_seconds:.1f} s): "
        f"duration={steady_duration:.1f} s, "
        f"processed={steady_processed_frames} ({rate(steady_processed_frames, steady_duration):.1f} FPS), "
        f"age p50={percentile_ms(steady_frame_ages, 50)} ms, "
        f"p95={percentile_ms(steady_frame_ages, 95)} ms, "
        f"max={maximum_ms(steady_frame_ages)} ms, "
        f"wait p95={percentile_ms(steady_inference_waits, 95)} ms, "
        f"max lag={steady_max_sequence_lag} frame(s)",
        flush=True,
    )
    print(
        "Steady frame classes: "
        f"pose age p50/p95={percentile_ms(steady_pose_ages, 50)}/"
        f"{percentile_ms(steady_pose_ages, 95)} ms (n={len(steady_pose_ages)}), "
        f"detector age p50/p95={percentile_ms(steady_detector_ages, 50)}/"
        f"{percentile_ms(steady_detector_ages, 95)} ms "
        f"(n={len(steady_detector_ages)}, refreshes={steady_detector_refreshes})",
        flush=True,
    )
    handedness = handedness_estimator.snapshot
    print(
        "Handedness: "
        f"prediction={handedness.label}, confidence={handedness.confidence:.1%}, "
        f"locked={'yes' if handedness.locked else 'no'}, "
        f"evidence left/right={handedness.left_evidence:.2f}/"
        f"{handedness.right_evidence:.2f}, "
        f"motion observations={handedness.motion_observations}, "
        f"overhead p50/p95={percentile_ms(handedness_timings, 50)}/"
        f"{percentile_ms(handedness_timings, 95)} ms",
        flush=True,
    )
    print(
        "Temporal pose pipeline: "
        f"generated={temporal_pose_frames}, "
        f"window={len(temporal_pose_buffer)} frame(s)/"
        f"{args.pose_buffer_seconds:g} s, "
        f"feature overhead p50/p95={percentile_ms(pose_feature_timings, 50)}/"
        f"{percentile_ms(pose_feature_timings, 95)} ms, "
        f"logged={pose_feature_logger.count if pose_feature_logger is not None else 0}, "
        f"log overhead p50/p95={percentile_ms(pose_log_timings, 50)}/"
        f"{percentile_ms(pose_log_timings, 95)} ms",
        flush=True,
    )
    if pose_feature_logger is not None:
        print(f"Pose feature log: {pose_feature_logger.path}", flush=True)
    if hybrid_enabled:
        if detector_worker is not None:
            print(
                "Background detector: "
                f"submitted={detector_refreshes}, completed={detector_results_completed}, "
                f"latency p50/p95={percentile_ms(detector_result_latencies, 50)}/"
                f"{percentile_ms(detector_result_latencies, 95)} ms, "
                f"result lag p50/p95={percentile_value(detector_result_lags, 50)}/"
                f"{percentile_value(detector_result_lags, 95)} frame(s)",
                flush=True,
            )
            print(
                "Background detector (steady): "
                f"completed={steady_detector_results_completed}, "
                f"latency p50/p95={percentile_ms(steady_detector_result_latencies, 50)}/"
                f"{percentile_ms(steady_detector_result_latencies, 95)} ms, "
                f"result lag p50/p95={percentile_value(steady_detector_result_lags, 50)}/"
                f"{percentile_value(steady_detector_result_lags, 95)} frame(s)",
                flush=True,
            )
        print(
            "Detector reasons (all): " + format_counts(detector_reason_counts),
            flush=True,
        )
        print(
            "Detector reasons (steady): " + format_counts(steady_detector_reason_counts),
            flush=True,
        )
        for label, diagnostics in (
            ("all", edge_diagnostics),
            ("steady", steady_edge_diagnostics),
        ):
            print(
                f"Edge diagnostics ({label}): events={diagnostics.events}, "
                f"side hits: {format_top_counts(diagnostics.sides)}; "
                f"joint hits: {format_top_counts(diagnostics.joints)}",
                flush=True,
            )
            print(
                f"Edge event shape ({label}): landmarks/event: "
                f"{format_top_counts(diagnostics.landmark_counts)}; "
                f"clamping: {format_top_counts(diagnostics.clamped)}; "
                f"confidence p50/p95={percentile_value(diagnostics.confidence_samples, 50)}/"
                f"{percentile_value(diagnostics.confidence_samples, 95)}; "
                f"clamped-only suppressed={diagnostics.clamped_only_suppressed}; "
                f"repeated same-hit={diagnostics.repeated_same_hit_events}, "
                f"max streak={diagnostics.max_same_hit_streak}",
                flush=True,
            )


if __name__ == "__main__":
    main()
