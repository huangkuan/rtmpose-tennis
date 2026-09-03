from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALLOWED_LABELS = {
    "forehand",
    "backhand",
    "forehand_volley",
    "backhand_volley",
    "serve",
    "overhead",
    "recovery",
    "unknown",
}


@dataclass(frozen=True)
class AnnotationSegment:
    start: float
    end: float
    label: str

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


def load_annotations(path: Path) -> tuple[dict[str, Any], tuple[AnnotationSegment, ...]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("annotation file must contain a non-empty 'segments' list")

    segments: list[AnnotationSegment] = []
    previous_end: float | None = None
    for index, raw in enumerate(raw_segments):
        try:
            segment = AnnotationSegment(
                start=float(raw["start"]),
                end=float(raw["end"]),
                label=str(raw["label"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid annotation segment {index}") from error
        if segment.label not in ALLOWED_LABELS:
            raise ValueError(f"unsupported label {segment.label!r} in segment {index}")
        if segment.start < 0 or segment.end <= segment.start:
            raise ValueError(f"invalid time range in segment {index}")
        if previous_end is not None and segment.start < previous_end:
            raise ValueError(f"segment {index} overlaps the previous segment")
        segments.append(segment)
        previous_end = segment.end
    return document, tuple(segments)


def label_records(
    records: Iterable[dict[str, Any]],
    segments: tuple[AnnotationSegment, ...],
) -> Iterable[dict[str, Any]]:
    segment_index = 0
    for record in records:
        timestamp = float(record["time_seconds"])
        while segment_index < len(segments) and timestamp >= segments[segment_index].end:
            segment_index += 1
        label = "unlabeled"
        if segment_index < len(segments) and segments[segment_index].contains(timestamp):
            label = segments[segment_index].label
        yield {**record, "stroke_label": label}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join timestamped pose JSONL records with stroke annotations",
    )
    parser.add_argument("--pose-log", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata, segments = load_annotations(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.pose_log.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        records = (json.loads(line) for line in source if line.strip())
        for record in label_records(records, segments):
            label = record["stroke_label"]
            counts[label] = counts.get(label, 0) + 1
            destination.write(json.dumps(record, separators=(",", ":")) + "\n")

    count_summary = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    print(
        f"Labeled {sum(counts.values())} poses from {metadata.get('video', 'unknown video')}: "
        f"{count_summary}"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
