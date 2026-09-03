# RTMPose tennis player prototype

A first-stage real-time pipeline that reads a live camera or video file, detects people with
the detector bundled with MMPose's `human` alias, estimates body keypoints using
RTMPose-small, and displays the annotated video. When several people are visible, the
largest centrally located person is selected as the tennis player for downstream
processing.

See [optimizations.md](optimizations.md) for the chronological performance
work, measured tradeoffs, and current benchmark configuration.


## Setup

Clone the repository and use Python 3.11. On macOS, the setup script creates an
isolated `.venv311`, installs the known-compatible PyTorch/OpenMMLab stack, and
applies the required workaround for chumpy's legacy build:

```bash
git clone <repository-url>
cd rtmpose_tennis
./scripts/setup_macos.sh
source .venv311/bin/activate
```

Install Python 3.11 first if necessary:

```bash
brew install python@3.11
```

If Homebrew's Python is not exposed as `python3.11`, provide its path:

```bash
RTMPOSE_PYTHON="$(brew --prefix python@3.11)/bin/python3.11" ./scripts/setup_macos.sh
```

The script is safe to rerun and accepts an alternate environment location via
`RTMPOSE_VENV`. For manual or non-macOS installation, use the pinned package
list in `requirements.txt`; MMCV may require a platform-specific build. The
equivalent manual sequence is:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "numpy>=1.24,<2" "torch==2.1.2" "torchvision==0.16.2"
python -m pip install "openmim==0.3.9" "setuptools<81"
python -m pip install --no-build-isolation "chumpy==0.70"
mim install "mmengine>=0.8,<1"
mim install "mmcv>=2.0.1,<2.2.0"
python -m pip install -e .
```

The first run downloads the pretrained detector and RTMPose weights.

## Run

```bash
rtmpose-tennis --camera 0
```

For repeatable debugging with a recorded clip:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m rtmpose_tennis.app \
  --video "./data/input/kid.mp4" \
  --realtime-video \
  --headless \
  --device mps \
  --detector-device cpu \
  --model small \
  --detector-interval-seconds 1.0 \
  --async-detector \
  --crop-margin 0.35 \
  --tracking-alpha 0.5
```

Local videos are intentionally excluded from version control. Place debugging
clips under `data/input/` or use any local path; each collaborator must supply
their own appropriately licensed footage.

`--camera` and `--video` are mutually exclusive. If neither is supplied, camera
index 0 is used. Video processing exits cleanly at the end of the file and runs
as fast as inference permits.

For low-latency live capture, continuously read the camera on a background
thread and retain only its newest frame:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --camera 0 --async-camera \
  --device mps --detector-device cpu \
  --model small --detector-interval-seconds 1.0 --async-detector --crop-margin 0.35 \
  --tracking-alpha 0.35 --preview-scale 0.5
```

To test the same latest-frame behavior repeatably with a recorded clip, pace it
at its encoded frame rate:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --video "./data/input/wnn.mp4" --realtime-video \
  --device mps --detector-device cpu \
  --model small --detector-interval-seconds 1.0 --async-detector --crop-margin 0.35 \
  --tracking-alpha 0.35 --preview-scale 0.5
```

Normal `--video` mode remains synchronous and processes frames in order as fast
as possible. Use it for landmark-quality checks and deterministic debugging;
use `--realtime-video` for camera-like latency and frame-dropping benchmarks.
The overlay and exit summary report displayed-frame age p50/p95, inference wait
p95, sequence lag, and the percentage of frames deliberately replaced by the
latest-frame buffer. The terminal summary additionally reports capture and
processing FPS, counts, maximum latency, and separate detector-refresh and
normal-pose latency. Steady-state results exclude the first three seconds by
default; adjust this with `--metrics-warmup-seconds SECONDS`. Hybrid mode also
reports why each detector refresh occurred: `scheduled_interval`,
`missing_crop`, `missing_pose`, `low_keypoints`, or `crop_edge`. Edge diagnostics
break those events down by boundary side, triggering COCO joint, confidence,
landmarks per event, source-frame clamping, and repeated same-joint/edge streaks.
An edge event does not request immediate detection when every triggering side is
already clamped to the source image, because detection cannot reveal pixels
beyond that boundary; these suppressed events are counted separately.

`--async-detector` moves periodic RTMDet refreshes to a single-flight
background worker. Crop-pose inference continues while detection is running;
new requests are rejected until the outstanding result is consumed, so stale
detector jobs cannot accumulate. The terminal reports submitted/completed jobs,
request-to-result latency, and result lag in source frames. Omit the switch to
retain the synchronous detector for A/B comparisons.

For the screenless production path, use `--headless` to bypass preview resize,
drawing, overlays, window creation, and display calls while retaining all
performance measurements:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --video "./data/input/pro.mov" --realtime-video \
  --device mps --detector-device cpu --model small \
  --detector-interval-seconds 1.0 --async-detector \
  --crop-margin 0.35 --tracking-alpha 0.5 --headless
```

Headless mode prints a compact live status every two seconds and the normal
detailed summary at shutdown. Change the live reporting cadence with
`--status-interval SECONDS`. `pose output` counts successful pose samples, not
preview frames or inference attempts. Stop a headless camera session with
Ctrl+C; shutdown remains clean and prints the final metrics.

`--detector-interval-seconds 1.0` is recommended for live and real-time
simulation modes. It keeps routine RTMDet refreshes at a stable wall-clock rate
when camera FPS or processing throughput changes. Recovery caused by a missing
crop, low-confidence pose, or actionable crop edge remains immediate. The
legacy `--detector-interval N` frame-based schedule remains available for
repeatable comparisons; the two interval options are mutually exclusive.

Automatic handedness inference is enabled by default with `--handedness auto`.
It accumulates conservative evidence from asymmetric high-speed wrist motion.
The result remains `unknown` until sufficient evidence is available, then
reports `left` or `right`, confidence, and whether the decision is locked. For
known players or controlled tests, bypass inference with `--handedness left`
or `--handedness right`. The terminal summary reports left/right evidence,
motion observation counts, and estimator overhead p50/p95.
In preview mode, the dominant-side wrist landmark is green; other landmarks
remain red. No wrist is highlighted while automatic handedness is `unknown`.

The gesture-recognition hand-off is a timestamped, body-normalized pose window.
It retains the latest two seconds by default; change this with
`--pose-buffer-seconds`. To inspect or label the resulting trajectory, write
newline-delimited JSON with:

```bash
rtmpose-tennis \
  --video "./data/input/pro.mov" --realtime-video --headless \
  --handedness right --pose-log "./data/output/pro-pose.jsonl" \
  --device mps --detector-device cpu --model small \
  --detector-interval-seconds 1.0 --async-detector
```

Each JSONL record contains the source sequence and timestamp, 17 normalized
COCO keypoints and confidence scores, wrist velocity and acceleration, elbow
angle, arm extension, wrist separation, torso angles, and midline-crossing
state. Left-handed poses are reflected and relabelled into the same canonical
right-handed representation. Use a manual handedness value while preparing
labelled data because automatic handedness is not yet validated sufficiently.
`data/output/` is excluded from Git.

Versioned stroke intervals live in `data/annotations/`. Join an annotation file
to its pose log before classifier development with:

```bash
python -m rtmpose_tennis.annotations \
  --pose-log ./data/output/pro-pose.jsonl \
  --annotations ./data/annotations/pro.json \
  --output ./data/output/pro-labeled.jsonl
```

The output adds `stroke_label` to every pose record while leaving source logs
unchanged. Samples outside annotated intervals are marked `unlabeled`.
Annotations currently cover right- and left-handed forehands, backhands,
volleys, a serve, an overhead, and recovery periods.

Run the first event-level recognition benchmark across every matched annotation
and pose log with:

```bash
python -m rtmpose_tennis.stroke_analysis \
  --pose-dir ./data/output \
  --annotations ./data/annotations \
  --report ./data/output/stroke-baseline.json
```

The default benchmark uses a two-stage classifier. Aggregate features first
separate groundstrokes from `other`; confidence-repaired, smoothed, phase-aware
features then separate forehands from backhands. Every test video remains
untouched while neighbor counts are selected with inner leave-one-video-out
validation. This prevents neighboring frames and test-video settings from
leaking into training. Explicit `unknown` intervals are excluded.

Use `--method aggregate` to reproduce the original baseline or `--method
smoothed` to evaluate the temporal features alone. All methods assume known
event boundaries; automatic stroke segmentation remains a separate next stage.

Evaluate automatic stroke-boundary proposals from the continuous pose logs
with:

```bash
python -m rtmpose_tennis.stroke_segmentation \
  --pose-dir ./data/output \
  --annotations ./data/annotations \
  --report ./data/output/stroke-segmentation.json
```

The detector repairs and smooths the dominant-wrist trajectory, finds separated
motion peaks, then searches backward for motion onset and forward for
deceleration. It targets all labelled stroke types; `recovery` is negative
footage and `unknown` or unlabeled footage is excluded. Parameters are selected
using only the other videos for every held-out-video fold. Use `--boundary-mode
fixed` for the original fixed-window comparison or `--selection fixed` to test
explicit parameter values. Reported recognition accuracy and segmentation F1
are separate metrics; they are not yet an end-to-end live score.

Connect automatic proposals to the recognizer and evaluate the complete offline
pipeline with:

```bash
python -m rtmpose_tennis.stroke_pipeline \
  --pose-dir ./data/output \
  --annotations ./data/annotations \
  --report ./data/output/stroke-pipeline.json
```

Both segmentation and classifier settings are selected from training videos
only before each held-out video is scored. End-to-end precision requires both a
matched interval and the correct class. `false alerts/minute` counts only
unmatched proposals; detected strokes assigned the wrong class are reported as
classification errors instead. Training-video classifier examples use the same
automatically proposed window shape as the held-out video, avoiding a mismatch
between manually bounded training events and adaptive test proposals.

The default temporal representation locates five cumulative dominant-wrist
motion anchors spanning preparation through follow-through. It compares joint
position, direction, extension, and timing at those phases instead of relying
on one maximum-speed frame. Use `--classifier-features smoothed` to reproduce
the earlier uniformly sampled temporal baseline.

The JSON report also contains a per-proposal `audit` with detection IoU,
groundstroke-gate and forehand/backhand decisions, nearest-class distances,
pose confidence, phase-anchor timestamps, and the stage responsible for each
error. A selective result compares the motion-phase and smoothed temporal
classifiers and accepts a stroke only when they agree and exceed a distance
margin chosen from training videos. `selective_classification` reports accepted
accuracy, coverage, abstentions, and confident false alerts separately from the
unfiltered end-to-end benchmark.

Peak-aligned classification is available as an offline experiment:

```bash
python -m rtmpose_tennis.stroke_pipeline \
  --pose-dir ./data/output \
  --annotations ./data/annotations \
  --classifier-alignment peak \
  --report ./data/output/stroke-pipeline-peak.json
```

This retains adaptive boundaries for interval matching and reporting, but
extracts temporal classifier features from a consistent window around the
proposal's dominant-wrist motion peak. Each held-video fold selects the window
before/after the peak using only its training videos. Interval alignment remains
the default because it currently performs better on the five-video benchmark.

Useful options:

```bash
rtmpose-tennis --camera 1 --device cuda:0 --width 1280 --height 720
rtmpose-tennis --video ./samples/forehand.mp4 --device cuda:0
rtmpose-tennis --device cpu --score-threshold 0.4
rtmpose-tennis --camera 0 --handedness right
rtmpose-tennis --video ./samples/forehand.mp4 --pose-log ./data/output/forehand.jsonl
```

## Performance tuning

Start by processing every second frame. The last detected pose is drawn on the
intermediate frame, preserving a responsive video loop while halving inference
work:

```bash
rtmpose-tennis --video ./samples/forehand.mp4 --device cpu --infer-every 2
```

For a controlled view containing one prominent player, bypass the RTMDet person
detector and apply RTMPose to the full frame:

```bash
rtmpose-tennis --video ./samples/forehand.mp4 --device cpu --whole-image
```

The two options can be combined for maximum first-stage speed:

```bash
rtmpose-tennis --video ./samples/forehand.mp4 --device cpu --whole-image --infer-every 2
```

`--whole-image` trades multi-person selection and a tight detector bounding box
for speed, so compare landmark quality before making it the default. The overlay
reports model FPS separately from display-loop FPS for an honest comparison.

RTMPose-small is the application default, balancing landmark stability and CPU
speed. Its pretrained checkpoint is downloaded and cached on first use. Three
short model presets make controlled comparisons easy:

```bash
rtmpose-tennis --video ./samples/forehand.mp4 --device cpu --whole-image --model tiny
rtmpose-tennis --video ./samples/forehand.mp4 --device cpu --whole-image --model small
rtmpose-tennis --video ./samples/forehand.mp4 --device cpu --whole-image --model medium
```

Flip-test augmentation is disabled at runtime because it adds a mirrored second
forward pass intended for offline evaluation, not real-time inference.

### Multiple-player hybrid mode

When an opponent or bystander is visible, do not use `--whole-image`. Hybrid
mode detects all people periodically, selects the large central foreground
player, and runs RTMPose on that player's retained crop between detections:

```bash
rtmpose-tennis --video ./samples/match.mp4 --device cpu --model small --detector-interval 10
```

Start with an interval of 10 frames. Lower values recover faster from rapid
movement or camera motion but run the detector more often; higher values are
faster but can lose a player who exits the retained crop. `--crop-margin 0.2`
controls the padding around the detected player. Hybrid mode cannot be combined
with `--whole-image`.

On Apple Silicon, MMCV non-maximum suppression does not support MPS. Hybrid mode
therefore runs periodic RTMDet refreshes on CPU by default while running the
crop-only RTMPose pass on MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --video ./samples/match.mp4 --device mps --model tiny \
  --detector-interval 4 --crop-margin 0.35
```

Use `--detector-device cpu` to state the split explicitly. The startup message
prints the selected pose and detector devices.

The on-screen performance overlay reports rolling average latency for frame
decoding/capture, periodic detector-only refreshes, crop-only pose inference,
and drawing/window display. Hybrid mode uses RTMDet-tiny by default and no
longer performs a redundant CPU pose pass during detector refreshes. Override
the detector with `--detector-model MODEL_NAME` if needed.

Hybrid mode also uses the confident pose landmarks as a near-zero-cost tracker:
the crop centre and size follow the player smoothly on every frame. RTMDet is
requested early when too few landmarks remain confident or landmarks approach
the crop edge; `--detector-interval` is now the maximum time between corrective
detections rather than the only way the crop moves. A practical starting point
for live tennis is:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --camera 0 --device mps --detector-device cpu --model small \
  --detector-interval 30 --crop-margin 0.35 --tracking-alpha 0.35
```

To reduce synchronous visualization overhead without changing inference input
or stored keypoint coordinates, render a half-resolution preview:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --video ./data/input/wnn.mp4 --device mps --detector-device cpu \
  --model small --detector-interval 30 --crop-margin 0.35 \
  --tracking-alpha 0.35 --preview-scale 0.5
```

The `draw+show` metric includes preview resizing, so it reports the true net
effect. `--preview-scale` affects visualization only; RTMPose and all downstream
gesture data continue to use the original-resolution frame coordinates.

Press `q` or Escape to stop. On macOS, grant camera access to the terminal or
IDE that launches Python. If `mps` causes an unsupported-operation error, omit
`--device` or use `--device cpu`.

## Architecture and next step

`select_player()` exposes `PlayerPose.keypoints` and `PlayerPose.scores` on every
frame. The next phase should normalize coordinates relative to hips/torso, buffer
roughly 1–3 seconds of poses, track the same athlete over time, and classify the
sequence (serve, forehand, backhand, ready position). Keep feedback as a separate
layer driven by joint angles and phase timing so it can explain a classification.

For robust court use, the immediate upgrades are persistent person tracking,
optional side/rear camera calibration, and a recording mode that saves
timestamped keypoint sequences for labelling.
