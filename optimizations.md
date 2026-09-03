# Performance optimizations

This document records the optimization work performed on the RTMPose tennis
prototype, including measured results on a 16 GB Apple M1 MacBook Pro. Values
are approximate rolling averages observed on the same representative tennis
video; they should be treated as comparative measurements rather than general
RTMPose benchmarks.

## Goals and measurement

The original goal was real-time body-pose extraction suitable for later tennis
gesture recognition and feedback. A useful first target was 25–30 completed
poses per second while retaining stable wrists, ankles, and torso landmarks.

The overlay reports:

- **model FPS**: inference throughput, excluding decoding and visualization.
- **output FPS**: synchronous end-to-end processed/displayed frames per second.
- **decode**: OpenCV camera capture or video decoding latency.
- **detect**: detector-only RTMDet refresh latency.
- **crop pose**: RTMPose inference on the retained player crop.
- **draw+show**: preview resizing, drawing, text, and OpenCV window display.

Output FPS is the closest current approximation to delivered pose samples per
second because the application is still a synchronous pipeline.

## Optimization history

### 1. Initial RTMPose-medium and RTMDet-medium pipeline

The first implementation used MMPose's `human` alias. It ran RTMDet-medium and
RTMPose-medium on each full frame, selected the largest central person, and drew
the resulting 17 COCO body landmarks.

Observed result:

```text
Model throughput: approximately 1.6 FPS
```

This established correctness but was far below real-time performance.

### 2. Video input through the live-frame pipeline

Video-file input was added alongside camera input. Both sources now feed the
same frame-by-frame inference, player selection, landmark drawing, and timing
pipeline. Recorded video made performance and quality comparisons repeatable.

### 3. Detector bypass with whole-image mode

`--whole-image` was added to bypass RTMDet and treat the full frame as one
person crop.

Observed result with RTMPose-medium:

```text
Model throughput: 1.6 -> 6.2 FPS
Speedup: approximately 4x
```

This proved that person detection was a major bottleneck. It worked when one
large player dominated the frame, but failed when an opponent was also visible:
top-down RTMPose expects one person per crop and cannot reliably separate two
people inside a whole-frame crop.

`--whole-image` remains useful for controlled single-person scenes but is not
the default solution for tennis matches.

### 4. Configurable inference skipping

`--infer-every N` was added to run inference every Nth input frame and reuse the
last pose on intermediate frames. This can improve preview responsiveness, but
it does not increase pose samples per second and therefore is not preferred for
gesture recognition.

### 5. Smaller RTMPose models and flip-test removal

Model presets were introduced:

```text
tiny   -> RTMPose-t
small  -> RTMPose-s (current default)
medium -> MMPose human alias / RTMPose-m
```

RTMPose-tiny plus whole-image mode reached approximately 44 model FPS, but its
landmarks were unstable because the player occupied a small fraction of the
full-frame model input. RTMPose-small was selected as the quality/speed balance.

Flip-test augmentation was disabled. Official evaluation configurations can
perform a second mirrored forward pass for accuracy; this is inappropriate for
latency-sensitive live inference.

### 6. Initial hybrid detection and crop inference

A hybrid mode was added with `--detector-interval`:

```text
Periodically detect all people
-> select the large central foreground player
-> retain an expanded crop
-> run detector-free RTMPose on the crop between refreshes
```

This restored multi-player correctness while avoiding detection on every frame.
The first version still used MMPose's combined detector-plus-pose inferencer for
refresh frames, which loaded a duplicate pose model and performed a redundant
CPU pose pass.

Observed result with RTMPose-small and interval 8:

```text
Model throughput: approximately 13.4 FPS
Output throughput: approximately 10.4 FPS
```

### 7. Apple MPS pose inference with CPU detection

Running the entire OpenMMLab pipeline on `mps` failed because MMCV non-maximum
suppression has no MPS implementation:

```text
RuntimeError: nms_impl: implementation for device mps:0 not found
```

Hybrid mode was split across devices:

```text
Periodic RTMDet and NMS -> CPU
Frequent crop RTMPose   -> Apple MPS GPU
Capture and display     -> CPU
```

Observed result with RTMPose-small and detector interval 8:

```text
Model throughput: approximately 16.8 FPS
Output throughput: approximately 12.1 FPS
```

This retained compatibility while accelerating the frequent pose pass.

### 8. Separate stage timing

Rolling measurements were added for decoding, detector refreshes, crop pose,
and drawing/display. The first measurements exposed the real bottleneck:

```text
decode:        approximately   1.4 ms
detect+pose:   approximately 445-471 ms
crop pose:     approximately  17.5 ms
draw+show:     approximately  15.9 ms
```

At interval 8, the combined refresh cost contributed about 59 ms per frame and
dominated the pipeline.

### 9. Detector-only RTMDet-tiny refreshes

The combined MMPose refresh pipeline was replaced with MMDetection's
RTMDet-tiny `DetInferencer`. Refreshes now perform detection only, filter COCO
person class 0, and rank candidates by confidence, area, and centrality. The
selected box is followed by the normal MPS crop pose pass.

This removed the redundant CPU pose pass and duplicate CPU pose model.

Observed changes:

```text
Detector refresh: 445-471 -> approximately 200 ms
Model throughput: 16-17   -> approximately 23-28 FPS
Output throughput: 11-12 -> approximately 17-18 FPS
```

Detector refresh latency improved by roughly 56%.

### 10. Pose-guided adaptive crop tracking

The formerly fixed crop was changed into a lightweight pose-guided tracker.
Confident full-resolution keypoints now update a smoothed crop centre and size
on every frame. RTMDet is requested early when:

- too few landmarks remain confident;
- a landmark approaches the crop boundary;
- no valid player crop exists; or
- the maximum detector interval expires.

The crop-size change per frame is constrained to avoid sudden shrinking or
growth. `--detector-interval` is now a maximum re-anchor interval rather than
the only time the crop can move.

Using a maximum interval of 30 frames produced:

```text
Model throughput: approximately 31-43 FPS
Output throughput: approximately 23-25.4 FPS
Detector latency: approximately 203 ms per refresh
Crop pose: approximately 17 ms
Draw+show: approximately 15.8 ms
```

The crop successfully followed substantial lateral movement and scale changes
while remaining locked to the foreground player.

### 11. Reduced-resolution preview

`--preview-scale` was added. Detection, pose inference, tracking, and stored
keypoints remain at original resolution; only a visualization-only frame and
copy of the pose are scaled. At `--preview-scale 0.5`, the preview contains 25%
of the original pixels.

Observed result:

```text
draw+show:       15.8 -> approximately 12.6 ms
Output rate:     23-25.4 -> approximately 27 FPS
Model rate:      approximately 40-42 FPS
```

The preview is softer when enlarged by the window, but inference and downstream
pose coordinates are unaffected.

## Current synchronous baseline

Run the current optimized video pipeline with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m rtmpose_tennis.app \
  --video "./data/input/wnn.mp4" \
  --device mps \
  --detector-device cpu \
  --model small \
  --detector-interval 30 \
  --crop-margin 0.35 \
  --tracking-alpha 0.35 \
  --preview-scale 0.5
```

For a live camera, replace the video argument with:

```bash
--camera 0
```

Current representative frame budget:

```text
Amortized detector: 203 / 30 = approximately  6.8 ms/frame
Crop pose:                          approximately 17.2 ms/frame
Preview draw+show:                  approximately 12.6 ms/frame
Decode:                             approximately  1.5 ms/frame
                                                -------
Total:                              approximately 38.1 ms/frame
Expected output rate:               approximately 26-27 FPS
```

From the original 1.6 FPS prototype to the current approximately 27 FPS output,
end-to-end throughput improved by about 17x while adding multi-player handling,
adaptive crop tracking, device splitting, and detailed profiling.

## Current tradeoffs and limitations

- Latest-frame capture can run asynchronously, but detector, pose, and display
  work remain serialized in the processing loop.
- A detector refresh still causes a roughly 230-260 ms latency spike.
- `model FPS` excludes decoding and display; `output FPS` is the meaningful
  current end-to-end delivery rate.
- Pose landmarks can be wrong during motion blur, limb crossing, or occlusion.
- Pose-guided tracking depends on sufficiently confident landmarks and uses
  RTMDet for recovery.
- A 50% preview is visibly softer, although raw frames and pose data are not.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is still useful for unsupported PyTorch MPS
  operations.

## Asynchronous latest-frame input

Live-camera and real-time simulated-video modes now support asynchronous
latest-frame processing:

```text
Capture thread -> always retain newest camera frame
Inference loop -> process newest available frame; discard stale frames
Display        -> show newest completed pose
```

Enable it for a camera with `--async-camera`. Enable a repeatable camera-like
test from a video with `--video CLIP --realtime-video`; frames are released at
the file's encoded FPS before entering the same one-frame latest buffer. Plain
`--video` remains synchronous for deterministic regression and pose-quality
comparisons.

The overlay and terminal summary expose the measurements needed for an A/B
comparison:

- displayed-frame age p50 and p95;
- frame wait before inference p95;
- lag behind the newest captured sequence;
- captured, processed, inferred, and stale-frame counts;
- capture/output FPS and stale-frame percentage;
- maximum displayed-frame age and maximum sequence lag;
- normal-pose versus detector-refresh frame age;
- steady-state measurements after a configurable warm-up period;
- full-session and steady-state detector refresh reasons, separating scheduled
  correction from missing crops/poses, low-confidence landmarks, and crop-edge
  proximity.

The default warm-up exclusion is three seconds and can be changed with
`--metrics-warmup-seconds`. Full-session results are retained alongside the
steady-state report so startup cost remains visible.

The objective is bounded frame age rather than zero dropped frames. A real-time
pipeline should discard obsolete frames instead of allowing latency to grow.

### Crop correction before redetection

Redetection diagnostics showed that fast videos could request RTMDet on nearly
every crop-edge event even though pose tracking corrected the crop immediately
afterward. The hybrid loop now updates the crop from the current pose before
testing its edges. Missing poses and insufficient confident keypoints still
request immediate recovery, while an edge-triggered refresh occurs only when
the corrected crop remains unsafe.

Detailed edge diagnostics identify the boundary side and COCO joints involved,
record triggering confidence and landmark count, flag edges that cannot expand
because the crop is clamped to the source frame, and measure consecutive events
that repeat at least one identical joint/edge hit. Both complete-session and
post-warm-up summaries are emitted for controlled tracker-policy experiments.

When every landmark edge hit corresponds to a crop side already clamped to the
source image, the event is now recorded but does not request immediate RTMDet.
Periodic detection and missing/low-confidence pose recovery remain active. This
prevents futile repeated detections when a correctly tracked athlete is close
to the camera frame boundary.

### Clamped-edge diagnosis and controlled experiments

Four videos were first run with identical baseline settings:

```text
model=small, detector interval=30, crop margin=0.35,
tracking alpha=0.35, preview scale=0.5
```

Detector-reason counters showed a progression from stable tracking to repeated
edge-driven recovery:

| Video | Scheduled | Crop edge | Low keypoints | Steady detector rate | Steady output |
|---|---:|---:|---:|---:|---:|
| `wnn.mp4` | 22 | 6 | 0 | 0.98/s | 24.9 FPS |
| `female.mp4` | 15 | 11 | 0 | 0.95/s | 24.1 FPS |
| `backview.mp4` | 19 | 60 | 5 | 1.76/s | 18.9 FPS |
| `pro.mov` | 0 | 23 | 0 | 3.03/s | 8.8 FPS |

RTMPose latency stayed near 40-59 ms while detector frames remained roughly
230-260 ms. Overall throughput therefore followed detector frequency rather
than pose speed.

Three controlled changes were tested on `pro.mov`:

1. increasing crop margin from 0.35 to 0.45;
2. increasing tracking alpha from 0.35 to 0.50; and
3. updating the crop before evaluating its edges.

None materially reduced the number of crop-edge events. Detailed diagnostics
then identified the actual condition: all 24 steady edge events occurred at a
left crop edge already clamped to the source frame. The left ankle produced 21
of 28 landmark/edge hits, with confidence p50/p95 of 0.76/0.98 and repeated
streaks up to five frames. RTMDet could not reveal pixels outside the video, so
these refreshes were futile.

### Five-video validation

After clamped-only edge suppression, the difficult videos improved strongly
while the healthy videos did not regress:

| Video | Source | Steady FPS before | Steady FPS after | Drop after | Refreshes before -> after | Suppressed edges |
|---|---:|---:|---:|---:|---:|---:|
| `wnn.mp4` | 30 FPS | 24.9 | 25.3 | 17.5% | 28 -> 25 | 0 |
| `female.mp4` | 30 FPS | 24.1 | 25.2 | 17.2% | 26 -> 23 | 0 |
| `backview.mp4` | 30 FPS | 18.9 | 23.8 | 21.4% | 84 -> 53 | 94 |
| `pro.mov` | 58.6 FPS | 11.6 | 25.1 | 59.4% | 24 -> 9 | 80 |
| `kid.mp4` | 30 FPS | not measured | 24.5 | 20.2% | not measured -> 19 | 44 |

`pro.mov` more than doubled steady throughput, from 11.6 to 25.1 FPS, while
normal-pose p95 remained 47.0 ms. Its 84 steady edge events divided exactly
into 80 suppressed clamped-only events and four actionable detector requests.

`backview.mp4` improved from 18.9 to 23.8 FPS and reduced frame replacement
from 36.7% to 21.4%. Of 105 steady edge events, 94 were source-boundary-clamped
and 11 remained actionable. Its missing/low-confidence recovery behavior was
not disabled.

`kid.mp4` contained a 43-frame repeated right-edge streak involving multiple
high-confidence body landmarks. Forty-four clamped-only events were suppressed,
yet three internal crop-edge events still requested recovery and scheduled
detection continued at the expected rate.

The 59.4% replacement rate for `pro.mov` is now primarily the expected mismatch
between a 58.6 FPS input and a roughly 25 FPS processing path, not pathological
detector repetition. Latest-frame replacement keeps normal pose age near
40-47 ms instead of allowing latency to accumulate.

### 15. Single-flight background RTMDet worker

Periodic RTMDet refreshes can now run concurrently with crop-pose inference by
passing `--async-detector`. The main loop submits a copy of the newest frame and
continues tracking the current crop with RTMPose while the CPU detector works.
Only one request may be pending or running, and another is not accepted until
the completed result has been consumed. This prevents an unbounded detector
queue from turning throughput into stale feedback.

The synchronous path remains available by omitting the switch, allowing direct
A/B testing. New exit metrics report background jobs submitted and completed,
request-to-result latency p50/p95, and result staleness p50/p95 in source
frames. These measurements tell us whether overlapping the roughly 230-260 ms
detector refresh improves output FPS without applying detections that are too
old to re-anchor a fast-moving player reliably.

Example real-time benchmark:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --video "./data/input/pro.mov" --realtime-video \
  --device mps --detector-device cpu --model small \
  --detector-interval 30 --async-detector \
  --crop-margin 0.35 --tracking-alpha 0.5 --preview-scale 0.5
```

### Background-detector five-video validation

The same five-video suite was rerun with `--async-detector`. The synchronous
results below are the post-clamped-edge-suppression baseline, so the comparison
isolates the benefit of overlapping RTMDet with crop-pose inference.

| Video | Source FPS | Sync steady FPS | Async steady FPS | Gain | Sync drop | Async drop |
|---|---:|---:|---:|---:|---:|---:|
| `pro.mov` | 58.6 | 25.1 | 31.3 | +24.7% | 59.4% | 48.1% |
| `wnn.mp4` | 30.0 | 25.3 | 30.0 | +18.6% | 17.5% | 1.4% |
| `female.mp4` | 30.0 | 25.2 | 29.7 | +17.9% | 17.2% | 2.1% |
| `backview.mp4` | 30.0 | 23.8 | 29.8 | +25.2% | 21.4% | 1.4% |
| `kid.mp4` | 30.0 | 24.5 | 29.3 | +19.6% | 20.2% | 4.0% |

Every video gained approximately 18-25% steady throughput. All 30 FPS sources
reached 29.3-30.0 FPS, while the 58.6 FPS source improved to 31.3 FPS. Frame
replacement on the 30 FPS clips fell from 17-21% to 1-4%. The remaining 48.1%
replacement on `pro.mov` is expected because its source rate remains almost
twice the processing rate.

Freshness also improved in the long tail:

| Video | Sync age p95 | Async age p95 | Sync max age | Async max age | Async max lag |
|---|---:|---:|---:|---:|---:|
| `pro.mov` | 139.4 ms | 51.3 ms | 241.7 ms | 75.0 ms | 4 frames |
| `wnn.mp4` | 59.2 ms | 51.9 ms | 282.7 ms | 87.7 ms | 2 frames |
| `female.mp4` | 54.9 ms | 56.7 ms | 255.8 ms | 163.3 ms | 4 frames |
| `backview.mp4` | 63.2 ms | 56.9 ms | 304.0 ms | 105.6 ms | 3 frames |
| `kid.mp4` | 43.6 ms | 60.6 ms | 249.7 ms | 80.8 ms | 2 frames |

`kid.mp4` was the only clip with a meaningful p95 regression, from 43.6 to
60.6 ms, probably reflecting CPU, memory-bandwidth, or scheduling contention
while RTMDet and RTMPose overlap. Its median remained essentially unchanged,
however, and its maximum age and source lag improved substantially. This is a
tradeoff to watch on slower hardware rather than a reason to revert the worker.

Background detector timing and result staleness were:

| Video | Detector latency p50/p95 | Result lag p50/p95 |
|---|---:|---:|
| `pro.mov` | 225.9/255.9 ms | 13/15 frames |
| `wnn.mp4` | 226.0/259.9 ms | 6/7 frames |
| `female.mp4` | 258.4/350.9 ms | 7/10 frames |
| `backview.mp4` | 254.4/267.0 ms | 7/8 frames |
| `kid.mp4` | 263.5/269.2 ms | 7.5/8 frames |

The larger frame lag on `pro.mov` represents approximately the same elapsed
detector time at its higher source FPS. All submitted requests completed on the
first four clips. On `kid.mp4`, 25 of 26 completed results were consumed; the
last request was still outstanding when the video ended and therefore had no
later frame on which its result could be applied.

Recovery behavior remained controlled. `backview.mp4`, the most difficult
30 FPS case, used seven crop-edge and six low-keypoint refreshes while reaching
29.8 FPS. `kid.mp4` retained its long source-boundary streak, but 48 of 52 edge
events were clamped-only and only two crop-edge refreshes were submitted.
Single-flight submission coalesced repeated edge observations instead of
building a queue of stale detector work.

The background worker is therefore retained. Motion compensation for stale
detections is deferred unless visual testing reveals a crop jump when a result
arrives; the throughput, freshness, and recovery counters do not currently
show a need for that added complexity. The next optimization should be chosen
from a fresh timing profile now that synchronous RTMDet stalls are removed.

### 16. Headless pose-output mode

The final coaching product will not show a video preview: pose analysis will
ultimately produce audible instructions. The `--headless` path therefore
removes preview resizing, skeleton and text drawing, window creation,
`imshow`, and `waitKey` from the critical path while leaving capture,
detection, pose inference, tracking, diagnostics, and metrics intact.

Headless sessions report successful pose-output FPS separately from processed
frame rate and inference-attempt count. A compact status line is printed every
two seconds by default, including pose-output FPS, processed FPS, rolling frame
age p50/p95, crop-pose latency, and dropped-frame percentage. The interval is
configurable with `--status-interval`. End-of-video and Ctrl+C shutdown both
produce the complete session summary, including background-detector metrics.

This mode establishes the production-relevant compute ceiling without spending
roughly 13 ms per frame on debugging visuals. The graphical path remains
available for landmark-quality and crop-stability inspection.

#### Headless validation

`backview.mp4` demonstrated the latency benefit on a 30 FPS source. Throughput
was already source-limited, moving only from 29.8 to 30.0 FPS, but steady frame
age p50/p95 fell from 31.1/56.9 ms to 17.7/32.8 ms. Inference-wait p95 fell
from 23.2 to 0.3 ms, dropped frames declined from 1.4% to 0.9%, and every one of
the 1,426 steady processed frames produced a valid pose.

`pro.mov` exposed the actual compute gain because its source rate is 58.6 FPS:

| Metric | Preview | Headless | Change |
|---|---:|---:|---:|
| Steady pose/processed FPS | 31.3 | 53.7 | +71.6% |
| Dropped frames | 48.1% | 12.3% | -35.8 points |
| Steady age p50 | 40.6 ms | 21.5 ms | -47.0% |
| Steady age p95 | 51.3 ms | 36.4 ms | -29.0% |
| Steady maximum age | 75.0 ms | 58.0 ms | -22.7% |

The headless crop-pose model ran at 16.0/25.5 ms p50/p95. After startup crop
acquisition, all 406 steady processed frames produced valid poses. This
confirmed that preview rendering—not RTMPose—had become the dominant serial
cost after background detection was introduced.

### 17. Time-based scheduled redetection

Headless `pro.mov` reached 53.7 successful pose outputs per second, exposing a
rate-dependent behavior in `--detector-interval 30`: the same setting scheduled
RTMDet about once per second at 30 processed FPS but about every 0.56 seconds at
53.7 FPS. Faster processing therefore caused more detector contention even
though tracking had not become less reliable.

`--detector-interval-seconds` now provides a wall-clock schedule independent of
source FPS, dropped frames, preview mode, and device speed. A value of `1.0`
preserves the original intended refresh cadence on 30 FPS input. Any successful
submission, whether scheduled or recovery-driven, resets the elapsed-time
schedule. Missing-crop, missing-pose, low-keypoint, and actionable crop-edge
recovery still request RTMDet immediately.

The original frame-based option remains supported for compatibility and
controlled comparisons, but it cannot be combined with the seconds option.
Production-oriented benchmarks should now use:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 rtmpose-tennis \
  --video "./data/input/pro.mov" --realtime-video --headless \
  --device mps --detector-device cpu --model small \
  --detector-interval-seconds 1.0 --async-detector \
  --crop-margin 0.35 --tracking-alpha 0.5
```

#### Time-based validation

On `pro.mov`, replacing the 30-frame schedule with a one-second schedule
reduced steady detector submissions from 15 to 10 and scheduled reasons from
11 to five. Five additional crop-edge recoveries reset the elapsed-time clock,
so fewer than one scheduled request per wall-clock second was necessary.

| Metric | 30-frame schedule | 1-second schedule | Change |
|---|---:|---:|---:|
| Steady pose-output FPS | 53.7 | 55.0 | +2.4% |
| Dropped frames | 12.3% | 10.4% | -1.9 points |
| Steady age p50 | 21.5 ms | 16.4 ms | -23.7% |
| Steady age p95 | 36.4 ms | 36.1 ms | stable |
| Pose latency p95 | 25.5 ms | 23.7 ms | -7.1% |
| Steady detector submissions | 15 | 10 | -33.3% |

The resulting 55.0 successful pose samples per second represent about 94% of
the 58.6 FPS source rate. Fourteen detector jobs were submitted during the full
session and 13 results were consumed; the final job completed at end-of-file
without a later frame on which to apply it.

`kid.mp4` confirmed that the time schedule preserves the intended cadence on a
30 FPS source. It reached 29.8 steady pose-output FPS, dropped 2.6% of frames,
and produced valid poses on all 635 steady frames. Steady detector submissions
remained 23, comprising 21 scheduled and two crop-edge requests—nearly
identical to the former 30-frame behavior. Compared with its graphical run,
frame-age p50/p95 improved from 31.7/60.6 ms to 17.6/44.6 ms.

Its prolonged source-boundary condition also remained controlled: 53 of 59
edge events were clamped-only and suppressed, despite a repeated streak of 57
frames. This validates time-based scheduling across both 30 FPS and roughly
60 FPS inputs without weakening data-driven recovery.

### 18. Temporal handedness inference

Gesture features need a stable definition of the dominant side so left-handed
poses can be canonicalized before stroke classification. `--handedness auto`
now maintains a conservative temporal evidence state rather than classifying a
single ambiguous frame. Manual `left` and `right` modes provide deterministic
overrides for known players and evaluation.

The estimator compares shoulder-width-normalized left and right wrist
velocities. It votes only during sufficiently fast, asymmetric motion and is
rate-limited to avoid counting every frame of the same movement. Ambiguous
motion does not vote, limiting the influence of toss-arm movement and
two-handed backhands.

Automatic output begins as `unknown`. A provisional side requires at least
four weighted evidence units and a 67% winning share. The decision locks only
after at least ten evidence units with an 82% share. Runtime and final metrics
report the label, confidence, lock state, left/right evidence, motion
observation counts, and estimator overhead p50/p95. Synthetic unit and
headless integration tests measured overhead below the displayed 0.1 ms
resolution, so no meaningful throughput impact is expected.

Real-video validation should check known left- and right-handed players across
forehands, serves, one- and two-handed backhands, motion blur, and rear/side
views. Accuracy and time-to-decision are now more important than further FPS
optimization; uncertain sessions should remain `unknown` rather than forcing
an incorrect coaching orientation.

### 19. Timestamped normalized pose pipeline

The first gesture-recognition foundation is now separated from video rendering
and model inference. Every successful inferred pose can produce a
`TemporalPoseFrame` containing the source sequence, capture timestamp,
normalized keypoints, confidence scores, and derived `PoseFeatures`. A
time-bounded `TemporalPoseBuffer` retains the latest two seconds by default,
independent of whether the source is running at 30 or roughly 60 FPS.

Normalization uses hip center as the origin, shoulder width as scale, the
left-to-right shoulder line as the local horizontal axis, and its perpendicular
directed toward the hips as the local vertical axis. This removes image
translation, scale, and in-plane lean. When handedness is known to be left,
coordinates are
reflected and bilateral COCO joints are swapped so the dominant side occupies
the canonical right-side indices. The buffer clears when canonical handedness
changes, preventing mixed coordinate conventions in one temporal window.

Current per-frame features include:

- left, right, and dominant wrist velocity;
- dominant wrist acceleration and local position;
- dominant elbow angle and arm extension;
- wrist separation and dominant-wrist midline crossing;
- shoulder angle, hip angle, and their wrapped difference;
- torso scale and mean landmark confidence.

The optional `--pose-log PATH` writer records this interface as JSONL for
offline plots, stroke segmentation, and labelled-dataset preparation. Logging
is opt-in so production inference does not pay JSON serialization or disk-I/O
cost. The terminal reports generated samples, current window size, feature
extraction overhead, logged count, and logging overhead.

Synthetic tests verify translation/scale body normalization, equivalence of
mirrored left-handed and canonical right-handed poses, normalized velocity,
time-window pruning, JSON schema, and full headless-loop integration. The next
step is to log known right-handed forehand and backhand clips, visualize the
dominant wrist trajectory and torso features, and define a generic swing
candidate segment before assigning stroke labels.

Four labelled clips now cover right- and left-handed forehands and backhands,
plus recovery, volleys, a serve, and an overhead. The companion
`rtmpose_tennis.annotations` tool validates ordered, non-overlapping intervals
and joins them to a pose JSONL stream by timestamp. This establishes a
repeatable ground-truth dataset without coupling labels or offline file I/O to
the real-time inference loop.

### 20. Offline event-recognition baseline

`rtmpose_tennis.stroke_analysis` builds one robust feature vector per annotated
event and evaluates a three-nearest-neighbor baseline using leave-one-video-out
splits. Groundstroke forehands and backhands are separate classes; recovery,
serve, overhead, and volleys are grouped as `other` for the first experiment.
This avoids frame leakage and measures recognition only—the annotated event
boundaries are supplied to the model, so automatic segmentation is not yet part
of the score.

The initial four-video baseline contains 72 events and scores 59/72 (81.9%).
It recognizes 17/22 forehands and 37/39 `other` events, but only 5/11
backhands. Per-video accuracy ranges from 66.7% on the left-handed mixed clip
to 100% on the forehand-only clip. The result is useful as a reproducible
starting point, not a production accuracy claim. Backhand recall and
cross-player canonicalization should improve before live integration.

Adding the first 51.3 seconds of the right-handed `baseline.mp4` session adds
13 forehands and five backhands. The corrected five-video evaluation contains
107 events and scores 86/107 (80.4%). It recognizes 24/35 forehands, 7/16
backhands, and 55/56 `other` events. An earlier draft mistakenly represented a
15.7-second region containing five forehands as one recovery event, which
inflated accuracy to 83.5%; that result is obsolete. The corrected errors are
dominated by forehand/backhand confusion, making robust temporal direction and
pose-outlier handling the next feature-quality target.

A second-pass review of the left-handed `mix.mp4` boundaries increases the
dataset to 108 events and the score to 88/108 (81.5%). Forehand recognition is
27/35, backhand recognition is 6/16, and `other` recognition is 55/57. The held
out `mix.mp4` fold improves from 66.7% to 71.4%. Five of its seven groundstroke
backhands are still classified as `other` and one as forehand, confirming that
backhand trajectory representation—not merely annotation noise—is the leading
model weakness.

Reviewing the right-handed `wnn.mp4` boundaries increases the inventory to 109
events and improves the baseline to 91/109 (83.5%). Its held-out fold improves
from 81.0% to 86.4%, while aggregate backhand recognition improves from 6/16
to 9/16. The remaining `wnn.mp4` errors are its opening forehand classified as
`other`, one backhand classified as forehand, and the verified 30.83–31.61
recovery tail classified as forehand.

Extending verified `baseline.mp4` labels through 72.25 seconds adds four
forehands, two backhands, and five recovery events. An explicitly uncertain
60.04–65.04 interval is labelled `unknown` and excluded from classifier training
and evaluation. The benchmark now contains 120 events and scores 101/120
(84.2%): 29/39 forehands, 11/18 backhands, and 61/63 `other` events. The held-out
`baseline.mp4` fold scores 38/46 (82.6%).

### 21. Confidence-aware temporal stroke features

The second offline feature path rejects low-confidence or implausible joint
coordinates, interpolates the remaining dominant wrist, elbow, shoulder, and
non-dominant wrist observations onto 21 normalized time points, applies a short
median/weighted smoother, and retains seven ordered swing phases. It derives
motion direction, speed, extension, elbow angle, wrist separation, peak timing,
and start-to-end displacement from the repaired trajectory.

Smoothed features alone also score 101/120 (84.2%), but materially change the
error distribution: backhand recognition improves from 11/18 to 14/18 and
direct forehand/backhand confusion drops from nine events to three, while ten
forehands are rejected as `other`. A hierarchical design therefore uses the
aggregate representation as a groundstroke gate and the temporal
representation only for forehand/backhand direction.

To avoid choosing neighbor counts on the test video, each outer held-out-video
fold performs an inner leave-one-video-out search using only its four training
videos. This nested hierarchical evaluation scores 104/120 (86.7%): 29/39
forehands, 14/18 backhands, and 61/63 `other` events. The original aggregate
method remains available with `--method aggregate`, and the temporal-only
comparison with `--method smoothed`. This is an offline recognition result with
known event boundaries and has no impact on live inference FPS.

### 22. Offline automatic stroke-boundary proposals

`rtmpose_tennis.stroke_segmentation` evaluates the first continuous-stream
event detector independently of stroke recognition. It confidence-filters and
interpolates the canonical dominant-wrist trajectory, applies time-based median
and mean smoothing, converts velocity to a compressed motion-energy signal,
selects separated local peaks, and places preparation/follow-through windows
around them. Forehands, backhands, volleys, serves, and overheads are all stroke
events; recovery is negative footage, while explicit `unknown` and unlabeled
regions are outside evaluation.

Each outer held-out-video fold selects its threshold quantile, peak separation,
maximum search window, and low-motion boundary quantile using only the other
four videos. The adaptive detector searches backward from the peak for motion
onset and forward for deceleration, retaining a short context pad. At interval
IoU 0.30, it matches 60/61 strokes from 64 proposals: 93.8% precision, 98.4%
recall, and 96.0% F1. Mean matched IoU improves to 0.695, with mean absolute
start/end errors of 267/265 ms. The earlier fixed-window result was 93.7% F1,
0.633 IoU, and 427/300 ms errors.

This remains an offline segmentation result and does not yet combine proposal
classification, confidence-based rejection, or live feedback latency. It adds
no work to the real-time application until explicitly integrated.

### 23. Offline end-to-end stroke pipeline

`rtmpose_tennis.stroke_pipeline` passes automatically generated intervals into
the hierarchical recognizer. For every outer held-out video, segmentation and
classifier settings are selected using only the other four videos. Proposal
features are extracted from the predicted window rather than the verified
annotation, making this the first measurement of compounded detection and
recognition behavior.

The first adaptive-window run still trained the recognizer on manually bounded
events, producing only 39/61 correct and 62.4% end-to-end F1 because adaptive
test windows had a different feature distribution. Training each outer-fold
classifier on automatically proposed windows from its four training videos
corrects that mismatch. Matched training proposals inherit their verified
class; unmatched training proposals teach the groundstroke gate `other`.

The proposal-shaped result correctly detects and classifies 45/61 strokes from
64 proposals. End-to-end precision is 70.3%, recall is 73.8%, and F1 is 72.0%.
Detection recall is 98.4%, matched-proposal classification is 75.0%, and
unmatched proposals produce 1.49 false alerts per evaluated minute. Per-video
correct counts are 15/24 for `baseline.mp4`, 4/6 for `kid.mp4`, 10/14 for
`mix.mp4`, 6/6 for `pro.mov`, and 10/11 for `wnn.mp4`. At IoU 0.50, end-to-end
F1 is 65.6% and matched-proposal classification is 82.0%.

This remains offline and adds no work to the real-time application. The next
decision is whether to improve cross-session proposal classification further or
establish a streaming state-machine baseline with explicit decision latency.

### 24. Peak-aligned classification experiment

Automatic proposals now retain the timestamp of their dominant-wrist motion
peak separately from their adaptive onset and deceleration boundaries. The
offline pipeline can therefore continue using adaptive intervals for detection
IoU and user-facing start/end times while extracting temporal classification
features from a consistent peak-relative window. For every outer held-video
fold, 16 windows from 0.45–0.90 seconds before and after the peak are evaluated
using only the four training videos; neighbour counts are selected inside the
same training-only process.

The experiment did not improve the current dataset. Interval alignment remains
at 45/61 correctly detected and classified strokes, 75.0% matched-proposal
classification, and 72.0% end-to-end F1. Peak alignment produces 42/61,
70.0%, and 67.2%, respectively. It improves `baseline.mp4` from 15/24 to 17/24
correct but reduces `pro.mov` from 6/6 to 4/6, `wnn.mp4` from 10/11 to 8/11,
and `mix.mp4` from 10/14 to 9/14.

The likely limitation is semantic rather than computational: the largest 2D
wrist-speed peak is not guaranteed to represent the same stroke phase across
viewpoints and players. It may occur during acceleration, near contact, or in
follow-through, and pose noise can move it further. Peak alignment is retained
behind `--classifier-alignment peak` for reproducible comparison, while
interval alignment stays the default and preserves the established baseline.

### 25. Multi-anchor motion-phase classification

The next temporal representation replaces the single-peak assumption with five
anchors at 10%, 30%, 50%, 70%, and 90% of cumulative dominant-wrist motion.
These anchors adapt to how motion is distributed within each automatically
proposed interval. The classifier compares dominant and non-dominant wrist
positions, wrist velocity and speed, arm extension, elbow angle, wrist
separation, phase timing, and displacement between adjacent phases. Confidence
filtering, trajectory repair, smoothing, handedness canonicalization, and
training-video-only model selection remain unchanged.

With verified event boundaries, the hierarchical held-video result improves
from 104/120 (86.7%) to 111/120 (92.5%). Direct forehand/backhand confusion is
eliminated in this test: 36/39 forehands and 14/18 backhands are correct, while
61/63 `other` events remain correct.

At the operational IoU 0.30 threshold with automatic proposals, correctly
detected and classified strokes improve from 45/61 to 49/61. Matched-proposal
classification rises from 75.0% to 81.7%, and end-to-end F1 rises from 72.0%
to 78.4%; detection recall and unmatched false alerts remain 98.4% and 1.49 per
minute because the proposal stage is unchanged. Results improve on
`baseline.mp4` (15/24 to 17/24), `mix.mp4` (10/14 to 12/14), and `wnn.mp4`
(10/11 to 11/11), stay 6/6 on `pro.mov`, and regress from 4/6 to 3/6 on
`kid.mp4`.

At strict IoU 0.50, the result is slightly lower than the former representation:
40 versus 41 correctly detected and classified strokes, and 64.0% versus 65.6%
F1. Motion phases are promoted as the operational IoU 0.30 default because of
the larger cross-video classification improvement, while
`--classifier-features smoothed` retains the previous representation for exact
comparison. This feature extraction remains offline and does not affect live
application FPS.

### 26. Confidence-aware audit and abstention

The end-to-end report now records every matched proposal, unmatched proposal,
and missed stroke. Matched audit entries include the verified and predicted
classes, IoU, dominant-wrist motion peak, five phase-anchor timestamps, mean
pose confidence, nearest-class distances and votes for both classifier stages,
and an explicit `groundstroke_gate` or `forehand_backhand` failure stage.
Across the unfiltered benchmark, the 11 classification errors divide into six
forehand/backhand errors and five groundstroke-gate errors; there are also four
unmatched proposals and one missed stroke.

A distance margin from either representation alone was not sufficiently
calibrated. The selective rule therefore requires agreement between the
motion-phase and uniformly sampled temporal classifiers, then applies their
minimum nearest-class distance margin. Each outer fold selects its confidence
threshold from only the other four videos, targeting at least 90% accepted
accuracy while retaining as much training coverage as possible; the held-out
video never selects its threshold.

On the held-video benchmark, the rule accepts 40/60 matched proposals (66.7%
coverage) and correctly classifies 37/40 (92.5%). None of the four unmatched
proposals passes the confidence filter, reducing confident false alerts from
1.49 to 0.00 per evaluated minute. Among accepted verified forehands and
backhands, 37/38 are fully correct; the remaining forehand is rejected by the
groundstroke gate as `other`, so there are no accepted direct
forehand/backhand inversions in this small dataset.

This is a selective safety metric, not a replacement for the unfiltered 78.4%
end-to-end F1. Coverage varies materially by held-out video: 100% for `pro.mov`,
72.7% for `wnn.mp4`, 84.6% for `mix.mp4`, 54.2% for `baseline.mp4`, and only
33.3% for `kid.mp4`. The accepted `kid.mp4` subset is still just 1/2 correct,
confirming that broader player and non-groundstroke training data are needed
before real-time audible decisions.
