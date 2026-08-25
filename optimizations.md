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
