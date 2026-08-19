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

- The application is synchronous, so display work still blocks inference.
- A detector refresh still causes a roughly 200 ms latency spike.
- `model FPS` excludes decoding and display; `output FPS` is the meaningful
  current end-to-end delivery rate.
- Pose landmarks can be wrong during motion blur, limb crossing, or occlusion.
- Pose-guided tracking depends on sufficiently confident landmarks and uses
  RTMDet for recovery.
- A 50% preview is visibly softer, although raw frames and pose data are not.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is still useful for unsupported PyTorch MPS
  operations.

## Next optimization

The next planned architectural step is asynchronous latest-frame processing:

```text
Capture thread -> always retain newest camera frame
Inference loop -> process newest available frame; discard stale frames
Display        -> show newest completed pose
```

This should reduce live-camera latency and allow capture/display work to overlap
inference. The synchronous baseline above should be retained for regression and
quality comparisons.
