# Pipeline Internals

The ingestion pipeline is a sequence of pure transformation stages.
Each stage takes typed inputs and returns typed outputs — no file I/O or
database calls happen inside a stage.

## Data types

All stages pass data through the types defined in `ingestion.types`:

```
BoundingBox(x1, y1, x2, y2)
    └── .center  → (float, float)   mid-point
    └── .foot    → (float, float)   bottom-center (used as ground position)

Detection(track_id, bbox, confidence)
    └── raw output from the detector; no team, no court, no pose

PlayerState(track_id, bbox, confidence, team, court_pos, pose, on_court)
    └── fully enriched; built by assembling all stage outputs

BallState(bbox, confidence, court_pos)

FrameState(frame_id, timestamp_s, players, ball)
    └── one per video frame; written to DuckDB by the writer
```

::: ingestion.types
    options:
      show_source: false
      members:
        - BoundingBox
        - Detection
        - PlayerState
        - BallState
        - FrameState

## Stage 1 — Detection & tracking

**Module:** `ingestion.pipeline.detection.Detector`

**Input:** raw video frame (`np.ndarray`, BGR)

**Output:** `(list[Detection], Detection | None)` — player detections + ball detection

The detector runs YOLO26 on each frame with two targets:
- **Person class** → player bounding boxes
- **Sports ball class** → ball bounding box (highest-confidence result only)

[BoT-SORT](https://github.com/NirAharon/BoT-SORT) (built into ultralytics) assigns
a stable `track_id` to each player across frames. The same player keeps the same ID
even if they temporarily leave the frame. ReID via `osnet_x0_25_market.pt` further
stabilises identities across longer occlusions.

```
frame (1280×720 BGR)
    │
    │  YOLO26 inference
    ▼
raw detections + track IDs
    │
    │  BoundingBox construction
    ▼
list[Detection]  +  Detection | None (ball)
```

## Stage 2 — Pose estimation

**Module:** `ingestion.pipeline.pose.PoseEstimator`

**Input:** frame + `list[Detection]`

**Output:** `list[list[Keypoint] | None]` — one entry per detection, `None` if no pose matched

YOLO26-pose runs a single GPU forward pass per frame (all players at once).
Results are matched back to player detections by center-point distance.
Each pose contains 17 COCO keypoints:

```
0: nose          5: left shoulder    10: right wrist   15: left ankle
1: left eye      6: right shoulder   11: left hip      16: right ankle
2: right eye     7: left elbow       12: right hip
3: left ear      8: right elbow      13: left knee
4: right ear     9: left wrist       14: right knee
```

Each `Keypoint` has `(x, y, z, visibility)` where `visibility` is a confidence
score — keypoints below ~0.5 are unreliable and should be treated as absent.

## Stage 3 — Team classification

**Module:** `ingestion.pipeline.team.TeamClassifier`

**Input:** frame + `BoundingBox`

**Output:** `"A"` | `"B"` | `"unknown"`

Team assignment works in two phases:

### Warm-up (first N frames)

For each player detection, the classifier extracts a **torso crop** (the upper
40% of the bounding box) and computes an **HSV histogram** of the hue channel.
These histograms are accumulated as samples.

### Fitting

After the warm-up, `fit()` runs **K-Means** (k=2) on the collected HSV histograms.
The two clusters correspond to the two teams' jersey colors. Cluster labels are
assigned consistently — the team label mapping is stable within a match.

### Classification

For each subsequent frame, `classify()` extracts the torso HSV histogram and
predicts the cluster. Players whose histogram doesn't fit either cluster cleanly
(e.g. referees in different colors) get team `"unknown"`.

!!! note
    Team labels "A" and "B" are **arbitrary within a match** — there is no
    guarantee that "A" always refers to the home team. The backend layer
    maps these to real team names via the `team_a_name` / `team_b_name` fields
    in the `matches` table.

## Stage 4 — Court mapping

**Module:** `ingestion.pipeline.court.CourtMapper`

**Input:** pixel coordinate `(float, float)`

**Output:** court coordinate `(float, float) | None`

Given a [calibration file](running.md#court-calibration), the mapper computes a
**homography matrix** (3×3) using four pixel↔court point correspondences via
`cv2.findHomography`. This matrix transforms any pixel position into real-world
metres on the 40m × 20m court.

```
pixel (x_px, y_px)
    │  H · [x_px, y_px, 1]ᵀ
    ▼
court (x_m, y_m)  where  x_m ∈ [0, 40],  y_m ∈ [0, 20]
```

Returns `None` if the transformed point falls outside valid court bounds — this
happens when the camera is zoomed onto a region that can't be reliably mapped
(e.g. a close-up of a player near the bench).

## Orchestrator

`ingestion.orchestrator.IngestionOrchestrator` wires all four stages together.
It owns all stage instances and handles the warm-up state machine:

```python
orchestrator = IngestionOrchestrator(settings)
orchestrator.run(video_path, match_id)
```

Internally:

```
for frame in iter_frames(video):          # Phase 1: warm-up
    detections, _ = detector.detect(frame)
    for det in detections:
        team_classifier.collect(frame, det.bbox)

team_classifier.fit()                     # fit K-Means once

with FrameWriter(conn, match_id):
    for frame in iter_frames(video):      # Phase 2: full pipeline
        detections, ball = detector.detect(frame)
        poses = pose_estimator.estimate(frame, detections)
        players = [
            PlayerState(
                team=team_classifier.classify(frame, det.bbox),
                court_pos=court_mapper.transform(det.bbox.foot),
                pose=poses[i],
                ...
            )
            for i, det in enumerate(detections)
        ]
        writer.write(FrameState(...))

_compute_velocities(conn, match_id, fps)  # Phase 3: SQL post-processing
_mark_ball_carrier(conn, match_id)
```

The video is read **twice**: once for warm-up, once for processing. This avoids
buffering hundreds of raw frames in memory.
