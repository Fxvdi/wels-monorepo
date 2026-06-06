# Ingestion — Overview

The ingestion pipeline turns a raw match video into structured data in DuckDB.
It is the bridge between the recorded footage and every downstream feature —
analytics, heatmaps, and ML predictions.

## End-to-end flow

```
┌──────────────┐
│  match.mp4   │  ← any camera angle, 720p+, 25 FPS+
└──────┬───────┘
       │  wels-ingest
       ▼
┌─────────────────────────────────────────────┐
│  Stage 1 — Computer Vision pipeline         │
│                                             │
│  1. Detect players + ball  (YOLO26)         │
│  2. Track identities       (BoT-SORT)       │
│  3. Estimate body pose     (YOLO26-pose)    │
│  4. Classify team          (K-Means HSV)    │
│  5. Map to court coords    (homography)     │
└──────────────────┬──────────────────────────┘
                   │  per-frame FrameState
                   ▼
┌──────────────────────────┐
│  data/matches.duckdb     │  ← matches, frames, players, ball tables
└──────────────────┬───────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
  Backend API            ML training
  (analytics)            (action prediction)
```

## Data directory layout

All runtime data lives under `data/` at the repo root. This directory is not committed.

```
data/
├── matches.duckdb          ← single DuckDB file, all matches
├── models/
│   ├── yolo26m.pt          ← downloaded automatically on first run
│   ├── yolo26m-pose.pt     ← downloaded automatically on first run
│   └── action_best.pt      ← produced by wels-train
└── videos/                 ← recommended location for match recordings
    └── 2026-04-13_wels_vs_linz.mp4
```

!!! note
    Video files can be stored anywhere — you pass the full path to `wels-ingest`.
    The `data/videos/` location is a convention, not a requirement.

## What gets stored

After processing one match, DuckDB contains:

| Table | Rows per match (60 min @ 25 FPS) | Description |
|-------|----------------------------------|-------------|
| `matches` | 1 | Match metadata: video path, FPS, team names |
| `frames` | ~90,000 | One row per frame: timestamp, player count |
| `players` | ~1,260,000 | Per-player per-frame: position, team, velocity |
| `ball` | ~90,000 | Ball position per frame (NULL when not detected) |
| `action_labels` | varies | Manual annotations added for ML training |

Stored on disk with DuckDB's columnar compression: roughly **50–80 MB per match**.

## Three processing phases

The orchestrator runs three phases per video:

```
Phase 1 — Warm-up (first 150 frames)
  Collect player jersey-color samples for team classification.
  Players are written as team="unknown" during this phase.

Phase 2 — Main pass (all frames)
  Fit the team classifier on the collected samples, then run
  the full pipeline frame-by-frame and write to DuckDB.

Phase 3 — Post-processing (SQL)
  Compute velocity (position delta per frame via window functions).
  Mark ball carrier (nearest player to ball in each frame).
```

The warm-up length is configurable via `WELS_TEAM_WARMUP_FRAMES` (default: `150`).
