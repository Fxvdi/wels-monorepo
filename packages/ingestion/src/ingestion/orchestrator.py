"""
Ingestion orchestrator — wires the pipeline stages together.

Flow per video:
  1. Open video, register match in DuckDB
  2. Warm-up phase: collect player detections for team classifier fitting
     (team labels are "unknown" during this phase)
  3. Fit team classifier on the accumulated samples
  4. Main phase: run full pipeline frame-by-frame and write to DuckDB
  5. Post-processing: compute velocities and ball-carrier flags via SQL

The orchestrator is the only place that knows about all pipeline stages.
Individual stages (detection, pose, team, court) have no knowledge of each other.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ingestion.config import IngestionSettings
from ingestion.pipeline.court import CourtMapper
from ingestion.pipeline.detection import Detector
from ingestion.pipeline.pose import PoseEstimator
from ingestion.pipeline.team import TeamClassifier
from ingestion.storage.schema import connect
from ingestion.storage.writer import FrameWriter
from ingestion.types import BallState, FrameState, PlayerState
from ingestion.video import iter_frames

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    def __init__(self, settings: IngestionSettings) -> None:
        self._settings = settings

        self._detector = Detector(
            model_path=settings.detection_model,
            ball_model_path=settings.ball_model,
            confidence=settings.detection_confidence,
            ball_confidence=settings.ball_confidence,
            max_persons=settings.max_persons,
            device=settings.device,
            tracker=settings.tracker_config,
        )

        self._pose: PoseEstimator | None = None
        if not settings.skip_pose:
            self._pose = PoseEstimator(
                model_path=settings.pose_model,
                device=settings.device,
            )

        self._team = TeamClassifier(n_teams=settings.n_teams)

        self._court: CourtMapper | None = None
        if settings.calibration_path is not None:
            self._court = CourtMapper.from_file(settings.calibration_path)

    def run(self, video_path: Path, match_id: str) -> None:
        """
        Process one video file end-to-end and write all results to DuckDB.

        Raises ValueError if match_id already exists in the database.
        """
        settings = self._settings
        conn = connect(settings.duckdb_path)

        # Guard: reject duplicate match_ids
        existing = conn.execute("SELECT 1 FROM matches WHERE match_id = ?", [match_id]).fetchone()
        if existing:
            raise ValueError(f"match_id '{match_id}' already exists in the database")

        # Two-phase processing: warm-up → fit → main
        warmup_limit = settings.team_warmup_frames

        logger.info("Phase 1: team classifier warm-up (%d frames)", warmup_limit)
        for vf in iter_frames(video_path):
            if vf.frame_id >= warmup_limit:
                break
            players_raw, _ = self._detector.detect(vf.frame)
            for det in players_raw:
                self._team.collect(vf.frame, det.bbox)

        self._team.fit()
        logger.info("Team classifier fitted")

        # Register the match now that we know fps/total_frames
        # We need one more pass for metadata — read the first frame only
        meta = next(iter_frames(video_path))
        conn.execute(
            "INSERT INTO matches (match_id, video_path, fps, total_frames) VALUES (?,?,?,?)",
            [match_id, str(video_path), meta.fps, meta.total_frames],
        )
        conn.commit()

        logger.info("Phase 2: full pipeline (%d total frames)", meta.total_frames)
        with FrameWriter(conn, match_id) as writer:
            for vf in iter_frames(video_path):
                frame_state = self._process_frame(vf.frame, vf.frame_id, vf.timestamp_s)
                writer.write(frame_state)

                if vf.frame_id % 500 == 0:
                    logger.info("  frame %d / %d", vf.frame_id, vf.total_frames)

        logger.info("Phase 3: ghost track cleanup (min %d frames)", settings.ghost_threshold)
        removed = _remove_ghost_tracks(conn, match_id, settings.ghost_threshold)
        conn.commit()
        logger.info("  removed %d ghost track(s)", removed)

        logger.info("Phase 4: track interpolation (fill gaps within each track's lifespan)")
        added = _interpolate_tracks(conn, match_id)
        conn.commit()
        logger.info("  interpolated %d frame(s)", added)

        logger.info("Phase 5: post-processing (velocities + ball carrier)")
        _compute_velocities(conn, match_id, meta.fps)
        _mark_ball_carrier(conn, match_id)
        conn.commit()

        logger.info("Ingestion complete: %s", match_id)

    def _process_frame(self, frame: object, frame_id: int, timestamp_s: float) -> FrameState:
        import numpy as np

        assert isinstance(frame, np.ndarray)

        players_raw, ball_raw = self._detector.detect(frame)

        # Pose estimation (optional)
        poses = None
        if self._pose is not None and players_raw:
            poses = self._pose.estimate(frame, players_raw)

        # Assemble PlayerState for each detection
        players = []
        for i, det in enumerate(players_raw):
            team = self._team.classify(frame, det.bbox)
            court_pos = self._court.transform(det.bbox.foot) if self._court else None
            pose = poses[i] if poses is not None else None
            players.append(
                PlayerState(
                    track_id=det.track_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    team=team,
                    court_pos=court_pos,
                    pose=pose,
                )
            )

        # Ball state
        ball = None
        if ball_raw is not None:
            court_pos = self._court.transform(ball_raw.bbox.center) if self._court else None
            ball = BallState(
                bbox=ball_raw.bbox,
                confidence=ball_raw.confidence,
                court_pos=court_pos,
            )

        return FrameState(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            players=players,
            ball=ball,
        )


# ---------------------------------------------------------------------------
# Post-processing SQL — runs after all frames are written
# ---------------------------------------------------------------------------


def _compute_velocities(conn: object, match_id: str, fps: float) -> None:
    """
    Fill velocity_x / velocity_y from position deltas using SQL window functions.
    Requires court_x/court_y to be non-NULL (i.e. calibration was provided).
    """
    import duckdb as _duckdb

    assert isinstance(conn, _duckdb.DuckDBPyConnection)

    conn.execute(
        """
        UPDATE players AS p
        SET velocity_x = sub.vx,
            velocity_y = sub.vy
        FROM (
            SELECT
                match_id,
                frame_id,
                track_id,
                (court_x - LAG(court_x) OVER w)
                    / NULLIF(frame_id - LAG(frame_id) OVER w, 0) * ? AS vx,
                (court_y - LAG(court_y) OVER w)
                    / NULLIF(frame_id - LAG(frame_id) OVER w, 0) * ? AS vy
            FROM players
            WHERE match_id = ?
            WINDOW w AS (PARTITION BY match_id, track_id ORDER BY frame_id)
        ) sub
        WHERE p.match_id = sub.match_id
          AND p.frame_id = sub.frame_id
          AND p.track_id = sub.track_id
        """,
        [fps, fps, match_id],
    )


def _interpolate_tracks(conn: object, match_id: str) -> int:
    """
    Linearly interpolate missing frames within each track's lifespan.

    Mirrors tracking_pipeline.interpolate_tracks(): for each track, fills in
    frames between its first and last detection using numpy.interp.
    Inserts new rows into `players`; velocity/has_ball are left at defaults
    and recomputed in Phase 5.
    """
    import duckdb as _duckdb
    import numpy as np

    assert isinstance(conn, _duckdb.DuckDBPyConnection)

    rows = conn.execute(
        """
        SELECT track_id, frame_id, team, on_court,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2,
               pixel_foot_x, pixel_foot_y, confidence,
               court_x, court_y
        FROM players
        WHERE match_id = ?
        ORDER BY track_id, frame_id
        """,
        [match_id],
    ).fetchall()

    if not rows:
        return 0

    # Group rows by track_id (already ordered)
    tracks: dict[int, list[tuple[object, ...]]] = {}
    for row in rows:
        tid = int(row[0])
        tracks.setdefault(tid, []).append(row)

    new_rows: list[tuple[object, ...]] = []

    for tid, track_rows in tracks.items():
        frame_ids = np.array([r[1] for r in track_rows], dtype=np.int64)
        if frame_ids[0] == frame_ids[-1]:
            continue  # single frame, nothing to fill

        full_range = np.arange(frame_ids[0], frame_ids[-1] + 1, dtype=np.int64)
        missing = np.setdiff1d(full_range, frame_ids)
        if missing.size == 0:
            continue

        # Numeric columns — interpolate linearly
        num_keys = [
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "pixel_foot_x",
            "pixel_foot_y",
            "confidence",
        ]
        num_vals = {
            k: np.array([r[i + 4] for r in track_rows], dtype=np.float64)
            for i, k in enumerate(num_keys)
        }

        # Court coords — interpolate only where values are known
        raw_cx = np.array([r[11] if r[11] is not None else np.nan for r in track_rows])
        raw_cy = np.array([r[12] if r[12] is not None else np.nan for r in track_rows])
        has_court = not np.all(np.isnan(raw_cx))
        if has_court:
            known = ~np.isnan(raw_cx)
            known_fids = frame_ids[known]
            known_cx = raw_cx[known]
            known_cy = raw_cy[known]

        for mf in missing.tolist():
            interp = {k: float(np.interp(mf, frame_ids, num_vals[k])) for k in num_keys}

            court_x: float | None = None
            court_y: float | None = None
            if has_court and known_fids[0] <= mf <= known_fids[-1]:
                court_x = float(np.interp(mf, known_fids, known_cx))
                court_y = float(np.interp(mf, known_fids, known_cy))

            # Forward-fill team and on_court from the last known frame before mf
            team = track_rows[0][2]
            on_court = track_rows[0][3]
            for r in track_rows:
                if r[1] <= mf:
                    team = r[2]
                    on_court = r[3]
                else:
                    break

            new_rows.append(
                (
                    match_id,
                    int(mf),
                    tid,
                    team,
                    bool(on_court),
                    round(interp["bbox_x1"]),
                    round(interp["bbox_y1"]),
                    round(interp["bbox_x2"]),
                    round(interp["bbox_y2"]),
                    interp["pixel_foot_x"],
                    interp["pixel_foot_y"],
                    interp["confidence"],
                    court_x,
                    court_y,
                )
            )

    for row in new_rows:
        conn.execute(
            """
            INSERT INTO players
                (match_id, frame_id, track_id, team, on_court,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 pixel_foot_x, pixel_foot_y, confidence,
                 court_x, court_y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(row),
        )

    return len(new_rows)


def _remove_ghost_tracks(conn: object, match_id: str, min_frames: int) -> int:
    """Delete player rows whose track appeared in fewer than min_frames distinct frames."""
    import duckdb as _duckdb

    assert isinstance(conn, _duckdb.DuckDBPyConnection)

    ghost_ids: list[int] = [
        row[0]
        for row in conn.execute(
            """
            SELECT track_id
            FROM players
            WHERE match_id = ?
            GROUP BY track_id
            HAVING COUNT(DISTINCT frame_id) < ?
            """,
            [match_id, min_frames],
        ).fetchall()
    ]

    if not ghost_ids:
        return 0

    placeholders = ", ".join("?" * len(ghost_ids))
    conn.execute(
        f"DELETE FROM players WHERE match_id = ? AND track_id IN ({placeholders})",
        [match_id, *ghost_ids],
    )
    return len(ghost_ids)


def _mark_ball_carrier(conn: object, match_id: str) -> None:
    """Mark the player closest to the ball in each frame as has_ball=TRUE."""
    import duckdb as _duckdb

    assert isinstance(conn, _duckdb.DuckDBPyConnection)

    conn.execute(
        """
        UPDATE players AS p
        SET has_ball = TRUE
        FROM (
            SELECT pl.match_id, pl.frame_id, pl.track_id
            FROM players pl
            JOIN ball b ON pl.match_id = b.match_id AND pl.frame_id = b.frame_id
            WHERE pl.match_id = ?
              AND pl.court_x IS NOT NULL
              AND b.court_x  IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY pl.match_id, pl.frame_id
                ORDER BY (pl.court_x - b.court_x)^2 + (pl.court_y - b.court_y)^2
            ) = 1
        ) sub
        WHERE p.match_id = sub.match_id
          AND p.frame_id = sub.frame_id
          AND p.track_id = sub.track_id
        """,
        [match_id],
    )
