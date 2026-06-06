"""
Player and ball detector — YOLO11 + ByteTrack.

Ported from tracking_pipeline.py:
  - Per-frame tracking via model.track(persist=True), which keeps ByteTrack
    state alive across calls (equivalent to stream=True over the full video).
  - CLASS_NAMES {0: goalkeeper, 1: player, 2: referee} — custom handball model.
  - Ghost-track cleanup happens as a post-processing step in the orchestrator
    (SQL DELETE after all frames are written), not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ingestion.types import BoundingBox, Detection

# Class IDs for the custom handball model
_PLAYER_CLASS_IDS = {0, 1, 2}  # goalkeeper, player, referee


class Detector:
    """YOLO11 + ByteTrack player and ball detector."""

    def __init__(
        self,
        model_path: str,
        ball_model_path: str | None,
        confidence: float,
        ball_confidence: float,
        max_persons: int,
        device: str,
        tracker: str | Path = "bytetrack.yaml",
    ) -> None:
        # Lazy import — ultralytics is an optional [cv] dep, not installed in CI
        from ultralytics import YOLO  # type: ignore[import-unresolved]

        self._model = YOLO(model_path)
        self._ball_model = YOLO(ball_model_path) if ball_model_path else None
        self._conf = confidence
        self._ball_conf = ball_confidence
        self._max_persons = max_persons
        self._device = device
        self._tracker = str(tracker)  # ultralytics expects a str

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], Detection | None]:
        """
        Run detection + ByteTrack on one frame.

        Returns:
            players: tracked player detections (track_id stable across frames)
            ball:    highest-confidence ball detection, or None if not visible
        """
        results = self._model.track(
            frame,
            persist=True,  # keeps ByteTrack state alive between calls
            conf=self._conf,
            device=self._device,
            tracker=self._tracker,
            imgsz=640,
            verbose=False,
        )[0]

        players: list[Detection] = []

        if results.boxes.id is not None:
            for i in range(len(results.boxes)):
                cls_id = int(results.boxes.cls[i])
                if cls_id not in _PLAYER_CLASS_IDS:
                    continue
                if len(players) >= self._max_persons:
                    break
                track_id = int(results.boxes.id[i])
                conf = float(results.boxes.conf[i])
                x1, y1, x2, y2 = (int(v) for v in results.boxes.xyxy[i])
                players.append(
                    Detection(
                        track_id=track_id,
                        bbox=BoundingBox(x1, y1, x2, y2),
                        confidence=conf,
                    )
                )

        ball = self._detect_ball(frame)
        return players, ball

    def _detect_ball(self, frame: np.ndarray) -> Detection | None:
        """Run the dedicated ball model if configured, return best detection or None."""
        if self._ball_model is None:
            return None

        results = self._ball_model.track(
            frame,
            persist=True,
            conf=self._ball_conf,
            device=self._device,
            imgsz=640,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        best = int(results.boxes.conf.argmax())
        conf = float(results.boxes.conf[best])
        x1, y1, x2, y2 = (int(v) for v in results.boxes.xyxy[best])
        track_id = int(results.boxes.id[best]) if results.boxes.id is not None else -1
        return Detection(
            track_id=track_id,
            bbox=BoundingBox(x1, y1, x2, y2),
            confidence=conf,
        )
