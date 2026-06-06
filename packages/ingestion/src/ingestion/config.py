from pathlib import Path

from pydantic_settings import BaseSettings


class IngestionSettings(BaseSettings):
    model_config = {"env_prefix": "WELS_"}

    # Model identifiers — resolved against models_dir if not absolute paths
    detection_model: str = "yolo11m.pt"
    pose_model: str = "yolo11m-pose.pt"
    ball_model: str | None = None  # custom fine-tuned ball model; falls back to detection_model

    # Detection thresholds
    detection_confidence: float = 0.3
    ball_confidence: float = 0.25
    max_persons: int = 20
    n_teams: int = 2

    # Processing flags
    skip_pose: bool = False
    device: str = "cuda"  # set to "cpu" on machines without a GPU

    # Storage — paths are relative to the repo root (data/)
    duckdb_path: Path = Path("data/matches.duckdb")
    models_dir: Path = Path("data/models")

    # Court calibration JSON file (optional — no court mapping if unset)
    calibration_path: Path | None = None

    # How many frames to accumulate before fitting the team classifier
    team_warmup_frames: int = 150

    # Ghost-track cleanup: tracks seen in fewer than this many frames are removed
    ghost_threshold: int = 10

    # Tracker config passed to ultralytics — path relative to repo root or absolute
    tracker_config: Path = Path("packages/ingestion/botsort_custom.yaml")


settings = IngestionSettings()
