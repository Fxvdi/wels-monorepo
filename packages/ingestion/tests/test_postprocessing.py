"""
Unit tests for orchestrator post-processing functions:
  - _remove_ghost_tracks
  - _interpolate_tracks
"""

import duckdb

from ingestion.orchestrator import _interpolate_tracks, _remove_ghost_tracks
from ingestion.storage.schema import _SCHEMA_SQL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(_SCHEMA_SQL)
    conn.execute("INSERT INTO matches VALUES ('m1', 'v.mp4', 25.0, 100, NULL, NULL, NOW())")
    for fid in range(20):
        conn.execute(
            "INSERT INTO frames VALUES ('m1', ?, ?, 1, 1)",
            [fid, round(fid / 25.0, 4)],
        )
    return conn


def _add_player(
    conn: duckdb.DuckDBPyConnection,
    frame_id: int,
    track_id: int,
    x1: int = 100,
    y1: int = 200,
    x2: int = 150,
    y2: int = 280,
    court_x: float | None = None,
    court_y: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO players
            (match_id, frame_id, track_id, team, on_court,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             pixel_foot_x, pixel_foot_y, confidence,
             court_x, court_y)
        VALUES ('m1', ?, ?, 'A', TRUE, ?, ?, ?, ?, ?, ?, 0.9, ?, ?)
        """,
        [frame_id, track_id, x1, y1, x2, y2, (x1 + x2) / 2.0, float(y2), court_x, court_y],
    )


def _player_count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT COUNT(*) FROM players WHERE match_id='m1'").fetchone()[0]  # type: ignore[index]


def _track_ids(conn: duckdb.DuckDBPyConnection) -> set[int]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT track_id FROM players WHERE match_id='m1'"
        ).fetchall()
    }


# ---------------------------------------------------------------------------
# _remove_ghost_tracks
# ---------------------------------------------------------------------------


def test_ghost_tracks_removed_below_threshold() -> None:
    conn = _make_db()
    # Track 1: 15 frames (stable), Track 2: 3 frames (ghost)
    for fid in range(15):
        _add_player(conn, fid, track_id=1)
    for fid in range(3):
        _add_player(conn, fid, track_id=2)

    removed = _remove_ghost_tracks(conn, "m1", min_frames=10)

    assert removed == 1
    assert _track_ids(conn) == {1}


def test_ghost_tracks_nothing_removed_when_all_stable() -> None:
    conn = _make_db()
    for fid in range(12):
        _add_player(conn, fid, track_id=1)
    for fid in range(10):
        _add_player(conn, fid, track_id=2)

    removed = _remove_ghost_tracks(conn, "m1", min_frames=10)

    assert removed == 0
    assert _track_ids(conn) == {1, 2}


def test_ghost_tracks_exactly_at_threshold_kept() -> None:
    conn = _make_db()
    for fid in range(10):
        _add_player(conn, fid, track_id=1)

    removed = _remove_ghost_tracks(conn, "m1", min_frames=10)

    assert removed == 0
    assert 1 in _track_ids(conn)


def test_ghost_tracks_empty_table_returns_zero() -> None:
    conn = _make_db()
    removed = _remove_ghost_tracks(conn, "m1", min_frames=10)
    assert removed == 0


# ---------------------------------------------------------------------------
# _interpolate_tracks
# ---------------------------------------------------------------------------


def test_interpolate_fills_single_gap() -> None:
    conn = _make_db()
    # Frames 0 and 2 present, frame 1 missing
    _add_player(conn, 0, track_id=1, x1=100, x2=150)
    _add_player(conn, 2, track_id=1, x1=110, x2=160)

    added = _interpolate_tracks(conn, "m1")

    assert added == 1
    assert _player_count(conn) == 3

    row = conn.execute(
        "SELECT bbox_x1, bbox_x2 FROM players WHERE match_id='m1' AND frame_id=1"
    ).fetchone()
    assert row is not None
    assert row[0] == 105  # midpoint of 100 and 110
    assert row[1] == 155  # midpoint of 150 and 160


def test_interpolate_fills_multi_frame_gap() -> None:
    conn = _make_db()
    _add_player(conn, 0, track_id=1, x1=0, x2=50)
    _add_player(conn, 4, track_id=1, x1=100, x2=150)

    added = _interpolate_tracks(conn, "m1")

    assert added == 3  # frames 1, 2, 3
    rows = conn.execute(
        "SELECT frame_id, bbox_x1 FROM players WHERE match_id='m1' ORDER BY frame_id"
    ).fetchall()
    frame_to_x1 = {r[0]: r[1] for r in rows}
    assert frame_to_x1[1] == 25  # 0 + 1/4 * 100
    assert frame_to_x1[2] == 50  # midpoint
    assert frame_to_x1[3] == 75  # 0 + 3/4 * 100


def test_interpolate_no_gap_returns_zero() -> None:
    conn = _make_db()
    for fid in range(5):
        _add_player(conn, fid, track_id=1)

    added = _interpolate_tracks(conn, "m1")

    assert added == 0
    assert _player_count(conn) == 5


def test_interpolate_court_coords_when_present() -> None:
    conn = _make_db()
    _add_player(conn, 0, track_id=1, court_x=10.0, court_y=5.0)
    _add_player(conn, 2, track_id=1, court_x=20.0, court_y=15.0)

    _interpolate_tracks(conn, "m1")

    row = conn.execute(
        "SELECT court_x, court_y FROM players WHERE match_id='m1' AND frame_id=1"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 15.0) < 0.01
    assert abs(row[1] - 10.0) < 0.01


def test_interpolate_court_coords_null_when_no_calibration() -> None:
    conn = _make_db()
    _add_player(conn, 0, track_id=1, court_x=None, court_y=None)
    _add_player(conn, 2, track_id=1, court_x=None, court_y=None)

    _interpolate_tracks(conn, "m1")

    row = conn.execute(
        "SELECT court_x, court_y FROM players WHERE match_id='m1' AND frame_id=1"
    ).fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None


def test_interpolate_multiple_tracks_independently() -> None:
    conn = _make_db()
    # Track 1: gap at frame 1
    _add_player(conn, 0, track_id=1, x1=0, x2=50)
    _add_player(conn, 2, track_id=1, x1=20, x2=70)
    # Track 2: gap at frame 4
    _add_player(conn, 3, track_id=2, x1=100, x2=150)
    _add_player(conn, 5, track_id=2, x1=120, x2=170)

    added = _interpolate_tracks(conn, "m1")

    assert added == 2  # one gap per track
    assert _track_ids(conn) == {1, 2}


def test_interpolate_single_frame_track_unchanged() -> None:
    conn = _make_db()
    _add_player(conn, 5, track_id=1)

    added = _interpolate_tracks(conn, "m1")

    assert added == 0
    assert _player_count(conn) == 1


def test_interpolate_empty_table_returns_zero() -> None:
    conn = _make_db()
    added = _interpolate_tracks(conn, "m1")
    assert added == 0
