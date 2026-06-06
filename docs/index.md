# WELS — Handball Analytics Platform

A monorepo for a handball analytics platform that helps trainers analyze matches, ingest video data, and predict actions to adjust strategy.

## Quick Links

| | |
|---|---|
| [Getting Started](getting-started.md) | Set up your local environment |
| [Architecture](architecture.md) | System overview and design decisions |
| [Ingestion — Running](ingestion/running.md) | Process a match video end-to-end |
| [Storage](storage.md) | DuckDB schema and common queries |
| [ML — Training](ml/training.md) | Annotate and train the action predictor |
| [Contributing](contributing.md) | Code style, testing, and PR workflow |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Web framework | FastAPI + Uvicorn |
| Templating | Jinja2 + HTMX |
| Data models | Pydantic v2 |
| Package manager | uv |
| Object detection | YOLO26 (ultralytics) + BoT-SORT |
| ML framework | PyTorch + PyTorch Geometric |
| Storage | DuckDB |
| Linter / formatter | ruff |
| Type checker | ty |
| Task runner | moon |
