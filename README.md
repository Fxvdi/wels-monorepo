# WELS — Handball Analytics Platform

A monorepo for a handball analytics platform. It processes match video with computer vision, stores structured match data in DuckDB, and predicts player actions with a GCN + LSTM model — surfacing insights via a web dashboard.

## How it works

```
match.mp4  →  [wels-ingest]  →  data/matches.duckdb  ──────────────────▶  Backend API  →  Frontend
                                        ↑                                        ↑
                               [wels-train]  ←  action labels                   │
                                        ↓                                        │
                               data/models/action_best.pt                        │
                                        │                                        │
                                        ▼                                        │
                                [wels-score]  ─── formations, possession ────────┘
                                              ─── action_predictions ────────────┘
```

1. **Ingest** — run `wels-ingest` on a match video; the CV pipeline detects players, estimates poses, classifies teams, and maps positions to court coordinates
2. **Train** — annotate action labels in DuckDB, then run `wels-train` to produce a trained checkpoint
3. **Score** — run `wels-score` to pre-compute action predictions, formation labels, and possession phases into DuckDB (one-time batch job per match)
4. **Analyse** — the backend reads pre-computed results from DuckDB; no on-the-fly inference at request time

## Repository structure

```
wels-monorepo/
├── packages/
│   ├── backend/        # FastAPI REST API (port 8000)
│   ├── frontend/       # React + Vite SPA (port 3000)
│   ├── ingestion/      # CV pipeline: video → DuckDB
│   └── ml/             # GCN + LSTM model: train, score, analyse
├── data/               # Runtime data — not committed
│   ├── matches.duckdb  # All match data
│   ├── models/         # YOLO weights + trained checkpoints
│   └── videos/         # Recommended location for match recordings
├── docs/               # MkDocs documentation
├── .moon/              # Moon workspace config
├── Makefile
├── ruff.toml           # Shared lint/format config
└── ty.toml             # Shared type-check config
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| [uv](https://docs.astral.sh/uv/) | Manages Python venvs and installs Python 3.12 automatically. |
| Node + pnpm | Installed automatically by moon's toolchain — see `.moon/toolchain.yml`. |
| NVIDIA GPU (CUDA 12.1+) | Required for video ingestion and ML training. CPU fallback available but slow. |

No system Python is required — uv downloads and manages its own CPython builds.

## Quick start

**Linux / WSL (fresh machine)**
```bash
sh bootstrap.sh          # installs make, xz-utils, uv, and Python 3.12 — no system Python needed
source ~/.local/bin/env  # activate uv in the current shell (only needed if uv was just installed)
make setup               # venvs, deps, moon, pre-commit hooks
make dev                 # backend + frontend + docs in parallel
```

if this error appears:
```pnpm: command not found```

do:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts
nvm use --lts
npm install -g pnpm
pnpm -v
```


**macOS**
```bash
# Install uv if not already present:
curl -LsSf https://astral.sh/uv/install.sh | sh
make setup
make dev
```

**Windows**
```powershell
.\Make.ps1 setup
.\Make.ps1 dev
```

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Frontend | http://localhost:3000 |
| Docs | http://localhost:8080 |

## Processing a match video

```bash
cd packages/ingestion

# Basic run
uv run wels-ingest data/videos/match.mp4 2026-04-13_wels_vs_linz

# With court calibration (enables real-world metre coordinates)
uv run wels-ingest data/videos/match.mp4 2026-04-13_wels_vs_linz \
    --calibration data/court_cal.json

# CPU-only machine
uv run wels-ingest data/videos/match.mp4 2026-04-13_wels_vs_linz --device cpu
```

Install the CV runtime deps on the machine that processes video:
```bash
cd packages/ingestion && uv sync --all-extras
```

## Training and scoring

```bash
# Train the model
cd packages/ml
uv sync
uv run wels-train   # reads data/matches.duckdb, writes data/models/action_best.pt

# Pre-compute predictions for a match (run after each new checkpoint)
uv run wels-score 2026-04-13_wels_vs_linz
uv run wels-score 2026-04-13_wels_vs_linz \
    --checkpoint data/models/action_predictor_best.pt
```

`wels-score` writes three tables to DuckDB: `action_predictions`, `formations`, and
`possession_phases`. The backend reads from these — no inference at request time.

See [ML — Training](docs/ml/training.md) for the full annotation, training, and scoring workflow.

## Commands

```bash
make lint             # ruff check (all packages)
make format           # ruff format (all packages)
make typecheck        # ty (all packages)
make test             # pytest — non-integration tests (all packages)
make test-integration # integration tests (require GPU + video files)
make docs             # docs server at http://localhost:8080
```

Moon (parallel + cached):
```bash
./tools/moon run :lint
./tools/moon run :test
./tools/moon run :typecheck
```

## Adding dependencies

Python packages (`backend`, `ingestion`, `ml`):

```bash
cd packages/<name> && uv add <package>        # runtime dep
cd packages/<name> && uv add --dev <package>  # dev dep

# Ingestion CV deps (torch, ultralytics) are in the [cv] optional group:
cd packages/ingestion && uv add --optional cv <package>
```

Frontend (`packages/frontend`, pnpm):

```bash
cd packages/frontend && pnpm add <package>     # runtime dep
cd packages/frontend && pnpm add -D <package>  # dev dep
```

## Tooling

| Tool | Purpose | Config |
|------|---------|--------|
| [uv](https://docs.astral.sh/uv/) | Python package management, venvs | `pyproject.toml` per Python package |
| [pnpm](https://pnpm.io/) | Node package management (frontend) | `packages/frontend/package.json` |
| [ruff](https://docs.astral.sh/ruff/) | Python linting + formatting | `ruff.toml` |
| [ty](https://github.com/astral-sh/ty) | Python type checking | `ty.toml` |
| [eslint](https://eslint.org/) + [typescript-eslint](https://typescript-eslint.io/) | Frontend linting | `packages/frontend/eslint.config.js` |
| [tsc](https://www.typescriptlang.org/) | Frontend type checking | `packages/frontend/tsconfig*.json` |
| [Vite](https://vitejs.dev/) + [React](https://react.dev/) + [Tailwind v4](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/) | Frontend stack | `packages/frontend/` |
| [moon](https://moonrepo.dev/) | Parallel task runner + caching, installs Node + pnpm | `.moon/`, `moon.yml` per package |
| [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) | Documentation | `mkdocs.yml` |
| [pre-commit](https://pre-commit.com/) | Git hooks | `.pre-commit-config.yaml` |
| [GitHub Actions](https://github.com/features/actions) | CI | `.github/workflows/tests.yml` |
 