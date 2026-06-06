# CLAUDE.md — WELS Monorepo

## Project Overview

**WELS** is a handball analytics platform for trainers and analysts. It ingests match videos through a computer vision pipeline, stores structured frame-by-frame data in DuckDB, trains a GCN + LSTM model to predict player actions, and surfaces insights via a web dashboard.

## Architecture

```
Browser ──(HTTP + JSON)──> Frontend (port 3000, Vite + React)
                                 │
                            fetch() / ky / axios
                                 │
                                 ▼
                           Backend API (port 8000)
                                 │
                           data/matches.duckdb
                                 ▲
                          ┌──────┴───────┐
                   wels-ingest        wels-score
                  (CV pipeline)    (batch scoring)
                          │              │
                   Match video    trained *.pt
                     (MP4)        checkpoint
```

**Package boundaries** — packages communicate through DuckDB (data) and HTTP (services), never by importing each other:

| Package | Reads | Writes |
|---------|-------|--------|
| `ingestion` | video file | `data/matches.duckdb` (matches, frames, players, ball) |
| `ml` | `data/matches.duckdb` | `data/matches.duckdb` (action_predictions, formations, possession_phases) + `data/models/*.pt` |
| `backend` | `data/matches.duckdb` | HTTP responses |
| `frontend` | backend HTTP | HTML responses |

**Two-step data pipeline:**
1. `wels-ingest <video> <match_id>` — CV pipeline writes raw tracking data
2. `wels-score <match_id>` — ML scoring job reads tracking data and writes pre-computed predictions

## Repository Layout

```
wels-monorepo/
├── packages/
│   ├── backend/          # FastAPI REST API (port 8000)
│   │   └── src/backend/
│   │       ├── app.py        # FastAPI app factory
│   │       ├── config.py     # pydantic-settings (WELS_ prefix)
│   │       ├── models.py     # Pydantic domain models
│   │       └── routes/       # API route modules
│   ├── frontend/         # React + Vite SPA (port 3000)
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   ├── tsconfig*.json
│   │   ├── eslint.config.js
│   │   ├── moon.yml            # language: typescript; setup/run/lint/typecheck/build
│   │   ├── index.html
│   │   └── src/
│   │       ├── main.tsx
│   │       ├── app/            # App.tsx + view components + shadcn/ui primitives
│   │       ├── imports/        # Logo + static assets
│   │       └── styles/         # tailwind.css, theme.css, wels.css, fonts.css, index.css
│   ├── ingestion/        # CV pipeline: video → DuckDB
│   │   └── src/ingestion/
│   │       ├── pipeline/     # detection, pose, team, court (pure functions)
│   │       ├── storage/      # DuckDB schema + FrameWriter
│   │       ├── types.py      # shared dataclasses (BoundingBox, FrameState, …)
│   │       ├── orchestrator.py
│   │       ├── config.py
│   │       └── cli.py        # wels-ingest entry point
│   └── ml/               # GCN + LSTM action predictor + batch scoring
│       └── src/ml/
│           ├── data/         # DuckDB queries, graph construction, Dataset
│           ├── models/       # ActionPredictor (GCN + LSTM)
│           ├── training/     # training loop + evaluation
│           ├── analysis/     # formation classifier, possession phase detector
│           ├── storage/      # ML output tables schema (action_predictions, etc.)
│           ├── scoring.py    # MatchScorer: writes all 3 output tables
│           ├── inference.py  # load checkpoint → predict
│           ├── cli.py        # wels-score entry point
│           └── config.py
├── data/                 # runtime data — not committed
│   ├── matches.duckdb    # all match data
│   ├── models/           # YOLO weights + trained checkpoints
│   └── videos/           # recommended location for match recordings
├── docs/                 # MkDocs documentation
├── .moon/                # Moon workspace config
├── .github/
│   ├── workflows/tests.yml      # CI: lint + typecheck + test (all 4 packages)
│   └── instructions/python.instructions.md
├── Makefile
├── ruff.toml             # Shared lint/format config
└── ty.toml               # Shared type-check config (all rules as errors)
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Backend / ingestion / ml language | Python 3.12+ |
| Frontend language | TypeScript 5 (Node 20) |
| API framework | FastAPI + Uvicorn |
| Frontend stack | React 18 + Vite 6 + Tailwind CSS v4 + shadcn/ui (Radix) |
| Data models | Pydantic v2 |
| Config | pydantic-settings (`WELS_` prefix) |
| HTTP client (Python) | httpx (async) |
| Storage | **DuckDB** (`data/matches.duckdb`) |
| Object detection | **YOLO26** (ultralytics) + BoT-SORT |
| ML framework | **PyTorch** + **PyTorch Geometric** |
| Python package manager | **uv** (not pip, not Poetry) |
| Node package manager | **pnpm** (installed by moon's node toolchain) |
| Python build backend | hatchling |
| Task runner | **moon** (`./tools/moon`), also provisions Node + pnpm |
| Python linter/formatter | **ruff** |
| Frontend linter | **eslint** + typescript-eslint |
| Python type checker | **ty** (all rules as errors) |
| Frontend type checker | **tsc** (`tsc -b --noEmit`, strict) |
| Test framework (Python) | pytest + pytest-asyncio |
| Docs | MkDocs Material |
| CI | GitHub Actions |

## Common Commands

```bash
# First-time setup
make setup          # venvs (Python), pnpm install (frontend), moon, pre-commit hooks

# Development
make dev            # backend + frontend + docs (parallel)
make run-backend    # → http://localhost:8000
make run-frontend   # → http://localhost:3000 (Vite dev server)
make build-frontend # production build to packages/frontend/dist/
make docs           # → http://localhost:8080

# Code quality
make lint           # ruff (Python) + eslint (frontend)
make format         # ruff format (Python only)
make typecheck      # ty (Python) + tsc (frontend)
make test           # pytest (Python packages; frontend has no tests yet)
make test-integration

# Video ingestion (requires GPU or --device cpu)
cd packages/ingestion
uv run wels-ingest <video> <match_id> [--calibration court.json]

# ML training
cd packages/ml
uv run wels-train

# Batch scoring (writes pre-computed predictions to DuckDB)
cd packages/ml
uv run wels-score <match_id>
uv run wels-score <match_id> --checkpoint data/models/action_predictor_best.pt

# Moon (parallel + cached)
./tools/moon run :lint
./tools/moon run :typecheck
./tools/moon run :test
```

## Development Conventions

### Adding a new backend route
1. Create a module in `packages/backend/src/backend/routes/`.
2. Define an `APIRouter` and register it in `app.py`.
3. Use Pydantic v2 models in `models.py` or a local `schemas.py`.

### Adding a new frontend page / view
1. Add a React component under `packages/frontend/src/app/components/` (e.g. `TeamStats.tsx`).
2. Compose with shadcn/ui primitives from `src/app/components/ui/` (`Button`, `Card`, `Dialog`, …). Import with `@/` alias: `import { Button } from "@/app/components/ui/button"`.
3. Wire into `App.tsx` — either extend the `AppState` union or introduce `react-router` (already installed).
4. Use Tailwind utility classes for layout/spacing. Reach for the WELS semantic classes in `src/styles/wels.css` (`.nav`, `.page-header`, `.card`, `.btn-primary`) when you want WELS-branded chrome.
5. Never hard-code colors — use the CSS variables in `src/styles/theme.css` (`--primary`, `--foreground`, `--color-wels-navy`, …). Tailwind color utilities like `text-primary` resolve through those variables.

### Working on the ingestion pipeline
- `pipeline/` modules are **pure functions** — no file I/O, no DB calls, no global state inside them.
- All data passed between stages uses the typed dataclasses in `types.py`.
- The four pipeline stubs (`detection`, `pose`, `team`, `court`) have `NotImplementedError` bodies; port them from `CV-POC-Wels/pipeline/` one at a time.
- `torch` and `ultralytics` are in the `[cv]` optional group — install with `uv sync --all-extras` on GPU machines.

### Working on the ML package
- The ml package has **two modes**: training (reads DuckDB, writes checkpoint) and scoring (reads DuckDB, writes DuckDB).
- `data/features.py` contains all DuckDB read queries; keep SQL out of other modules.
- `data/graphs.py` converts frame dicts to PyG `Data` objects — no I/O.
- `models/action.py` is a plain `nn.Module` — no DuckDB, no file I/O.
- `scoring.py` / `MatchScorer` is the only place that writes to DuckDB (3 tables: `action_predictions`, `formations`, `possession_phases`).
- `analysis/formation.py` and `analysis/possession.py` are pure functions — no DB, no GPU. Test them without any setup.
- `storage/schema.py` defines the ML output tables. `connect()` requires the DB to already exist (run `wels-ingest` first).
- Use `WELS_DEVICE=cpu` on machines without a GPU.
- After training a new checkpoint, run `wels-score` to refresh pre-computed predictions in DuckDB.

### Adding a dependency
Python packages (`backend`, `ingestion`, `ml`):
```bash
cd packages/<name> && uv add <package>
```
For ingestion CV deps (torch, ultralytics): `uv add --optional cv <package>`.

Frontend (`packages/frontend`, pnpm):
```bash
cd packages/frontend && pnpm add <package>        # runtime
cd packages/frontend && pnpm add -D <package>     # dev
```

Or use the `/add-dep` skill.

### Configuration
Python packages use `pydantic-settings` with `WELS_` prefix:
```bash
export WELS_DEVICE=cuda
export WELS_DUCKDB_PATH=data/matches.duckdb
```

The frontend reads env vars through Vite with the `VITE_` prefix (available at `import.meta.env.VITE_*`). Put them in `packages/frontend/.env.local` (gitignored):
```bash
VITE_BACKEND_URL=http://localhost:8000
```

### Testing
- Non-integration tests require no GPU and no video files — run anywhere.
- Integration tests are marked `@pytest.mark.integration` and excluded from CI.
- Ingestion unit tests (types, storage) only need `duckdb` + `numpy`; no torch.
- ML unit tests (graph construction, model shape) need torch but no GPU.

## CI

GitHub Actions runs on every PR across all four packages:
1. Lint — ruff for Python, eslint for frontend
2. Type check — ty (Python, GitHub annotation format) + tsc (frontend)
3. Tests — `pytest -m "not integration"` for Python packages; the frontend gate is `pnpm build` (Vite + tsc as a single compile check) since there are no unit tests yet
4. Posts a pass/fail summary comment on the PR

Ingestion installs without the `[cv]` extras (no torch/ultralytics on CI).
ML installs normally; torch runs on CPU on the GitHub runner and is cached between runs.
Frontend installs with `pnpm install --frozen-lockfile`; the pnpm store is cached between runs via `actions/setup-node`.

## Available Claude Skills

Custom skills are defined under `.claude/commands/` and invocable as slash commands:

| Skill | Purpose |
|-------|---------|
| `/new-package <name>` | Scaffold a new Python package in the monorepo |
| `/new-component <desc>` | Add a Jinja2 macro component to the frontend |
| `/new-route <desc>` | Scaffold a new route (backend or frontend) |
| `/add-dep <dep> to <pkg>` | Add a dependency via uv |
| `/check` | Run the full quality suite (lint + typecheck + test) |
| `/port-pipeline <module>` | Port a pipeline stub from CV-POC into the ingestion package |
| `/train` | Run wels-train and report results, then prompt to run wels-score |

## Key Design Decisions

- **Package boundaries via DuckDB + HTTP**: packages never import each other. `ingestion` writes raw tracking data to DuckDB; `ml` reads that data and writes pre-computed predictions back; `backend` serves all tables via HTTP. No on-the-fly inference at request time.
- **Pipeline modules are pure**: `pipeline/` functions take typed inputs, return typed outputs — no side effects. This makes them testable without a GPU.
- **torch/ultralytics are optional in ingestion**: unit tests run without GPU deps; the `[cv]` group is only installed on machines that process video.
- **Ingestion triggered as FastAPI BackgroundTask**: no separate worker queue. Sufficient for batch processing; can be extracted later if needed.
- **Frontend and backend ship independently**: separate runtimes (Node vs Python), separate ports, separate dependency trees. Either side can be swapped without touching the other.
- **React + Vite + Tailwind v4 + shadcn/ui on the frontend**: typed, accessible primitives; WELS brand tokens live in CSS variables and drive both shadcn components and WELS semantic classes (`.nav`, `.card`, `.page-header` from `src/styles/wels.css`).
- **uv + pnpm for packages**: `uv` for Python, `pnpm` for the frontend — both are deterministic, both are fast.
- **moon for task caching and toolchain provisioning**: tasks only re-run when inputs change; moon also installs Node + pnpm from `.moon/toolchain.yml` so contributors don't need to install them manually.
- **ruff + ty, not black/mypy**: both written in Rust; order-of-magnitude faster.
- **eslint + tsc for the frontend**: flat eslint config with `typescript-eslint` + `react-hooks` + `react-refresh`; `tsc -b --noEmit` for type-checking, Vite for bundling.
