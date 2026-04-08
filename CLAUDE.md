# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL4CO Display is a Flask-based web platform for interactive reinforcement learning training and visualization, built on top of the [RL4CO](https://github.com/ai4co/rl4co) library. It supports multi-user training of combinatorial optimization models (TSP, CVRP, etc.) with real-time progress streaming via SSE.

## Setup

**Prerequisites**: Python 3.8+, MySQL 8.0+

```bash
# Install dependencies
pip install -r requirements.txt          # production
pip install -r requirements-dev.txt      # development + testing

# Configure database
cp config/config.example.py config/config.py
# Edit config/config.py with MySQL credentials

# Initialize DB schema
mysql -u root -p flaskdemo_user < config/database_init_with_auth.sql

# Run the app
python app.py
# or
bash scripts/start.sh
```

App runs at `http://localhost:5000`. The `config/config.py` file is gitignored — never commit it.

## Common Commands

```bash
# Run tests
pytest tests/

# Run a single test file
pytest tests/test_parse_dataset.py -v

# Run with coverage
pytest --cov=. tests/

# Lint
flake8 .
pylint *.py modules/

# Format
black .
isort .
```

## Architecture

### Flask Blueprint Structure

`app.py` is the entry point — it creates the Flask app, manages DB connections, holds global training state (`training_status`, `training_queues` dicts), and registers Blueprints:

| Module | Blueprint | Responsibilities |
|---|---|---|
| `app_auth.py` | `auth_bp` | Login, register, logout, session |
| `app_pages.py` | `pages_bp` | Page rendering (index, benchmark, profile) |
| `app_stats.py` | `stats_bp` | Statistics APIs (with 5-min cache) |
| `app_compat.py` | `compat_bp` | Model/problem compatibility validation |
| `app_training.py` | `training_bp` | Training start, SSE progress stream |
| `app_files.py` | `files_bp` | Dataset upload/management, checkpoint download |
| `app_gpu.py` | `gpu_bp` | GPU status and allocation |

Blueprints receive shared state via init functions (e.g., `init_training_globals()`), not via imports — this avoids circular imports.

### Dependency Injection Pattern

`app.py` injects DB accessors into `auth_module` via `auth_module.init_db_accessors(get_db, ...)`. Similarly, each blueprint receives globals at startup. This is the established pattern — follow it when adding new blueprints.

### Database Access

- **Request context**: Use `get_db()` → returns `g.db` (auto-closed on teardown)
- **Background threads**: Use `get_background_db()` → creates an independent connection (must be closed manually)
- Manager classes: `UserManager`, `TrainingSessionManager`, `FileManager` (from `auth_module.py`) wrap DB operations

### Training Flow

1. `POST /api/start_training` validates config, creates a DB session record, spawns a background thread
2. Background thread calls `real_rl4co_training()` (from `modules/rl_training/`) or `simulate_training()` as fallback
3. Progress messages go into a `Queue` keyed by `session_id`
4. `GET /api/training_progress/<session_id>` SSE endpoint reads from the queue and streams to the client
5. Files (plots, GIFs, checkpoints) are saved under `static/model_plots/user_<id>/` and `checkpoints/user_<id>/`

### modules/ Structure

```
modules/
├── algorithms/       # RL algorithm implementations
├── policies/         # Policy model definitions
├── problems/         # Problem env wrappers (tsp.py, cvrp.py, etc.)
│   └── base_problem.py
├── rl_training/      # Per-problem trainer classes
│   ├── base_trainer.py
│   ├── tsp_trainer.py, cvrp_trainer.py, ...  # one per problem type
│   ├── training_functions.py  # dispatches to correct trainer
│   └── visualizations/
└── compatibility.py  # validates model+problem combinations
```

To add a new problem type, follow `docs/ADD_NEW_PROBLEM_TYPE_GUIDE.md`.

### RL4CO Availability

The app detects at startup whether `rl4co` is installed (`RL4CO_AVAILABLE` flag). If not, `simulate_training()` is used as a demo fallback. Real training requires `rl4co>=0.4.0`, `torch>=2.0.0`, and `lightning>=2.0.0`.

### File Storage Layout

- `static/model_plots/user_<id>/` — training result images and GIFs (served as static files)
- `checkpoints/user_<id>/` — model checkpoint files
- `datasets/user_<id>/` — user-uploaded dataset files
- `logs/` — rotating log files (`rl4co_display_YYYYMMDD.log`)

### Error Codes

Defined in `logging_config.py`: `1xxx` auth, `2xxx` params, `3xxx` resources, `4xxx` operations, `5xxx` system.

### Key Global State in app.py

- `training_status`: dict mapping `session_id → status dict`
- `training_queues`: dict mapping `session_id → Queue`
- A background reaper thread cleans up completed sessions after 30 minutes
- `api_cache` (SimpleCache): caches stats API responses for 5 minutes
