# Production-Ready Text-to-SQL Assistant

A robust, enterprise-grade Text-to-SQL system designed with defense-in-depth safety, AST-based query validation, read-only database confinement, automated error-correction loops, and structured evaluation.

## Tech Stack

- **Language & Runtime:** Python 3.12
- **Package Manager:** `uv`
- **Web Framework:** FastAPI, Uvicorn
- **UI:** Streamlit
- **Database & ORM:** PostgreSQL 16 (Chinook sample database), SQLAlchemy 2
- **LLM Provider:** Gemini (swappable via `call_llm` in `app/generator.py`)
- **Query Parser & Validator:** `sqlglot`
- **Data Validation & Settings:** Pydantic v2, `pydantic-settings`
- **Testing:** `pytest`, `pytest-asyncio`
- **Containerization:** Docker & Docker Compose

---

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py         # Type-safe environment configuration via pydantic-settings
│   ├── schema.py         # Database schema reflection & metadata extraction (Stage 2)
│   ├── generator.py      # LLM query generation with structured output (Stage 2)
│   ├── validator.py      # AST-based SQL validation with sqlglot (Stage 3)
│   ├── executor.py       # Safe execution using read-only role & retry loop (Stage 4)
│   └── api.py            # FastAPI endpoints (Stage 5)
├── eval/
│   ├── __init__.py
│   ├── dataset.json      # Gold query evaluation dataset (Stage 6)
│   └── run_eval.py       # Benchmark evaluation script (Stage 6)
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py  # Interactive Streamlit UI (Stage 5)
├── tests/
│   ├── __init__.py
│   └── test_stage1_setup.py
├── docker/
│   └── init/
│       ├── 01_chinook.sql             # Chinook database schema & seed data
│       └── 02_create_readonly_role.sql # llm_readonly role with 5s timeout & SELECT-only
├── docker-compose.yml     # PostgreSQL service definition
├── pyproject.toml         # uv project specification & dependencies
├── .env.example           # Environment template
└── README.md
```

---

## Security & Defense-in-Depth Principles

1. **Role-Based Least Privilege:**
   - The application enforces a strict separation between admin and read-only database connections.
   - The assistant **never** executes user-generated or LLM-generated SQL queries using the admin credentials.
   - LLM queries execute strictly under the `llm_readonly` role.
2. **Database Engine Hardening:**
   - `llm_readonly` is granted `SELECT` only on tables in the `public` schema.
   - Schema creation and data modification privileges (`CREATE`, `UPDATE`, `INSERT`, `DELETE`, `TRUNCATE`) are explicitly revoked.
   - A hard `statement_timeout = '5s'` is enforced at the database role level in PostgreSQL.
3. **AST Validation:**
   - Queries are parsed into Abstract Syntax Trees using `sqlglot` before execution to prevent SQL injection, multi-statement attacks, and unauthorized table access.
4. **Configuration Integrity:**
   - All connection strings, timeouts, and API keys are strictly configured through environment variables with Pydantic validation.

---

## Quickstart

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2. Install Dependencies

```powershell
uv sync
```

### 3. Environment Configuration

Copy the sample environment file:

```powershell
cp .env.example .env
```

### 4. Start Full Stack with Docker Compose

Start PostgreSQL, FastAPI backend, and Streamlit UI together:

```powershell
docker compose up -d --build
```

- **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### 5. Running Locally with UV

Alternatively, run services locally on your host:

```powershell
# 1. Start database
docker compose up -d postgres

# 2. Start FastAPI server
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# 3. In another terminal, start Streamlit UI
uv run streamlit run ui/streamlit_app.py
```

### 6. Run Test Suite

Run all 115 unit and integration tests:

```powershell
uv run pytest
```
