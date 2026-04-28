# Copilot Agent Instructions for Axon

## Project Overview

**Axon** (`axoniq` on PyPI) is a graph-powered code intelligence engine. It indexes a codebase into a structural knowledge graph using a 12-phase ingestion pipeline backed by [KuzuDB](https://kuzudb.com/) (embedded graph database). The graph is exposed via:

- A **CLI** (`axon`) for developers
- An **MCP server** for AI agents (tools like `axon_query`, `axon_context`, `axon_impact`, etc.)
- An **interactive web dashboard** (FastAPI + React/Sigma.js) at `localhost:8420`

**Package name:** `axoniq` | **Import name:** `axon` | **Python ≥ 3.11**

---

## Repository Structure

```
src/axon/
├── cli/            # Typer CLI entry point (main.py)
├── config/         # Language detection (languages.py), .gitignore loading
├── core/
│   ├── cypher_guard.py      # Validates read-only Cypher before execution
│   ├── diff.py              # Branch comparison via git worktrees
│   ├── embeddings/          # fastembed-based vector generation
│   ├── graph/
│   │   ├── model.py         # NodeLabel, RelType, GraphNode, GraphRelationship
│   │   └── graph.py         # KnowledgeGraph in-memory store
│   ├── ingestion/           # The 12-phase pipeline (pipeline.py orchestrates)
│   │   ├── pipeline.py      # run_pipeline(), reindex_files(), build_graph()
│   │   ├── walker.py        # File discovery (respects .gitignore)
│   │   ├── structure.py     # File/Folder nodes + CONTAINS edges
│   │   ├── parser_phase.py  # tree-sitter parsing → symbol nodes
│   │   ├── imports.py       # IMPORTS edge resolution
│   │   ├── calls.py         # CALLS edges with confidence scores
│   │   ├── heritage.py      # EXTENDS / IMPLEMENTS edges
│   │   ├── types.py         # USES_TYPE edges
│   │   ├── community.py     # Leiden algorithm community detection
│   │   ├── processes.py     # Execution flow tracing from entry points
│   │   ├── dead_code.py     # Multi-pass dead code detection
│   │   ├── coupling.py      # Git history co-change analysis
│   │   └── watcher.py       # watchfiles-based live re-indexing
│   ├── search/              # Hybrid BM25 + vector + fuzzy search
│   └── storage/
│       ├── base.py          # StorageBackend abstract interface
│       └── kuzu_backend.py  # KuzuDB implementation
├── mcp/
│   ├── server.py            # MCP server registration (stdio + HTTP)
│   ├── tools.py             # Tool handlers: query, context, impact, etc.
│   └── resources.py         # MCP resources: overview, dead-code, schema
└── web/
    ├── app.py               # FastAPI app factory
    ├── routes/              # REST API route handlers
    └── frontend/            # React + Sigma.js (TypeScript, built separately)

tests/
├── core/                    # Unit tests for all core modules
├── cli/                     # CLI tests
├── mcp/                     # MCP server/tool tests
├── web/                     # API route tests
└── e2e/                     # End-to-end tests
```

---

## Development Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/) (preferred) or pip.

```bash
# Install all dependencies including dev extras
uv sync --extra dev
# OR with pip
pip install -e ".[dev]"
```

The project uses `uv.lock` for reproducible installs. Always prefer `uv` for running commands.

---

## Key Commands

### Run tests
```bash
uv run pytest                          # All tests
uv run pytest -x -q                    # Stop on first failure, quiet
uv run pytest tests/core/test_pipeline.py -v   # Specific module
uv run pytest -m "not slow"            # Skip slow tests
uv run pytest --tb=short               # Short tracebacks
```

### Lint and format
```bash
uv run ruff check src/ tests/          # Check lint issues
uv run ruff check src/ tests/ --fix    # Auto-fix lint issues
uv run ruff format src/ tests/         # Format code
uv run ruff format --check src/ tests/ # Check formatting without changes
```

Ruff is configured in `pyproject.toml`: `line-length = 100`, rules `E, F, I, N, W`, target `py311`.

### CLI usage (after install)
```bash
axon analyze .          # Index current repo
axon ui                 # Launch web dashboard at localhost:8420
axon serve --watch      # Start MCP server with live reload
axon watch              # Background file watcher only
axon host --watch       # Shared host: UI + MCP
axon diff main..feature # Structural branch comparison
```

---

## The Ingestion Pipeline

`run_pipeline()` in `src/axon/core/ingestion/pipeline.py` is the main entry point. It runs 12 sequential phases:

1. **File Walking** — discovers supported files (`*.py`, `*.ts`, `*.tsx`, `*.js`, `*.jsx`, `*.mjs`, `*.cjs`)
2. **Structure** — `File`/`Folder` nodes + `CONTAINS` edges
3. **Parsing** — tree-sitter AST extraction → `Function`, `Class`, `Method`, `Interface`, `TypeAlias`, `Enum` nodes + `DEFINES` edges
4. **Import Resolution** — `IMPORTS` edges
5. **Call Tracing** — `CALLS` edges with confidence scores (1.0 exact, 0.8 receiver, 0.5 fuzzy)
6. **Heritage** — `EXTENDS` / `IMPLEMENTS` edges
7. **Type Analysis** — `USES_TYPE` edges
8. **Community Detection** — Leiden algorithm, produces `Community` nodes + `MEMBER_OF` edges
9. **Process Detection** — BFS from entry points → `Process` nodes + `STEP_IN_PROCESS` edges
10. **Dead Code Detection** — multi-pass with decorator/protocol/export exemptions
11. **Change Coupling** — git co-change analysis → `COUPLED_WITH` edges
12. **Embeddings (optional, post-load)** — 384-dim vectors via fastembed (BAAI/bge-small-en-v1.5)

---

## Data Model

Defined in `src/axon/core/graph/model.py`:

**Node labels** (`NodeLabel` enum): `FILE`, `FOLDER`, `FUNCTION`, `CLASS`, `METHOD`, `INTERFACE`, `TYPE_ALIAS`, `ENUM`, `COMMUNITY`, `PROCESS`

**Relationship types** (`RelType` enum): `CONTAINS`, `DEFINES`, `CALLS`, `IMPORTS`, `EXTENDS`, `IMPLEMENTS`, `MEMBER_OF`, `STEP_IN_PROCESS`, `USES_TYPE`, `EXPORTS`, `COUPLED_WITH`

**Node ID format:** `{label.value}:{file_path}:{symbol_name}` (deterministic, colon-separated)

---

## Storage Backend

`KuzuBackend` (`src/axon/core/storage/kuzu_backend.py`) stores the graph in `.axon/kuzu/` within the indexed repo. Key methods: `bulk_load(graph)`, `load_graph()`, `add_nodes()`, `add_relationships()`, `remove_nodes_by_file()`, `rebuild_fts_indexes()`, `store_embeddings()`.

Cypher queries run via `kuzu_backend.query()`.

`cypher_guard.py` is used to validate *user-supplied* Cypher (MCP/web) is read-only before execution.
Internal storage writes (bulk load, inserts, deletes) are executed by the storage backend.

---

## MCP Tools

Defined in `src/axon/mcp/tools.py`, registered in `src/axon/mcp/server.py`:

| Tool | Purpose |
|------|---------|
| `axon_query` | Hybrid search: BM25 + vector + fuzzy, results grouped by execution flow |
| `axon_context` | 360° symbol view: callers, callees, type refs, community, dead code status |
| `axon_impact` | Blast radius grouped by depth (direct/indirect/transitive) |
| `axon_dead_code` | All unreachable symbols grouped by file |
| `axon_detect_changes` | Map git diff → affected symbols and flows |
| `axon_list_repos` | All indexed repos with stats |
| `axon_cypher` | Read-only Cypher against the knowledge graph |
| `axon_communities` | Community listing |
| `axon_coupling` | Change coupling data |
| `axon_call_path` | Call path between two symbols |
| `axon_cycles` | Circular dependency detection |
| `axon_explain` | Natural language graph explanation |
| `axon_file_context` | File-level context |
| `axon_review_risk` | Change review risk assessment |
| `axon_test_impact` | Which tests are affected by a change |

---

## Adding a New Language Parser

1. Create `src/axon/core/ingestion/languages/<lang>.py` following the pattern in `python.py` or `typescript.py`
2. Register the language in `src/axon/core/ingestion/languages/__init__.py`
3. Add the tree-sitter grammar dependency to `pyproject.toml`
4. Add the file extension mapping in `src/axon/config/languages.py` → `SUPPORTED_EXTENSIONS`
5. Write tests in `tests/core/test_parser_<lang>.py`
6. Verify integration: `axon analyze <test-repo>` should index the new language

---

## Code Style Conventions

- **Type hints** on all public APIs; skip on obvious locals
- **`_extract_*` method naming** pattern in parsers
- No unnecessary abstractions: three similar lines > a premature helper
- Comments only where logic is non-obvious
- Follow `from __future__ import annotations` at top of modules
- Private helpers use leading underscore: `_run_embedding_phase`, `_write_collected_edges`
- Dataclasses for data transfer objects (`PipelineResult`, `GraphNode`, `GraphRelationship`)

---

## Contribution Requirements

**IMPORTANT — read before opening any PR:**

1. Every PR must reference a maintainer-approved issue with `Closes #N` or `Fixes #N` in the body. The CI check (`pr-check.yml`) **will fail** your PR automatically if this is missing (exception: trivial fixes ≤ 5 changed lines).
2. One logical change per PR. Do not bundle unrelated changes.
3. Tests are mandatory: bug fix → regression test; new feature → coverage.
4. Pass lint: `ruff check src/ tests/ --fix && ruff format src/ tests/`
5. Do NOT add project-config files like `CLAUDE.md`, `.cursorrules`, etc. — they will be closed without review.
6. Use the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.

Commit message format: `<type>: <short description>` (types: `fix`, `feat`, `test`, `refactor`, `docs`, `chore`). Subject ≤ 72 characters.

---

## CI/CD

- **`pr-check.yml`** — runs on every PR to verify a linked issue reference exists (skips for repo owner `harshkedia177`)
- **`publish.yml`** — triggered on `v*` tags; runs tests then builds and publishes to PyPI. Frontend is built with `npm ci && npm run build` in `src/axon/web/frontend/` before packaging.

---

## Frontend

The web dashboard lives in `src/axon/web/frontend/` (React + TypeScript + Sigma.js + Vite). It is pre-built — the compiled `dist/` is included in the wheel. To work on it:

```bash
cd src/axon/web/frontend
npm ci
npm run dev   # Vite HMR on :5173 (use `axon ui --dev` to proxy)
npm run build # Production build into dist/
```

Node.js 20+ is required for the frontend build.

---

## Known Patterns and Gotchas

- The `.axon/` directory (created in the target repo, not this repo) holds the KuzuDB database. It is excluded from version control via `.gitignore`.
- `uv.lock` is committed and must stay in sync with `pyproject.toml`. Run `uv sync` after any dependency change.
- `fastembed` downloads model weights on first use (~100MB). In CI or offline environments, this may be slow or fail — the pipeline gracefully degrades to FTS-only search when embeddings fail.
- The `skills-lock.json` file at the repo root is for Copilot agent skill configuration; do not modify it.
- Cypher queries against KuzuDB use KuzuDB's Cypher dialect, not Neo4j's. Some Neo4j features are unavailable.
- `asyncio_mode = "auto"` is set in pytest config — all async tests work without explicit `@pytest.mark.asyncio`.
