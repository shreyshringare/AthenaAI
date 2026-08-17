.PHONY: install dev lint type test test-unit test-integration up down logs demo clean

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	uv pip install -e ".[dev]"

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/

type:
	uv run mypy src/

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v --tb=short

test-unit:
	uv run pytest tests/unit/ -v --tb=short

test-integration:
	uv run pytest tests/integration/ -v --tb=short

# ── Docker ─────────────────────────────────────────────────────────────────────
up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

# ── Demo ───────────────────────────────────────────────────────────────────────
# Run the API locally (no Docker) using MockModel
dev:
	uv run uvicorn athenai.gateway.app:app --reload --port 8000

# Quick smoke test — works with both `make dev` and `make up`
demo:
	@echo "=== Health check ==="
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@echo "=== Chat (mock mode) ==="
	@curl -s -X POST http://localhost:8000/v1/chat \
	  -H "Content-Type: application/json" \
	  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}' \
	  | python -m json.tool
	@echo ""
	@echo "=== Agent run with calculator tool ==="
	@curl -s -X POST http://localhost:8000/v1/agents/run \
	  -H "Content-Type: application/json" \
	  -d '{"task":"Calculate 42 * 17 + 100","tools":["calculator"]}' \
	  | python -m json.tool
	@echo ""
	@echo "=== Metrics (Prometheus) ==="
	@curl -s http://localhost:8000/metrics | head -20

demo-stream:
	@echo "=== Streaming chat ==="
	@curl -s -N -X POST http://localhost:8000/v1/chat/stream \
	  -H "Content-Type: application/json" \
	  -d '{"messages":[{"role":"user","content":"Tell me a joke"}],"stream":true}'

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
