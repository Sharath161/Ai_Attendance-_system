.PHONY: setup models run worker test lint clean docker-up docker-down docker-logs

# ── Local development ─────────────────────────────────────────────────────────

setup:
	pip install -r requirements-dev.txt
	python -m worker.download_models
	@echo ""
	@echo "Setup complete. Run 'make run' to start the server."

models:
	python -m worker.download_models

run:
	python start_server.py

worker:
	python -m worker.runner

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	python -m py_compile api/main.py core/models.py core/config.py worker/model_adapter.py worker/registration_updater.py
	@echo "Syntax OK"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-up:
	docker-compose up --build -d
	@echo "API running at http://localhost:8000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api worker

docker-models:
	docker-compose exec api python -m worker.download_models

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.pyo" -delete 2>/dev/null; true

reset-db:
	@echo "WARNING: deletes all attendance data"
	rm -f work/attendance.db work/attendance.db-shm work/attendance.db-wal
