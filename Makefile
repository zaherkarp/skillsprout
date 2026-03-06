.PHONY: help build up down restart logs logs-api logs-celery logs-db shell-api shell-db test seed clean prune migrate health dev-local check-docker

# Colors for terminal output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RESET  := \033[0m

help: ## Show this help message
	@echo "$(GREEN)SkillSprout - Docker Commands$(RESET)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

check-docker:
	@command -v docker >/dev/null 2>&1 || { \
		echo "$(YELLOW)ERROR: Docker is not installed or not in PATH.$(RESET)"; \
		echo ""; \
		echo "Options:"; \
		echo "  1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"; \
		echo "  2. Run locally without Docker: $(BLUE)make dev-local$(RESET)"; \
		echo ""; \
		exit 1; \
	}

build: check-docker ## Build Docker images
	@echo "$(YELLOW)Building Docker images...$(RESET)"
	docker compose build

up: check-docker ## Start all services
	@echo "$(GREEN)Starting all services...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)✓ Services started!$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available at:$(RESET)"
	@echo "  - Web UI:    http://localhost:8000"
	@echo "  - API Docs:  http://localhost:8000/api/v1/docs"
	@echo "  - Health:    http://localhost:8000/api/v1/health"
	@echo ""
	@echo "$(YELLOW)Run 'make logs' to see output$(RESET)"

down: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(RESET)"
	docker compose down
	@echo "$(GREEN)✓ Services stopped$(RESET)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(RESET)"
	docker compose restart
	@echo "$(GREEN)✓ Services restarted$(RESET)"

logs: ## Show logs from all services
	docker compose logs -f

logs-api: ## Show logs from API service only
	docker compose logs -f api

logs-celery: ## Show logs from Celery worker only
	docker compose logs -f celery-worker

logs-db: ## Show logs from database only
	docker compose logs -f db

shell-api: ## Open shell in API container
	docker compose exec api /bin/bash

shell-db: ## Open PostgreSQL shell
	docker compose exec db psql -U skillsprout -d skillsprout

test: ## Run tests in isolated environment
	@echo "$(YELLOW)Running tests...$(RESET)"
	docker compose --profile test run --rm test
	@echo "$(GREEN)✓ Tests complete$(RESET)"

seed: ## Seed demo data
	@echo "$(YELLOW)Seeding demo data...$(RESET)"
	docker compose exec api python scripts/seed_demo.py
	@echo "$(GREEN)✓ Demo data seeded$(RESET)"

migrate: ## Run database migrations
	@echo "$(YELLOW)Running database migrations...$(RESET)"
	docker compose exec api alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(RESET)"

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	@echo "$(YELLOW)Creating new migration...$(RESET)"
	docker compose exec api alembic revision --autogenerate -m "$(MSG)"
	@echo "$(GREEN)✓ Migration created$(RESET)"

health: ## Check health of all services
	@echo "$(YELLOW)Checking service health...$(RESET)"
	@echo ""
	@echo "$(BLUE)Database:$(RESET)"
	@docker compose exec db pg_isready -U skillsprout && echo "$(GREEN)✓ Healthy$(RESET)" || echo "$(RED)✗ Unhealthy$(RESET)"
	@echo ""
	@echo "$(BLUE)Redis:$(RESET)"
	@docker compose exec redis redis-cli ping && echo "$(GREEN)✓ Healthy$(RESET)" || echo "$(RED)✗ Unhealthy$(RESET)"
	@echo ""
	@echo "$(BLUE)API:$(RESET)"
	@curl -sf http://localhost:8000/api/v1/health > /dev/null && echo "$(GREEN)✓ Healthy$(RESET)" || echo "$(RED)✗ Unhealthy$(RESET)"
	@echo ""

ps: ## Show running containers
	docker compose ps

stats: ## Show container resource usage
	docker stats $$(docker compose ps -q)

clean: ## Stop services and remove volumes (keeps images)
	@echo "$(YELLOW)Stopping services and removing volumes...$(RESET)"
	docker compose down -v
	@echo "$(GREEN)✓ Cleaned up$(RESET)"

prune: ## Remove all Docker resources (including images)
	@echo "$(YELLOW)WARNING: This will remove all containers, volumes, and images$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v --rmi all; \
		echo "$(GREEN)✓ Pruned$(RESET)"; \
	else \
		echo "$(YELLOW)Cancelled$(RESET)"; \
	fi

dev: build up seed ## Full dev environment setup (build, start, seed)
	@echo ""
	@echo "$(GREEN)========================================$(RESET)"
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"
	@echo "$(GREEN)========================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available at:$(RESET)"
	@echo "  - Web UI:    http://localhost:8000"
	@echo "  - API Docs:  http://localhost:8000/api/v1/docs"
	@echo "  - Health:    http://localhost:8000/api/v1/health"
	@echo ""
	@echo "$(YELLOW)Useful commands:$(RESET)"
	@echo "  - make logs       - View all logs"
	@echo "  - make logs-api   - View API logs"
	@echo "  - make shell-api  - Open API container shell"
	@echo "  - make test       - Run tests"
	@echo "  - make down       - Stop all services"
	@echo ""

dev-local: ## Run locally without Docker (SQLite, no Redis/Celery)
	@echo "$(GREEN)Starting SkillSprout locally (no Docker required)...$(RESET)"
	@echo ""
	@command -v python3 >/dev/null 2>&1 || { echo "$(YELLOW)ERROR: python3 is not installed.$(RESET)"; exit 1; }
	@if [ ! -d ".venv" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(RESET)"; \
		python3 -m venv .venv; \
	fi
	@echo "$(YELLOW)Installing dependencies...$(RESET)"
	@. .venv/bin/activate && pip install -r requirements.txt --quiet
	@echo "$(YELLOW)Starting FastAPI server...$(RESET)"
	@echo ""
	@echo "$(GREEN)========================================$(RESET)"
	@echo "$(GREEN)  Development server starting!$(RESET)"
	@echo "$(GREEN)========================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available at:$(RESET)"
	@echo "  - Web UI:    http://localhost:8000"
	@echo "  - API Docs:  http://localhost:8000/api/v1/docs"
	@echo ""
	@echo "$(YELLOW)Note: Using SQLite (in-memory) and demo mode.$(RESET)"
	@echo "$(YELLOW)Celery tasks will run synchronously.$(RESET)"
	@echo ""
	@. .venv/bin/activate && DEMO_MODE=true uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

quick-start: ## Quick start without building (uses cached images)
	@echo "$(GREEN)Quick starting services...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)✓ Started! Visit http://localhost:8000$(RESET)"

# Celery specific commands
celery-status: ## Show Celery worker status
	docker compose exec celery-worker celery -A app.tasks.celery_app status

celery-tasks: ## List registered Celery tasks
	docker compose exec celery-worker celery -A app.tasks.celery_app inspect registered

celery-active: ## Show active Celery tasks
	docker compose exec celery-worker celery -A app.tasks.celery_app inspect active

# Model training
train-model: ## Manually trigger model training
	@echo "$(YELLOW)Triggering model training...$(RESET)"
	docker compose exec celery-worker celery -A app.tasks.celery_app call app.tasks.tasks.train_calibration_model_task
	@echo "$(GREEN)✓ Training task queued$(RESET)"

# Database operations
db-backup: ## Backup database to backups/
	@mkdir -p backups
	@echo "$(YELLOW)Backing up database...$(RESET)"
	docker compose exec -T db pg_dump -U skillsprout skillsprout > backups/skillsprout_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Backup created in backups/$(RESET)"

db-restore: ## Restore database from latest backup
	@echo "$(YELLOW)Restoring database from latest backup...$(RESET)"
	@LATEST=$$(ls -t backups/*.sql | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "$(RED)No backup files found in backups/$(RESET)"; \
		exit 1; \
	fi; \
	echo "Restoring from $$LATEST..."; \
	docker compose exec -T db psql -U skillsprout skillsprout < $$LATEST
	@echo "$(GREEN)✓ Database restored$(RESET)"

# Testing variations
test-unit: ## Run only unit tests
	docker compose --profile test run --rm test pytest tests/unit/ -v

test-integration: ## Run only integration tests
	docker compose --profile test run --rm test pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	docker compose --profile test run --rm test pytest --cov=app --cov-report=html --cov-report=term

# Production build
build-prod: ## Build production image
	@echo "$(YELLOW)Building production image...$(RESET)"
	docker build --target production -t skillsprout:production .
	@echo "$(GREEN)✓ Production image built$(RESET)"

# Environment info
info: ## Show environment information
	@echo "$(BLUE)SkillSprout Environment Info$(RESET)"
	@echo ""
	@echo "$(YELLOW)Docker Compose Version:$(RESET)"
	@docker compose version
	@echo ""
	@echo "$(YELLOW)Docker Version:$(RESET)"
	@docker version --format '{{.Server.Version}}'
	@echo ""
	@echo "$(YELLOW)Services Status:$(RESET)"
	@docker compose ps
	@echo ""
