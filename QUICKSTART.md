# SkillSprout Quick Start Guide

Get up and running with SkillSprout in under 5 minutes using Docker.

## 🚀 One-Command Setup

```bash
make dev
```

That's it! This single command will:
1. Build all Docker images
2. Start PostgreSQL, Redis, FastAPI, and Celery
3. Run database migrations
4. Seed demo data
5. Show you the access URLs

**Access your application at:**
- 🌐 Web UI: http://localhost:8000
- 📚 API Docs: http://localhost:8000/api/v1/docs
- 💚 Health Check: http://localhost:8000/api/v1/health

---

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Make (optional but recommended)

**Check your installation:**
```bash
docker --version
docker-compose --version
make --version
```

---

## 🎯 Common Tasks

### View Logs
```bash
make logs          # All services
make logs-api      # API only
make logs-celery   # Celery worker only
```

### Run Tests
```bash
make test          # Full test suite
```

### Access Database
```bash
make shell-db      # PostgreSQL shell
```

### Access API Container
```bash
make shell-api     # Bash shell in API container
```

### Stop Everything
```bash
make down          # Stop services
make clean         # Stop and remove volumes
```

---

## 🧪 Testing

### Unit + Integration Tests
```bash
make test
```

### End-to-End Tests
```bash
# Ensure services are running
make up

# Run E2E tests
./scripts/test_e2e.sh
```

### Quick Smoke Test
```bash
./scripts/smoke_test.sh
```

---

## 🔄 Typical Workflow

### First Time Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd skillsprout

# 2. Start everything
make dev

# 3. Open browser to http://localhost:8000
```

### Daily Development
```bash
# Start services
make up

# View logs while coding
make logs-api

# Run tests before committing
make test

# Stop services when done
make down
```

### After Code Changes
```bash
# Restart API (picks up code changes)
make restart

# Or view logs to see hot reload
make logs-api
```

### After Model Changes
```bash
# Create migration
make migrate-create MSG="add new field"

# Apply migration
make migrate
```

---

## 🎓 Example User Flow

### Via Web UI
1. Visit http://localhost:8000
2. Click "Create Profile & Start"
3. Search for your occupation (e.g., "Software Developer")
4. Rate your skills
5. Get recommendations!

### Via API (curl)
```bash
# 1. Create user profile
curl -X POST http://localhost:8000/api/v1/user/profile \
  -H "Content-Type: application/json" \
  -d '{}'

# Save the user ID from response, then:

# 2. Search occupations
curl "http://localhost:8000/api/v1/occupations/search?q=software"

# 3. Set current occupation
curl -X POST http://localhost:8000/api/v1/user/1/current-occupation \
  -H "Content-Type: application/json" \
  -d '{"onet_code": "15-1252.00"}'

# 4. Get occupation skills
curl "http://localhost:8000/api/v1/occupations/15-1252.00/skills"

# 5. Rate skills (use element_ids from previous response)
curl -X POST http://localhost:8000/api/v1/user/1/skills/ratings \
  -H "Content-Type: application/json" \
  -d '{
    "ratings": [
      {"element_id": "2.B.1.a", "rating_0_4": 3},
      {"element_id": "2.B.8.a", "rating_0_4": 4},
      {"element_id": "2.B.1.g", "rating_0_4": 4}
    ]
  }'

# 6. Get recommendations
curl -X POST http://localhost:8000/api/v1/user/1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"limit_per_bucket": 10}'
```

---

## 🐛 Troubleshooting

### Services won't start
```bash
# Check if ports are in use
lsof -i :8000  # API
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Clean and rebuild
make clean
make build
make up
```

### Database issues
```bash
# Check database health
make health

# Reset database
make down
docker volume rm skillsprout_postgres_data
make up
```

### Need to see what's happening
```bash
# Check service status
make ps

# Check logs
make logs

# Check health
make health
```

### Tests failing
```bash
# Clean test environment
docker-compose -f docker-compose.test.yml down -v

# Run tests again
make test
```

---

## 📚 Next Steps

- **Full Docker guide**: See [DOCKER_TESTING.md](DOCKER_TESTING.md)
- **API documentation**: Visit http://localhost:8000/api/v1/docs
- **Modeling details**: See [MODELING_NOTES.md](MODELING_NOTES.md)
- **Complete README**: See [README.md](README.md)

---

## 🎉 Quick Win

Want to see it working immediately?

```bash
# 1. Start everything
make dev

# 2. Run smoke test
./scripts/smoke_test.sh

# 3. If all green, you're ready!
```

---

## 💡 Pro Tips

1. **Use `make help`** to see all available commands
2. **Keep `make logs-api` running** in a separate terminal while developing
3. **Run `make test` before every commit**
4. **Use `make shell-api`** to debug inside the container
5. **Check `make health`** if something seems wrong

---

## 🤔 Common Questions

**Q: Do I need Python installed locally?**
A: No! Docker handles everything.

**Q: Do I need PostgreSQL or Redis installed?**
A: No! Docker provides these services.

**Q: Can I use this for production?**
A: The Docker setup is production-ready, but review [Production Considerations](README.md#production-considerations) first.

**Q: How do I add O*NET credentials?**
A: Edit `docker-compose.yml` and set `ONET_USERNAME` and `ONET_PASSWORD`, then set `DEMO_MODE: "false"`.

**Q: How long does initial setup take?**
A: First build takes 2-3 minutes. After that, startup is ~10 seconds.

---

**Need help?** Run `make help` or check [DOCKER_TESTING.md](DOCKER_TESTING.md)
