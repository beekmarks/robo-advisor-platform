# Contributing to Robo-Advisor

Thanks for helping improve the project! This short guide covers how to run locally and basic commit conventions. For architecture and deployment, see `README.md` and `docs/deployment.md`.

## Quick start (local development)

Prereqs
- Docker Desktop
- Node.js 18+ (optional for frontend-only work)
- Python 3.11+ (optional if running services outside Docker)

Steps
1) Copy envs
   - `cp .env.example .env`
   - Fill non-empty values as needed (e.g., `OPENAI_API_KEY` for live LLM)
2) Start stack
   - `docker-compose up -d`
3) Open the UI
   - http://localhost:3000
4) Smoke test (optional)
   - `./smoke-test.sh`

Useful URLs (local)
- user: http://localhost:8080/health
- market-data: http://localhost:8082/health
- portfolio: http://localhost:8083/health
- rebalancing: http://localhost:8084/health
- llm: http://localhost:8085/health
- trade-execution: http://localhost:8086/health

## Coding guidelines (short version)

Backend (FastAPI / Python)
- Use async endpoints, Pydantic models, and type hints
- Use Decimal for money (never float); round to 2 dp for USD
- Validate inputs, return proper HTTP status codes
- Keep env-driven config (URLs, secrets)

Frontend (React)
- Functional components with hooks
- Handle API errors (Axios), keep UI responsive
- Keep props/state typed when possible (TS-friendly patterns)

Kubernetes
- App settings in ConfigMaps, secrets in Secrets
- Probes configured; prefer small containers and minimal privileges

Security
- Never commit secrets; `.env` is ignored by Git
- Use K8s Secrets or CI secrets for deployments

## Testing

Backend
- From an individual service folder, run: `python -m pytest`
- Mock external calls (market data, LLM) in unit tests

Frontend
- From `frontend/`: `npm test`

## Commit conventions

Use Conventional Commits to keep history readable and enable automation.
- `feat(service-name): add X`
- `fix(frontend): handle null from /health`
- `docs(readme): add demo flow`
- `refactor(rebalancing): extract drift calc`
- `test(market-data): mock yfinance history fallback`
- `chore(ci): bump node to 18`

Tips
- Scope with the service or area when possible: `feat(rebalancing)`, `fix(frontend)`
- Keep the subject under ~72 chars; add details in the body if needed

## Branching and PRs

- Create feature branches from `main`
- Keep PRs small, focused, and linked to an issue/task
- Add screenshots for UI changes and sample JSON for API changes

## Deployment

- Local: use Docker Compose (`docker-compose up -d`)
- Kubernetes: see `infrastructure/k8s/` and `docs/deployment.md` (EKS + Chaos)

## Reporting issues

Please include:
- Steps to reproduce
- Expected vs. actual behavior
- Logs or screenshots (scrub sensitive info)

Thanks for contributing and helping us ship safer, faster, and clearer code.
