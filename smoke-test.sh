#!/usr/bin/env bash
set -euo pipefail

BLUE="\033[34m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; NC="\033[0m"

log() { echo -e "${BLUE}[*]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERR]${NC} $*"; }

open_frontend() {
  local url="http://localhost:3000"
  log "Opening frontend: $url"
  if command -v open >/dev/null 2>&1; then
    open "$url" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" || true
  else
    warn "No opener found. Please open $url manually."
  fi
}

http_code() {
  curl -sS -o /dev/null -w "%{http_code}" "$1"
}

wait_for() {
  local name="$1" url="$2" max_tries="${3:-40}" sleep_s="${4:-3}"
  log "Waiting for $name at $url ..."
  local i=1 code
  while [[ $i -le $max_tries ]]; do
    code=$(http_code "$url" || echo 000)
    if [[ "$code" == "200" ]]; then
      ok "$name is healthy (HTTP 200)"
      return 0
    fi
    sleep "$sleep_s"; i=$((i+1))
  done
  err "$name did not become healthy in time"
  return 1
}

smoke_get() {
  local name="$1" url="$2"
  log "GET $name: $url"
  local resp
  resp=$(curl -sS "$url") || { err "$name GET failed"; return 1; }
  ok "$name responded"
  echo "$resp" | head -c 400 | sed 's/.*/    &/'
}

smoke_post_json() {
  local name="$1" url="$2" body="$3"
  log "POST $name: $url"
  local resp
  resp=$(curl -sS -H 'Content-Type: application/json' -d "$body" "$url") || { err "$name POST failed"; return 1; }
  ok "$name responded"
  echo "$resp" | head -c 400 | sed 's/.*/    &/'
}

main() {
  open_frontend

  # Wait for services
  wait_for "user-service"        "http://localhost:8080/health"
  wait_for "market-data-service" "http://localhost:8082/health"
  wait_for "portfolio-service"   "http://localhost:8083/health"
  wait_for "llm-service"         "http://localhost:8085/health"
  wait_for "trade-execution"     "http://localhost:8086/health"
  wait_for "rebalancing-service" "http://localhost:8084/health"
  # New services
  wait_for "knowledge-graph"     "http://localhost:8087/health" || true
  wait_for "symbolic-reasoning"  "http://localhost:8088/health" || true

  # Exercise key endpoints
  smoke_get  "Market Quote AAPL" "http://localhost:8082/quote/AAPL" || true

  smoke_post_json "LLM Chat" "http://localhost:8085/chat" '{"user_id":"smoke-user","message":"I am a moderate risk investor saving for retirement","conversation_history":[]}' || true

  smoke_post_json "Trade Execute" "http://localhost:8086/execute" '{"user_id":"smoke-user","portfolio_id":"smoke-portfolio","symbol":"AAPL","side":"buy","quantity":1}' || true

  # Seed knowledge graph with sample data, then test reasoning and compliance
  smoke_post_json "KG Populate" "http://localhost:8087/graph/populate" '{}' || true
  smoke_post_json "KG Reasoning" "http://localhost:8087/reasoning/multi-hop" '{"query":"find correlation paths for AAPL","max_hops":3}' || true

  # Verified rebalancing flow
  smoke_post_json "Rebalancing Verified" "http://localhost:8084/execute-rebalance/verified" '{
    "portfolio": {
      "user_id": "smoke-user",
      "holdings": {"AAPL": 10, "MSFT": 10},
      "target_allocation": {"AAPL": 0.6, "MSFT": 0.4},
      "last_rebalanced": "2024-01-01T00:00:00Z"
    },
    "trigger_type": "strategic"
  }' || true

  # Rebalancing basic check
  smoke_post_json "Rebalancing Check" "http://localhost:8084/check-rebalance" '{
    "portfolio": {
      "user_id": "smoke-user",
      "holdings": {"AAPL": 5, "MSFT": 3},
      "target_allocation": {"AAPL": 0.5, "MSFT": 0.5},
      "last_rebalanced": "2024-01-01T00:00:00Z"
    },
    "market_conditions": {"vix": 18, "sp500_trend": "neutral"}
  }' || true

  ok "Smoke test completed"
}

main "$@"
