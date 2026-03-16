#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.yaml"

echo "⏳ Stopping and deleting Loki/Grafana volumes..."
docker compose -f "$COMPOSE_FILE" down -v loki grafana

echo "🚀 Restarting services..."
docker compose -f "$COMPOSE_FILE" up -d loki grafana