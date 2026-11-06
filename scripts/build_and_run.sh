#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f models/model.joblib ]]; then
  echo "Нет models/model.joblib. Сначала выполните: make train" >&2
  exit 1
fi

IMAGE="${IMAGE_NAME:-credit-default-pd}"
CONTAINER="${CONTAINER_NAME:-credit-default-api}"

docker build -t "$IMAGE" .
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p 8000:8000 "$IMAGE"

echo "Ждем /health..."
ready=0
for _ in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "Контейнер не ответил на /health" >&2
  docker logs "$CONTAINER" || true
  exit 1
fi

curl -fsS http://127.0.0.1:8000/health
echo
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json
echo
