.PHONY: install train tune test lint api web up down monitor mlflow-ui

install:            ## Python deps (incl. training extras) + frontend deps
	uv sync --all-extras
	cd frontend && npm install

train:              ## Train with params.yaml, log to MLflow (sqlite), enforce the quality gate
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python -m sentiment_mlops.train

tune:               ## Same as train but runs the group aware grid search first
	sed 's/enabled: false/enabled: true/' params.yaml > /tmp/params.tune.yaml
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python -m sentiment_mlops.train --params /tmp/params.tune.yaml

test:
	SENTIMENT_LOG_PREDICTIONS=false uv run pytest --cov=sentiment_mlops

lint:
	uv run ruff check . && uv run ruff format --check .

api:                ## API with auto-reload on http://localhost:8000 (docs at /docs)
	uv run fastapi dev src/sentiment_mlops/api/main.py

web:                ## Frontend dev server on http://localhost:5173 (proxies /api to :8000)
	cd frontend && npm run dev

up:                 ## Full stack in Docker on http://localhost:8080
	docker compose up --build -d

down:
	docker compose --profile monitoring --profile tracking down

monitor:            ## Drift report from the prediction log
	uv run python -m sentiment_mlops.monitor

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
