.PHONY: up down clean mock-data

# Docker Compose commands
up:
	docker compose up -d

down:
	docker compose down

clean: down
	docker compose down -v

# Mock data generation
mock-data:
	uv run -m src.mock_data.make_index_mock

# Help command
help:
	@echo "Available commands:"
	@echo "  make up        - Start OpenSearch container"
	@echo "  make down      - Stop OpenSearch container"
	@echo "  make clean     - Stop and remove OpenSearch container and volumes"
	@echo "  make mock-data - Generate mock data and index it in OpenSearch" 