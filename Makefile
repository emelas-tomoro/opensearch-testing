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
	python src/make_index_mock.py

# Help command
help:
	@echo "Available commands:"
	@echo "  make up        - Start OpenSearch container"
	@echo "  make down      - Stop OpenSearch container"
	@echo "  make clean     - Stop and remove OpenSearch container and volumes"
	@echo "  make mock-data - Generate mock data and index it in OpenSearch" 