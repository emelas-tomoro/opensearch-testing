# OpenSearch Integration Tests

This project contains integration tests for OpenSearch, a powerful open-source search and analytics engine. The tests verify core functionality and ensure proper integration with OpenSearch.

## Features

- Comprehensive test suite for OpenSearch functionality
- Support for both local development and CI environments
- Tests for:
  - Basic cluster health and connectivity
  - Document indexing and searching
  - Bulk operations
  - Aggregations
  - Index mappings and settings
  - Complex queries with boolean logic and date ranges

## Prerequisites

- Python 3.12 or higher
- Docker (for local development)
- OpenSearch 3.0.0
- uv (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd opensearch-testing
```

2. Install the project and its dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Running Tests

### Local Development

The tests use testcontainers to spin up a local OpenSearch instance. Simply run:

```bash
pytest tests/test_opensearch.py -v
```

### CI Environment

In CI (GitHub Actions), the tests will automatically use the provided OpenSearch service. The CI pipeline is configured in `.github/workflows/ci.yml`.

## Test Structure

The test suite includes:

- `test_opensearch_smoke`: Basic health checks and version verification
- `test_opensearch_index_and_search`: Document indexing and search functionality
- `test_opensearch_bulk_operations`: Bulk document operations
- `test_opensearch_aggregations`: Aggregation queries and results
- `test_opensearch_mapping_and_settings`: Index configuration and settings
- `test_opensearch_complex_queries`: Advanced query scenarios including boolean queries, term matching, and range queries

## Dependencies

The project uses the following main dependencies:
- opensearch-py >= 2.8.0
- pytest >= 8.3.5
- testcontainers >= 4.10.0

## Configuration

The tests automatically handle configuration for different environments:
- Local development: Uses [testcontainers](https://testcontainers.com/modules/opensearch/?language=python) with OpenSearch 3.0.0.
- CI: Uses GitHub Actions [OpenSearch service](https://github.com/ankane/setup-opensearch).

### Environment Configuration

You can control the test environment using environment variables. Create a `.env` file in the project root to override default settings:

```bash
# .env
GITHUB_ACTIONS=true  # Set to true to use GitHub Actions OpenSearch service
```

When `GITHUB_ACTIONS=true`, the tests will use the GitHub Actions OpenSearch service instead of spinning up a local container. This is useful for:
- Testing against a specific OpenSearch version in CI
- Debugging CI-related issues locally
- Ensuring consistent behavior between local and CI environments