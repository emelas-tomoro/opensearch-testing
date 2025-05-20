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

## Prerequisites

- Python 3.x
- Docker (for local development)
- OpenSearch 2.11.0
- uv (Python package installer)

## Installation

1. Install the project and its dependencies:
```bash
uv sync
```

Alternatively, you can use GitHub Actions to run the tests (as defined in .github/ci.yml):
```yaml
name: OpenSearch Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - name: Set up OpenSearch
        uses: ankane/setup-opensearch@v1
        with:
          version: '2.11.0'
          port: 9200
          security: true
      - run: uv sync
      - run: pytest
```

## Running Tests

### Local Development

The tests use testcontainers to spin up a local OpenSearch instance. Simply run:

```bash
pytest
```

### CI Environment

In CI (GitHub Actions), the tests will automatically use the provided OpenSearch service.

## Test Structure

- `test_opensearch_smoke`: Basic health checks and version verification
- `test_opensearch_index_and_search`: Document indexing and search functionality
- `test_opensearch_bulk_operations`: Bulk document operations
- `test_opensearch_aggregations`: Aggregation queries and results
- `test_opensearch_mapping_and_settings`: Index configuration and settings

## Configuration

The tests automatically handle configuration for different environments:
- Local development: Uses testcontainers with OpenSearch 2.11.0
- CI: Uses GitHub Actions OpenSearch service