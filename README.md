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

## Setup and Running

### Starting OpenSearch

1. Start the OpenSearch and OpenSearch Dashboards containers:
```bash
make up
```

This will start:
- OpenSearch on http://localhost:9200
- OpenSearch Dashboards on http://localhost:5601

To stop the containers:
```bash
make down
```

To clean up (stop and remove containers and volumes):
```bash
make clean
```

### Generating Mock Data

The project includes a script to generate and index mock game account data. This is useful for testing and development.

1. Configure the mock data generation in `.env`:
```bash
# for tesing with github actions
GITHUB_ACTIONS='true'

# OpenSearch Configuration
# URL where OpenSearch is running (matches docker-compose.yaml)
OPENSEARCH_HOST=http://localhost:9200

# Authentication (leave blank since security plugin is disabled in docker-compose)
OPENSEARCH_USER=
OPENSEARCH_PASS=

# Index Configuration
# Name of the index to create/populate
INDEX_NAME=game_accounts

# Number of documents to generate and index
NUM_DOCS=100000

# Batch size for bulk indexing operations
BATCH_SIZE=5000

# Faker seed for reproducible fake data generation
PYTHONHASHSEED=0

# Set to 'true' to overwrite existing index, 'false' to error if index exists
FORCE_OVERWRITE=true 
```

2. Generate and index the mock data:
```bash
make mock-data
```

This will:
- Create the `game_accounts` index with appropriate mappings
- Generate synthetic game account data
- Index the data in batches
- Show progress and results

### Search Utilities

The project includes a comprehensive `GameAccountSearcher` class for advanced search operations. Key features:

- Support for multiple query types (match, term, prefix, wildcard, range)
- Boolean query combinations (must, should, filter, must_not)
- Range queries for dates and numbers
- Aggregation support
- Fuzzy matching
- Boost factors for relevance tuning

Example usage:
```python
from src.utils.search_utils import GameAccountSearcher, QueryType

# Basic search
searcher = GameAccountSearcher(client)
results = searcher.add_query("player_tag", "ABC12", QueryType.TERM).search()

# Complex search with multiple conditions
results = (
    searcher
    .add_query("alliance_name", "dragon", QueryType.MATCH, fuzziness="AUTO")
    .add_query("subscription_status", "premium", QueryType.TERM, context="filter")
    .add_range_query("last_login", gte="now-30d/d")
    .search(size=20, from_=0)
)
```

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