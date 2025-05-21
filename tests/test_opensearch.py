import pytest
from testcontainers.opensearch import OpenSearchContainer
import time
import os
from opensearchpy import OpenSearch

import logging

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def opensearch_client():
    """Fixture that provides an OpenSearch client, using either:
    - GitHub Actions OpenSearch service when in CI
    - Local testcontainer when running locally
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Use GitHub Actions OpenSearch service
        client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_auth=('admin', 'admin'),  # Default credentials for GitHub Actions OpenSearch
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        yield client
    else:
        # Use local testcontainer
        container = OpenSearchContainer(
            image="opensearchproject/opensearch:3.0.0",
            # port=9200,
        )#.with_exposed_ports(9200).with_bind_ports(9200, 9200)
        with container as container:
            client = container.get_client()  # wired up with the right port, creds, TLS settings
            # optionally wait on cluster health
            for _ in range(30):
                try:
                    health = client.cluster.health(wait_for_status="yellow", request_timeout=1)
                    if health["status"] in ("yellow", "green"):
                        break
                except Exception:
                    time.sleep(1)
            else:
                pytest.skip("OpenSearch did not start in time")
            yield client


def test_opensearch_smoke(opensearch_client):
    """Smoke test to verify OpenSearch is running and healthy"""
    # Check cluster health
    health = opensearch_client.cluster.health()
    assert health["status"] in ["green", "yellow"], f"Cluster health is {health['status']}"
    assert health["number_of_nodes"] > 0, "No nodes found in cluster"
    
    # Check cluster info
    info = opensearch_client.info()
    assert "version" in info, "Version information missing"
    assert "number" in info["version"], "Version number missing"
        
    # Check if we can create and delete an index
    test_index = "smoke-test-index"
    try:
        # Create index
        create_response = opensearch_client.indices.create(index=test_index)
        assert create_response["acknowledged"], "Index creation not acknowledged"
        
        # Verify index exists
        assert opensearch_client.indices.exists(index=test_index), "Index not found after creation"
        
        # Delete index
        delete_response = opensearch_client.indices.delete(index=test_index)
        assert delete_response["acknowledged"], "Index deletion not acknowledged"
        
        # Verify index is gone
        assert not opensearch_client.indices.exists(index=test_index), "Index still exists after deletion"
    except Exception as e:
        pytest.fail(f"Smoke test failed with error: {str(e)}")


def test_opensearch_index_and_search(opensearch_client):
    """Test basic indexing and searching functionality"""
    index_name = "integration-test-index"
    doc = {"title": "TestDoc", "body": "Testcontainers with OpenSearch"}

    # Create index and add document
    opensearch_client.index(index=index_name, id=1, body=doc)

    # Refresh to make the document searchable
    opensearch_client.indices.refresh(index=index_name)

    # Search for the document
    result = opensearch_client.search(
        index=index_name,
        body={"query": {"match": {"body": "Testcontainers"}}}
    )

    assert result["hits"]["total"]["value"] > 0
    assert result["hits"]["hits"][0]["_source"]["title"] == "TestDoc"


def test_opensearch_bulk_operations(opensearch_client):
    """Test bulk indexing and searching"""
    index_name = "bulk-test-index"
    docs = [
        {"title": "Doc1", "content": "First test document"},
        {"title": "Doc2", "content": "Second test document"},
        {"title": "Doc3", "content": "Third test document"}
    ]

    # Bulk index documents
    bulk_data = []
    for i, doc in enumerate(docs, 1):
        bulk_data.extend([
            {"index": {"_index": index_name, "_id": i}},
            doc
        ])
    
    opensearch_client.bulk(body=bulk_data)
    opensearch_client.indices.refresh(index=index_name)

    # Search for all documents
    result = opensearch_client.search(
        index=index_name,
        body={"query": {"match_all": {}}}
    )

    assert result["hits"]["total"]["value"] == 3
    assert len(result["hits"]["hits"]) == 3


def test_opensearch_aggregations(opensearch_client):
    """Test aggregation functionality"""
    index_name = "agg-test-index"
    
    # Create index with proper mapping for aggregations
    mapping = {
        "mappings": {
            "properties": {
                "category": {"type": "keyword"},  # Use keyword type for aggregations
                "value": {"type": "integer"}
            }
        }
    }
    
    # Create index with mapping
    opensearch_client.indices.create(index=index_name, body=mapping)
    
    docs = [
        {"category": "A", "value": 10},
        {"category": "A", "value": 20},
        {"category": "B", "value": 30},
        {"category": "B", "value": 40}
    ]

    # Index documents
    bulk_data = []
    for i, doc in enumerate(docs, 1):
        bulk_data.extend([
            {"index": {"_index": index_name, "_id": i}},
            doc
        ])
    
    opensearch_client.bulk(body=bulk_data)
    opensearch_client.indices.refresh(index=index_name)

    # Perform aggregation
    result = opensearch_client.search(
        index=index_name,
        body={
            "aggs": {
                "avg_by_category": {
                    "terms": {"field": "category"},
                    "aggs": {
                        "avg_value": {"avg": {"field": "value"}}
                    }
                }
            }
        }
    )

    # Verify aggregation results
    buckets = result["aggregations"]["avg_by_category"]["buckets"]
    assert len(buckets) == 2
    
    # Check category A average
    category_a = next(b for b in buckets if b["key"] == "A")
    assert category_a["avg_value"]["value"] == 15.0
    
    # Check category B average
    category_b = next(b for b in buckets if b["key"] == "B")
    assert category_b["avg_value"]["value"] == 35.0


def test_opensearch_mapping_and_settings(opensearch_client):
    """Test index mapping and settings"""
    index_name = "mapping-test-index"
    
    # Create index with custom mapping and settings
    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "count": {"type": "integer"},
                "date": {"type": "date"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    opensearch_client.indices.create(index=index_name, body=mapping)
    
    # Get index mapping and settings
    mapping_result = opensearch_client.indices.get_mapping(index=index_name)
    settings_result = opensearch_client.indices.get_settings(index=index_name)
    
    # Verify mapping
    assert "title" in mapping_result[index_name]["mappings"]["properties"]
    assert "count" in mapping_result[index_name]["mappings"]["properties"]
    assert "date" in mapping_result[index_name]["mappings"]["properties"]
    
    # Verify settings
    assert settings_result[index_name]["settings"]["index"]["number_of_shards"] == "1"
    assert settings_result[index_name]["settings"]["index"]["number_of_replicas"] == "0"


def test_opensearch_complex_queries(opensearch_client):
    """Test complex query scenarios including boolean queries, term matching, and range queries"""
    index_name = "complex-query-test-index"
    
    # Create index with mapping for our test data
    mapping = {
        "mappings": {
            "properties": {
                "account_id": {"type": "long"},
                "status": {"type": "keyword"},
                "amount": {"type": "float"},
                "created_at": {"type": "date"},
                "tags": {"type": "keyword"}
            }
        }
    }
    
    opensearch_client.indices.create(index=index_name, body=mapping)
    
    # Sample documents
    docs = [
        {
            "account_id": 12927781653,
            "status": "active",
            "amount": 1000.50,
            "created_at": "2024-01-01T00:00:00Z",
            "tags": ["premium", "verified"]
        },
        {
            "account_id": 12927781654,
            "status": "inactive",
            "amount": 500.75,
            "created_at": "2024-01-02T00:00:00Z",
            "tags": ["basic"]
        },
        {
            "account_id": 12927781655,
            "status": "active",
            "amount": 2000.25,
            "created_at": "2024-01-03T00:00:00Z",
            "tags": ["premium", "vip"]
        }
    ]
    
    # Bulk index documents
    bulk_data = []
    for i, doc in enumerate(docs, 1):
        bulk_data.extend([
            {"index": {"_index": index_name, "_id": i}},
            doc
        ])
    
    opensearch_client.bulk(body=bulk_data)
    opensearch_client.indices.refresh(index=index_name)
    
    # Test 1: Simple term query
    term_query = {
        "query": {
            "term": {
                "account_id": 12927781653
            }
        }
    }
    
    result = opensearch_client.search(index=index_name, body=term_query)
    assert result["hits"]["total"]["value"] == 1
    assert result["hits"]["hits"][0]["_source"]["account_id"] == 12927781653
    
    # Test 2: Boolean query with multiple conditions
    bool_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"status": "active"}},
                    {"range": {"amount": {"gte": 1000}}}
                ],
                "filter": [
                    {"terms": {"tags": ["premium"]}}
                ]
            }
        }
    }
    
    result = opensearch_client.search(index=index_name, body=bool_query)
    assert result["hits"]["total"]["value"] == 2
    assert all(hit["_source"]["status"] == "active" for hit in result["hits"]["hits"])
    assert all(hit["_source"]["amount"] >= 1000 for hit in result["hits"]["hits"])
    assert all("premium" in hit["_source"]["tags"] for hit in result["hits"]["hits"])
    
    # Test 3: Date range query
    date_query = {
        "query": {
            "range": {
                "created_at": {
                    "gte": "2024-01-02T00:00:00Z",
                    "lte": "2024-01-03T00:00:00Z"
                }
            }
        }
    }
    
    result = opensearch_client.search(index=index_name, body=date_query)
    assert result["hits"]["total"]["value"] == 2
    assert all("2024-01-02" in hit["_source"]["created_at"] or 
              "2024-01-03" in hit["_source"]["created_at"] 
              for hit in result["hits"]["hits"])

