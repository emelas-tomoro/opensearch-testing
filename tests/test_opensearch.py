import pytest
from testcontainers.opensearch import OpenSearchContainer
import time
import os
from opensearchpy import OpenSearch


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
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
        )
        yield client
    else:
        # Use local testcontainer
        container = OpenSearchContainer(
            image="opensearchproject/opensearch:2.11.0",
            port=9200,
        )
        with container as container:
            # Give OpenSearch time to fully initialize
            client = container.get_client()
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
    assert info["version"]["number"].startswith("2.11.0"), "Unexpected OpenSearch version"
    
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

