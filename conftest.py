import pytest
from testcontainers.opensearch import OpenSearchContainer
import time


@pytest.fixture(scope="session")
def opensearch_client():
    container = OpenSearchContainer(
        image="opensearchproject/opensearch:2.4.0",
        port=9200,
    )
    with container as container:
        # Give OpenSearch time to fully initialize
        time.sleep(10)
        client = container.get_client()
        yield client
