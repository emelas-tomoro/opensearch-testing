#!/usr/bin/env python3
"""
make_index_mock.py  •  Build a synthetic 'game_accounts' index for recovery-bot prototyping.

.env file:
------------------------------------
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

"""

import os, random, uuid, sys
from datetime import timedelta
from faker import Faker
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers, exceptions


def create_opensearch_client(host: str, user: str = None, pwd: str = None) -> OpenSearch:
    """Create and return an OpenSearch client with the given configuration."""
    client = OpenSearch(
        hosts=[host],
        http_auth=(user, pwd) if user or pwd else None,
        verify_certs=False,    # fine for local dev
        ssl_show_warn=False,
    )
    try:
        info = client.info()
        print(f"Connected to OpenSearch {info['version']['number']} at {host}")
        return client
    except exceptions.ConnectionError as e:
        sys.exit(f"❌  Cannot reach OpenSearch at {host}  •  {e}")


def create_index_mapping(client: OpenSearch, index_name: str, force: bool = False) -> None:
    """
    Create the index with the specified mapping.
    
    Args:
        client: OpenSearch client
        index_name: Name of the index to create
        force: If True, will overwrite existing index. If False, will raise error if index exists.
    """
    if client.indices.exists(index_name):
        if not force:
            raise ValueError(
                f"Index '{index_name}' already exists. Use FORCE_OVERWRITE=true in .env to overwrite, "
                "or choose a different index name."
            )
        print(f"Index '{index_name}' exists – deleting …")
        client.indices.delete(index_name)
    
    mapping = {
        "settings": {
            "index": {
                "refresh_interval": "120s",
                "number_of_shards": 16,
                "number_of_replicas": 2,
                "analysis": {
                    "normalizer": {
                        "name_normalizer": {
                            "type": "custom",
                            "filter": ["icu_folding", "lowercase"]
                        }
                    },
                    "analyzer": {
                        "name_tokenizer": {
                            "type": "custom",
                            "tokenizer": "icu_tokenizer",
                            "filter": ["icu_folding", "apostrophe"]
                        }
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "account_id": {"type": "long"},
                "alliance_id": {"type": "long"},
                "alliance_name": {"type": "text", "analyzer": "name_tokenizer"},
                "alliance_name_history": {"type": "text", "analyzer": "name_tokenizer"},
                "alliance_name_history_raw": {"type": "keyword"},
                "alliance_name_raw": {"type": "keyword"},
                "alliance_name_raw_lower": {"type": "keyword"},
                "avatar_id": {"type": "long"},
                "create_country": {"type": "keyword"},
                "dirty": {"type": "boolean"},
                "exp_level": {"type": "long"},
                "name": {"type": "text", "analyzer": "name_tokenizer"},
                "name_history": {"type": "text", "analyzer": "name_tokenizer"},
                "name_history_raw": {"type": "keyword"},
                "name_raw": {"type": "keyword"},
                "name_raw_lower": {"type": "keyword"},
                "name_raw_lower_rev": {"type": "keyword"},
                "name_raw_rev": {"type": "keyword"},
                "reputation": {"type": "long"},
                "updated": {"type": "date"}
            }
        }
    }
    
    client.indices.create(index_name, body=mapping)
    print(f"✅  Created index '{index_name}'")


# Predefined list of alliance names
ALLIANCE_NAMES = [
    "Dragon Warriors",
    "Phoenix Rising",
    "Shadow Knights",
    "Golden Eagles",
    "Silver Wolves",
    "Crimson Dragons",
    "Emerald Guardians",
    "Azure Knights",
    "Crystal Phoenix",
    "Thunder Legion"
]


def generate_document(seq: int, index_name: str, fake: Faker) -> dict:
    """Generate a single document with synthetic data."""
    name = fake.user_name()
    alliance_name = random.choice(ALLIANCE_NAMES)
    current_time = fake.date_time_between(start_date="-1y", end_date="now")
    
    return {
        "_index": index_name,
        "_id": str(uuid.uuid4()),
        "_source": {
            "account_id": random.randint(100000000, 999999999),
            "alliance_id": random.randint(100000000, 999999999),
            "alliance_name": alliance_name,
            "alliance_name_history": [],
            "alliance_name_history_raw": [],
            "alliance_name_raw": alliance_name,
            "alliance_name_raw_lower": alliance_name.lower(),
            "avatar_id": random.randint(100000000, 999999999),
            "create_country": fake.country_code(),
            "dirty": random.choice([True, False]),
            "exp_level": random.randint(1, 100),
            "name": name,
            "name_history": [],
            "name_history_raw": [],
            "name_raw": name,
            "name_raw_lower": name.lower(),
            "name_raw_lower_rev": name.lower()[::-1],
            "name_raw_rev": name[::-1],
            "reputation": random.randint(0, 1000),
            "updated": current_time.isoformat()
        },
    }


def index_documents(client: OpenSearch, index_name: str, num_docs: int, batch_size: int) -> None:
    """Index the generated documents in batches."""
    print(f"Indexing {num_docs:,} docs (batch {batch_size}) …")
    
    def doc_stream():
        for i in range(num_docs):
            try:
                yield generate_document(i, index_name, fake)
            except Exception as e:
                print(f"Error generating document {i}: {str(e)}")
                continue
    
    try:
        success, failed = helpers.bulk(
            client, 
            doc_stream(), 
            chunk_size=batch_size, 
            request_timeout=120,
            raise_on_error=False,
            raise_on_exception=False
        )
        
        if failed:
            print(f"⚠️  Some documents failed to index. Failed count: {len(failed)}")
            for error in failed[:5]:  # Show first 5 errors
                print(f"Error: {error}")
        else:
            print(f"✅ Successfully indexed {success} documents")
            
        client.indices.refresh(index_name)
        print(f"🎉  Done – '{index_name}' now holds {client.count(index=index_name)['count']:,} documents")
    except Exception as e:
        print(f"❌ Error during bulk indexing: {str(e)}")
        raise


def main():
    """Main function to create and populate the OpenSearch index with mock data."""
    # ── config ────────────────────────────────────────────────────────────────────
    HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
    USER = os.getenv("OPENSEARCH_USER")
    PWD = os.getenv("OPENSEARCH_PASS")
    INDEX = os.getenv("INDEX_NAME", "game_accounts")
    N_DOCS = int(os.getenv("NUM_DOCS", "100000"))
    CHUNK = int(os.getenv("BATCH_SIZE", "5000"))
    FORCE = os.getenv("FORCE_OVERWRITE", "").lower() in ("true", "1", "yes")

    # Initialize Faker with a fixed seed for reproducibility
    global fake
    fake = Faker()
    random.seed(42)

    # Create client and index
    client = create_opensearch_client(HOST, USER, PWD)
    create_index_mapping(client, INDEX, force=FORCE)
    
    # Generate and index documents
    index_documents(client, INDEX, N_DOCS, CHUNK)


if __name__ == "__main__":
    main()