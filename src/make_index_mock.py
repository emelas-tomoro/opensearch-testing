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
            "index": {"number_of_shards": 3, "number_of_replicas": 1},
            "analysis": {
                "analyzer": {
                    "player_tag_analyzer": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "account_id":        {"type": "keyword"},
                "player_id":         {"type": "keyword"},
                "player_tag":        {"type": "text", "analyzer": "player_tag_analyzer"},
                "avatar":           {"type": "text"},
                "alliance_name":     {"type": "text"},
                "level":            {"type": "integer"},
                # ── +11 useful recovery fields ──
                "email":             {"type": "keyword"},
                "phone_number":      {"type": "keyword"},
                "country":           {"type": "keyword"},
                "device_id":         {"type": "keyword"},
                "registration_date": {"type": "date"},
                "last_login":        {"type": "date"},
                "ip_address":        {"type": "ip"},
                "subscription_status": {"type": "keyword"},
                "account_status":      {"type": "keyword"},
                "preferred_language":  {"type": "keyword"},
                "date_of_birth":       {"type": "date"},
            },
        },
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
    reg = fake.date_time_between(start_date="-5y", end_date="now")
    last = reg + timedelta(days=random.randint(0, 365 * 5))
    return {
        "_index": index_name,
        "_id": str(uuid.uuid4()),
        "_source": {
            "account_id": f"ACC{seq:07d}",
            "player_id": str(uuid.uuid4()),
            "player_tag": f"#{random.randint(100000,9_999_999):07d}",
            "avatar": fake.catch_phrase(),
            "alliance_name": random.choice(ALLIANCE_NAMES),
            "level": random.randint(1, 100),
            "email": fake.email(),
            "phone_number": fake.phone_number(),
            "country": fake.country_code(),
            "device_id": fake.uuid4(),
            "registration_date": reg.isoformat(),
            "last_login": last.isoformat(),
            "ip_address": fake.ipv4_public(),
            "subscription_status": random.choice(["free", "premium"]),
            "account_status": random.choice(["active", "banned", "deleted"]),
            "preferred_language": random.choice(["en", "es", "fr", "de", "pt", "ru", "zh"]),
            "date_of_birth": str(fake.date_of_birth(minimum_age=13, maximum_age=60, tzinfo=None)),
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