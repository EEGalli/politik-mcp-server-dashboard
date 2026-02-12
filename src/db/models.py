"""
ArangoDB-datamodeller för politik-mcp-server.

Collections:
  - politicians:  Politiker (namn, parti, sociala medier)
  - promises:     Vallöften kopplade till politiker
  - analyses:     Videoanalyser (transkription, kategori, etc.)
  - comparisons:  Jämförelser mellan analyser och löften (edge collection)
"""

from arango import ArangoClient
import os

ARANGO_HOST = os.getenv("ARANGO_HOST", "http://127.0.0.1:8530")
ARANGO_DB = os.getenv("ARANGO_DB", "politik_mcp")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "")

COLLECTIONS = {
    "politicians": {
        "type": "document",
        "schema": {
            "name": str,           # "Ulf Kristersson"
            "party": str,          # "Moderaterna"
            "social_media": dict,  # {"tiktok": "@ulf", "instagram": "@ulf"}
            "created_at": str,
        },
    },
    "promises": {
        "type": "document",
        "schema": {
            "politician_key": str,  # Koppling till politiker
            "text": str,            # Löftets text
            "category": str,        # "Skola", "Sjukvård", etc.
            "source_url": str,      # Varifrån löftet hämtades
            "source_name": str,     # "Valmanifest 2022"
            "date": str,            # "2022-09-01"
            "created_at": str,
        },
    },
    "analyses": {
        "type": "document",
        "schema": {
            "video_file": str,
            "politician_key": str,
            "transcription": str,
            "summary": str,
            "category": str,
            "political_color": str,
            "platform": str,       # "tiktok", "instagram", etc.
            "analyzed_at": str,
        },
    },
    "comparisons": {
        "type": "edge",
        "schema": {
            "_from": str,  # analyses/{key}
            "_to": str,    # promises/{key}
            "match_score": float,   # 0.0-1.0
            "match_type": str,      # "supports", "contradicts", "unrelated"
            "explanation": str,
            "compared_at": str,
        },
    },
}


def get_db():
    """Anslut till ArangoDB och returnera databasobjektet."""
    client = ArangoClient(hosts=ARANGO_HOST)
    sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)

    if not sys_db.has_database(ARANGO_DB):
        sys_db.create_database(ARANGO_DB)

    db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD)
    return db


def ensure_collections(db):
    """Skapa alla collections om de inte redan finns."""
    for name, config in COLLECTIONS.items():
        if not db.has_collection(name):
            if config["type"] == "edge":
                db.create_collection(name, edge=True)
            else:
                db.create_collection(name)
    return db


def init_db():
    """Initiera databas och collections. Returnerar db-objektet."""
    db = get_db()
    ensure_collections(db)
    return db
