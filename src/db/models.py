"""
ArangoDB-datamodeller för politik-mcp-server.

Collections:
  - parties:      Partier (namn, förkortning, block, webbplats)
  - politicians:  Politiker (namn, parti-koppling, sociala medier)
  - promises:     Vallöften kopplade till partier
  - analyses:     Videoanalyser (transkription, kategori, etc.)
  - comparisons:  Jämförelser mellan analyser och löften (edge collection)
"""

from arango import ArangoClient
import os

COLLECTIONS = {
    "parties": {
        "type": "document",
        "schema": {
            "name": str,           # "Moderaterna"
            "abbreviation": str,   # "M"
            "block": str,          # "höger", "vänster", "mitt"
            "website_url": str,    # "https://moderaterna.se"
            "created_at": str,
        },
    },
    "politicians": {
        "type": "document",
        "schema": {
            "name": str,           # "Ulf Kristersson"
            "party_key": str,      # Koppling till parti
            "social_media": dict,  # {"tiktok": "@ulf", "instagram": "@ulf"}
            "created_at": str,
        },
    },
    "promises": {
        "type": "document",
        "schema": {
            "party_key": str,       # Koppling till parti
            "text": str,            # Löftets text
            "category": str,        # "Skola", "Sjukvård", etc.
            "source_url": str,      # Varifrån löftet hämtades
            "source_urls": list,    # Alla käll-URL:er där samma löfte hittats
            "source_count": int,    # Antal unika käll-URL:er för löftet
            "source_name": str,     # "Valmanifest 2022"
            "date": str,            # "2022-09-01"
            "mention_count": int,   # Hur många gånger löftet återkommit vid insamling
            "last_seen_at": str,    # Senaste gång löftet observerades i källmaterial
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
    "monitored_content": {
        "type": "document",
        "schema": {
            "politician_key": str,  # Koppling till politiker
            "platform": str,        # "tiktok", "twitter", "instagram"
            "item_id": str,         # Unikt RSS-item-ID (guid/link)
            "item_url": str,        # OriginalURL till inlägget
            "item_title": str,      # Titel/caption
            "analysis_key": str,    # Koppling till skapad analys
            "fetched_at": str,      # ISO timestamp
            "status": str,          # "analyzed", "skipped", "failed"
            "comparisons_done": int,
        },
    },
}


def get_db():
    """Anslut till ArangoDB och returnera databasobjektet."""
    arango_host = os.getenv("ARANGO_HOST", "http://127.0.0.1:8530")
    arango_db = os.getenv("ARANGO_DB", "politik_mcp")
    arango_user = os.getenv("ARANGO_USER", "root")
    arango_password = os.getenv("ARANGO_PASSWORD", "")

    client = ArangoClient(hosts=arango_host)
    sys_db = client.db("_system", username=arango_user, password=arango_password)

    if not sys_db.has_database(arango_db):
        sys_db.create_database(arango_db)

    db = client.db(arango_db, username=arango_user, password=arango_password)
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
