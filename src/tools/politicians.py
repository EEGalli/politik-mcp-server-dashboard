"""
MCP-tools för att hantera partier, politiker och vallöften.
"""

from datetime import datetime, timezone


# ── Partier ──────────────────────────────────────────────────────


def add_party(db, name: str, abbreviation: str, block: str = "", website_url: str = "") -> dict:
    """Lägg till ett parti i databasen."""
    collection = db.collection("parties")

    doc = {
        "name": name,
        "abbreviation": abbreviation,
        "block": block,
        "website_url": website_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = collection.insert(doc)
    doc["_key"] = result["_key"]
    return doc


def list_parties(db) -> list[dict]:
    """Lista alla partier."""
    cursor = db.aql.execute("FOR p IN parties SORT p.name RETURN p")
    return list(cursor)


def get_party(db, key: str) -> dict | None:
    """Hämta ett specifikt parti med nyckel."""
    collection = db.collection("parties")
    if collection.has(key):
        return collection.get(key)
    return None


# ── Politiker ────────────────────────────────────────────────────


def add_politician(db, name: str, party_key: str, social_media: dict | None = None) -> dict:
    """Lägg till en ny politiker i databasen."""
    collection = db.collection("politicians")

    doc = {
        "name": name,
        "party_key": party_key,
        "social_media": social_media or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = collection.insert(doc)
    doc["_key"] = result["_key"]
    return doc


def list_politicians(db, party_key: str | None = None) -> list[dict]:
    """Lista alla politiker, valfritt filtrerade per parti."""
    if party_key:
        cursor = db.aql.execute(
            "FOR p IN politicians FILTER p.party_key == @party_key RETURN p",
            bind_vars={"party_key": party_key},
        )
    else:
        cursor = db.aql.execute("FOR p IN politicians RETURN p")
    return list(cursor)


def get_politician(db, key: str) -> dict | None:
    """Hämta en specifik politiker med nyckel."""
    collection = db.collection("politicians")
    if collection.has(key):
        return collection.get(key)
    return None


# ── Vallöften ────────────────────────────────────────────────────


def add_promise(
    db,
    party_key: str,
    text: str,
    category: str,
    source_url: str = "",
    source_name: str = "",
    date: str = "",
) -> dict:
    """Lägg till ett vallöfte kopplat till ett parti."""
    collection = db.collection("promises")

    doc = {
        "party_key": party_key,
        "text": text,
        "category": category,
        "source_url": source_url,
        "source_name": source_name,
        "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = collection.insert(doc)
    doc["_key"] = result["_key"]
    return doc


def get_promises(db, party_key: str, category: str | None = None) -> list[dict]:
    """Hämta vallöften för ett parti, valfritt filtrerade per kategori."""
    bind_vars = {"party_key": party_key}
    query = "FOR p IN promises FILTER p.party_key == @party_key"

    if category:
        query += " FILTER p.category == @category"
        bind_vars["category"] = category

    query += " RETURN p"
    cursor = db.aql.execute(query, bind_vars=bind_vars)
    return list(cursor)


def delete_promise(db, promise_key: str) -> dict:
    """Radera ett vallöfte via dokumentnyckel."""
    collection = db.collection("promises")
    if not collection.has(promise_key):
        return {
            "deleted": False,
            "promise_key": promise_key,
            "error": "Löftet hittades inte",
        }

    collection.delete(promise_key, ignore_missing=True)
    return {
        "deleted": True,
        "promise_key": promise_key,
    }
