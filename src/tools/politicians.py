"""
MCP-tools för att hantera politiker och vallöften.
"""

from datetime import datetime, timezone


def add_politician(db, name: str, party: str, social_media: dict | None = None) -> dict:
    """Lägg till en ny politiker i databasen."""
    collection = db.collection("politicians")

    doc = {
        "name": name,
        "party": party,
        "social_media": social_media or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = collection.insert(doc)
    doc["_key"] = result["_key"]
    return doc


def list_politicians(db, party: str | None = None) -> list[dict]:
    """Lista alla politiker, valfritt filtrerade per parti."""
    if party:
        cursor = db.aql.execute(
            "FOR p IN politicians FILTER p.party == @party RETURN p",
            bind_vars={"party": party},
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


def add_promise(
    db,
    politician_key: str,
    text: str,
    category: str,
    source_url: str = "",
    source_name: str = "",
    date: str = "",
) -> dict:
    """Lägg till ett vallöfte kopplat till en politiker."""
    collection = db.collection("promises")

    doc = {
        "politician_key": politician_key,
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


def get_promises(db, politician_key: str, category: str | None = None) -> list[dict]:
    """Hämta vallöften för en politiker, valfritt filtrerade per kategori."""
    bind_vars = {"politician_key": politician_key}
    query = "FOR p IN promises FILTER p.politician_key == @politician_key"

    if category:
        query += " FILTER p.category == @category"
        bind_vars["category"] = category

    query += " RETURN p"
    cursor = db.aql.execute(query, bind_vars=bind_vars)
    return list(cursor)
