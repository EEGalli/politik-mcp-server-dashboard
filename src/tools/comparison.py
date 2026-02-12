"""
MCP-tools för att jämföra videoinnehåll med vallöften.
"""

from datetime import datetime, timezone


def save_comparison(
    db,
    analysis_key: str,
    promise_key: str,
    match_score: float,
    match_type: str,
    explanation: str,
) -> dict:
    """Spara en jämförelse mellan en videoanalys och ett vallöfte."""
    collection = db.collection("comparisons")

    doc = {
        "_from": f"analyses/{analysis_key}",
        "_to": f"promises/{promise_key}",
        "match_score": match_score,
        "match_type": match_type,  # "supports", "contradicts", "unrelated"
        "explanation": explanation,
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }
    result = collection.insert(doc)
    doc["_key"] = result["_key"]
    return doc


def get_consistency_report(db, politician_key: str) -> dict:
    """
    Generera en konsistensrapport för en politiker.
    Sammanställer alla jämförelser mellan deras videoinnehåll och vallöften.
    """
    query = """
    LET politician = DOCUMENT(CONCAT("politicians/", @politician_key))
    LET party = DOCUMENT(CONCAT("parties/", politician.party_key))

    LET promises = (
        FOR p IN promises
            FILTER p.party_key == politician.party_key
            RETURN p
    )

    LET analyses = (
        FOR a IN analyses
            FILTER a.politician_key == @politician_key
            RETURN a
    )

    LET comparisons = (
        FOR a IN analyses
            FILTER a.politician_key == @politician_key
            FOR c IN comparisons
                FILTER c._from == CONCAT("analyses/", a._key)
                LET promise = DOCUMENT(c._to)
                RETURN MERGE(c, {
                    analysis_summary: a.summary,
                    promise_text: promise.text,
                    promise_category: promise.category
                })
    )

    LET supports = (FOR c IN comparisons FILTER c.match_type == "supports" RETURN c)
    LET contradicts = (FOR c IN comparisons FILTER c.match_type == "contradicts" RETURN c)
    LET unrelated = (FOR c IN comparisons FILTER c.match_type == "unrelated" RETURN c)

    RETURN {
        politician: politician.name,
        party: party.name,
        party_abbreviation: party.abbreviation,
        total_promises: LENGTH(promises),
        total_analyses: LENGTH(analyses),
        total_comparisons: LENGTH(comparisons),
        supports_count: LENGTH(supports),
        contradicts_count: LENGTH(contradicts),
        unrelated_count: LENGTH(unrelated),
        consistency_score: LENGTH(comparisons) > 0
            ? LENGTH(supports) / LENGTH(comparisons)
            : null,
        contradictions: contradicts,
        supports: supports
    }
    """
    cursor = db.aql.execute(query, bind_vars={"politician_key": politician_key})
    results = list(cursor)
    return results[0] if results else {}
