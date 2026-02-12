"""
Promptmallar för jämförelseanalys.
Används av AI-klienten för att jämföra videoinnehåll med vallöften.
"""

COMPARE_CONTENT_VS_PROMISE = """
Du är en politisk analyst. Jämför följande videoinnehåll med ett vallöfte.

## Videoinnehåll (från sociala medier)
Politiker: {politician_name} ({party})
Plattform: {platform}
Transkription:
{transcription}

Sammanfattning:
{summary}

## Vallöfte
Kategori: {promise_category}
Löfte: {promise_text}
Källa: {promise_source}

## Uppgift
Bedöm hur väl videoinnehållet stämmer överens med vallöftet.

Svara i JSON-format:
{{
    "match_type": "supports" | "contradicts" | "unrelated",
    "match_score": 0.0-1.0,
    "explanation": "Kort förklaring på svenska"
}}
"""

CONSISTENCY_SUMMARY = """
Du är en politisk analyst. Sammanfatta följande konsistensrapport för en politiker.

## Rapport
Politiker: {politician_name} ({party})
Antal vallöften: {total_promises}
Antal analyserade videor: {total_analyses}
Antal jämförelser: {total_comparisons}
Stöder: {supports_count}
Motsäger: {contradicts_count}
Orelaterade: {unrelated_count}
Konsistenspoäng: {consistency_score}

## Motsägelser
{contradictions}

## Stöd
{supports}

Ge en balanserad, saklig sammanfattning på svenska. Undvik partiskhet.
"""
