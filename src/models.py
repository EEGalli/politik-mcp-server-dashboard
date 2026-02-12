"""
Pydantic-modeller for all data i politik-mcp-server.
Anvands for validering bade in och ut ur MCP-tools.
"""

from pydantic import BaseModel, Field


# -- Parti --

class Party(BaseModel):
    name: str = Field(description="Partiets fullstandiga namn, t.ex. 'Moderaterna'")
    abbreviation: str = Field(description="Partiforkortning, t.ex. 'M'")
    block: str = Field(default="", description="Politiskt block: vanster, hoger, mitt")
    website_url: str = ""


class PartyInDB(Party):
    _key: str = ""
    created_at: str = ""


# -- Politiker --

class SocialMedia(BaseModel):
    tiktok: str = ""
    instagram: str = ""
    twitter: str = ""


class Politician(BaseModel):
    name: str = Field(description="Politikerns fullstandiga namn")
    party_key: str = Field(description="Partiets databas-nyckel")
    social_media: SocialMedia = Field(default_factory=SocialMedia)


class PoliticianInDB(Politician):
    _key: str = ""
    created_at: str = ""


# -- Valloften --

class Promise(BaseModel):
    party_key: str = Field(description="Partiets databas-nyckel")
    text: str = Field(description="Valloftets text")
    category: str = Field(description="Kategori: Skatt, Sjukvard, Skola, etc.")
    source_url: str = ""
    source_name: str = ""
    date: str = ""


class PromiseInDB(Promise):
    _key: str = ""
    created_at: str = ""


# -- Extraherade valloften (fran scraping) --

class ExtractedPromise(BaseModel):
    """Ett enskilt vallofte extraherat fran en webbsida eller PDF."""
    text: str = Field(description="Kort, konkret och neutral policyformulering baserad pa kalltexten")
    category: str = Field(description="Kategori: Sjukvard, Skola & Utbildning, Rattsvasende, Ekonomi, Skatt, Arbetsmarknad, Socialpolitik, Forsvar, Migration, Klimat & Miljo, Bostader, Infrastruktur, Kultur, Jamstalldhet, EU & Utrikes, Digitalisering, Landsbygd, Ovrigt")


class ExtractedPromises(BaseModel):
    """Lista av valloften extraherade fran en sida."""
    promises: list[ExtractedPromise] = Field(description="Lista av konkreta valloften")


# -- Videoanalys --

class VideoAnalysis(BaseModel):
    politician_key: str = Field(description="Politikerns databas-nyckel")
    video_file: str = ""
    transcription: str = Field(description="Full transkription")
    summary: str = Field(description="Sammanfattning av innehallet")
    category: str = ""
    political_color: str = ""
    platform: str = ""


class VideoAnalysisInDB(VideoAnalysis):
    _key: str = ""
    analyzed_at: str = ""


# -- Jamforelse --

class ComparisonResult(BaseModel):
    """Resultat fran en AI-driven jamforelse mellan videoinnehall och vallofte."""
    match_type: str = Field(
        description="'supports' (stoder), 'contradicts' (motsager), eller 'unrelated' (orelaterat)"
    )
    match_score: float = Field(
        ge=0.0, le=1.0,
        description="Matchpoang 0.0-1.0"
    )
    explanation: str = Field(
        description="Forklaring av jamforelsen pa svenska"
    )
    key_quotes: list[str] = Field(
        default_factory=list,
        description="Relevanta citat fran transkriptionen"
    )


# -- Konsistensrapport --

class ConsistencyReport(BaseModel):
    politician_name: str
    party: str
    total_promises: int = 0
    total_analyses: int = 0
    total_comparisons: int = 0
    supports_count: int = 0
    contradicts_count: int = 0
    unrelated_count: int = 0
    consistency_score: float | None = None
