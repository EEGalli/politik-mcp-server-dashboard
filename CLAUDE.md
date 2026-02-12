# Instruktioner till Claude

- Svara på ett pedagogiskt och konkret sätt
- Undvik onödiga nedladdningar och annat krångel
- Var ödmjuk
- Gissa aldrig, säg till om du är osäker på något
- Innan en import tas bort: sök alltid efter att symbolen verkligen är oanvänd i hela filen

---

# Projektkontext

## Vad projektet gör
MCP-server för att analysera politikers sociala medier och jämföra med deras vallöften.
Använder videoanalys-pipeline från syster-projektet (video-content-analyzer).

## Arkitektur
- MCP-server med FastMCP (Python)
- ArangoDB för lagring (politiker, löften, analyser, jämförelser)
- Grafrelationer: politiker → löften, analyser → jämförelser → löften

## Python-miljö
- Ingen virtualenv – Python körs globalt

## Säkerhet
- Lägg aldrig lösenord eller tokens i chatten
- .env-filer ska aldrig committas
