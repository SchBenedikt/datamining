````markdown
# 🏗️ System-Architektur

## Übersicht

Das Unified News Mining System ist ein vollständig integriertes Crawler-System mit separaten Datenbanktabellen, einem einheitlichen Dashboard und zentraler Verwaltung über Docker.

---

## 📐 Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED NEWS MINING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐              ┌──────────────────────┐
│   Heise Crawlers     │              │   Chip Crawlers      │
├──────────────────────┤              ├──────────────────────┤
│ Archive Crawler      │              │ Archive Crawler      │
│ (rückwärts)          │              │ (vorwärts)           │
│ - Start: 2025/10     │              │ - Start: Seite 1     │
│ - Ziel: 2000/01      │              │ - Ziel: Letzte Seite │
│                      │              │                      │
│ Live Crawler         │              │ Live Crawler         │
│ (alle 5 Minuten)     │              │ (alle 10 Minuten)    │
│ - Prüft: Aktuellen   │              │ - Prüft: Seite 1     │
│   Monat              │              │   (neueste)          │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           │ INSERT INTO heise                   │ INSERT INTO chip
           │ (title, url, date,                  │ (title, url, date,
           │  author, category,                  │  author, keywords,
           │  keywords, ...)                     │  description, ...)
           │                                     │
           └─────────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   PostgreSQL DB      │
              ├──────────────────────┤
              │                      │
              │  ┌────────────────┐  │
              │  │  heise table   │  │
              │  │  (10 columns)  │  │
              │  └────────────────┘  │
              │                      │
              │  ┌────────────────┐  │
              │  │  chip table    │  │
              │  │  (12 columns)  │  │
              │  └────────────────┘  │
              │                      │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Streamlit   │  │ Discord Bot │  │ Export      │
│ Dashboard   │  │             │  │ Tools       │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ • Merge     │  │ • Heise     │  │ heise/      │
│   beide     │  │   stats     │  │ export_     │
│   Tabellen  │  │             │  │ articles.py │
│             │  │ • Chip      │  │             │
│ • Filter:   │  │   stats     │  │ chip/       │
│   - Quelle  │  │             │  │ export_     │
│   - Datum   │  │ • Heute     │  │ articles.py │
│   - Autor   │  │   & Total   │  │             │
│   - Kat.    │  │             │  │ Formate:    │
│             │  │ • Updates   │  │ • CSV       │
│ • 20+       │  │   alle 10   │  │ • XLSX      │
│   Viz.      │  │   Minuten   │  │ • JSON      │
│             │  │             │  │ • SQL       │
│ • Export    │  │             │  │             │
│ • AI        │  │             │  │             │
│   Analytics │  │             │  │             │
└─────────────┘  └─────────────┘  └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE STACK                         │
├─────────────────────────────────────────────────────────────────┤
│  heise-archive-crawler   │  heise-live-crawler                  │
│  chip-archive-crawler    │  chip-live-crawler                   │
│  streamlit-dashboard     │  discord-bot                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  DOCKER MANAGEMENT TOOLS (Optional)              │
├─────────────────────────────────────────────────────────────────┤
│  Portainer (Port 9000)   │  Dockge (Port 5001)                  │
│  - Container starten/stoppen/pausieren                          │
│  - Logs in Echtzeit ansehen                                     │
│  - Ressourcen-Monitoring                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Datenfluss

### 1. Crawling-Phase

```
Heise Archive Crawler:
  https://www.heise.de/newsticker/archiv/2025/10
  └─> Extrahiert Artikel-Metadaten
      └─> Speichert in heise-Tabelle
          └─> Geht zu 2025/09, 2025/08, ...

Heise Live Crawler:
  Jede 5 Minuten:
  └─> Prüft aktuellen Monat
      └─> Findet neue Artikel
          └─> Speichert nur Neue (Duplikate-Check via URL)

Chip Archive Crawler:
  https://www.chip.de/news/?p=1
  └─> Extrahiert Artikel-Metadaten
      └─> Speichert in chip-Tabelle
          └─> Geht zu Seite 2, 3, 4, ...

Chip Live Crawler:
  Jede 10 Minuten:
  └─> Prüft Seite 1 (neueste Artikel)
      └─> Findet neue Artikel
          └─> Speichert nur Neue (Duplikate-Check via URL)
```

### 2. Datenbank-Phase

```
PostgreSQL Datenbank:
  ├─> heise-Tabelle
  │   ├─> id, title, url, date, author
  │   ├─> category, keywords, word_count
  │   └─> editor_abbr, site_name
  │
  └─> chip-Tabelle
      ├─> id, url, title, author, date
      ├─> keywords, description, type
      └─> page_level1, page_level2, page_level3, page_template
```

### 3. Visualisierungs-Phase

```
Streamlit Dashboard:
  └─> SELECT * FROM heise
  └─> SELECT * FROM chip
      └─> pd.concat([df_heise, df_chip])
          └─> Filter nach Quelle
              └─> Visualisierungen:
                  ├─> Autoren-Netzwerke
                  ├─> Keyword-Analysen
                  ├─> Zeitanalysen
                  ├─> AI-Analysen
                  └─> Export-Funktionen
```

### 4. Benachrichtigungs-Phase

```
Discord Bot:
  └─> Jede 10 Minuten:
      ├─> SELECT COUNT(*) FROM heise
      ├─> SELECT COUNT(*) FROM chip
      └─> Postet Statistiken im Discord-Channel

E-Mail-Benachrichtigungen:
  └─> Bei Fehlern:
      └─> Sendet Alert an ALERT_EMAIL
```

---

## 📊 Datenbankschema

### Heise-Tabelle

```sql
CREATE TABLE IF NOT EXISTS heise (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    date TEXT,
    author TEXT,
    category TEXT,
    keywords TEXT,
    word_count INTEGER,
    editor_abbr TEXT,
    site_name TEXT
);

CREATE INDEX idx_heise_date ON heise(date);
CREATE INDEX idx_heise_author ON heise(author);
CREATE INDEX idx_heise_category ON heise(category);
```

**Beispieldaten:**
```json
{
  "id": 1,
  "title": "Neue KI-Technologie revolutioniert...",
  "url": "https://www.heise.de/news/...",
  "date": "2025-10-02T10:30:00",
  "author": "Max Mustermann",
  "category": "Künstliche Intelligenz",
  "keywords": "KI, Machine Learning, Innovation",
  "word_count": 450,
  "editor_abbr": "mm",
  "site_name": "heise online"
}
```

### Chip-Tabelle

```sql
CREATE TABLE IF NOT EXISTS chip (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    date TEXT,
    keywords TEXT,
    description TEXT,
    type TEXT,
    page_level1 TEXT,
    page_level2 TEXT,
    page_level3 TEXT,
    page_template TEXT
);

CREATE INDEX idx_chip_date ON chip(date);
CREATE INDEX idx_chip_author ON chip(author);
CREATE INDEX idx_chip_type ON chip(type);
```

**Beispieldaten:**
```json
{
  "id": 1,
  "url": "https://www.chip.de/news/...",
  "title": "Smartphone-Test 2025: Die besten...",
  "author": "CHIP Redaktion",
  "date": "2025-10-02",
  "keywords": "Smartphone, Test, Vergleich",
  "description": "Im großen Vergleichstest...",
  "type": "Test",
  "page_level1": "News",
  "page_level2": "Mobilfunk",
  "page_level3": "Smartphones",
  "page_template": "article"
}
```

---

## 🔧 Komponenten-Details

### Heise Crawler

**Datei:** `heise/main.py` (Archive), `heise/current_crawler.py` (Live)

**Funktionsweise:**
1. Lädt Archive-Seite: `https://www.heise.de/newsticker/archiv/YYYY/MM`
2. Parst HTML mit BeautifulSoup
3. Extrahiert Artikel-Links und Metadaten
4. Prüft Duplikate via URL
5. Speichert neue Artikel in `heise`-Tabelle
6. Bei < 10 Artikel/Tag: E-Mail-Alert

**Besonderheiten:**
- Rückwärts-Crawling (neueste zu älteste)
- Live-Crawler prüft nur aktuellen Monat
- Erkennt Editor-Kürzel (z.B. "mm", "js")
- Erfasst Wortanzahl

### Chip Crawler

**Datei:** `chip/main.py` (Archive), `chip/current_crawler.py` (Live)

**Funktionsweise:**
1. Lädt News-Seite: `https://www.chip.de/news/?p=PAGE`
2. Parst HTML mit BeautifulSoup
3. Extrahiert Artikel-Links und Metadaten aus `<script type="application/ld+json">`
4. Prüft Duplikate via URL
5. Speichert neue Artikel in `chip`-Tabelle

**Besonderheiten:**
- Vorwärts-Crawling (Seite 1 zu Seite N)
- Live-Crawler prüft nur Seite 1
- Extrahiert strukturierte Daten (JSON-LD)
- Erfasst Page-Hierarchie (Level 1-3)

### Streamlit Dashboard

**Datei:** `visualization/streamlit_app.py`

**Funktionsweise:**
1. Lädt Daten aus beiden Tabellen
2. Fügt `source`-Spalte hinzu ('heise' oder 'chip')
3. Merged DataFrames: `pd.concat([df_heise, df_chip])`
4. Bietet Filter-Optionen in Sidebar
5. Generiert Visualisierungen on-the-fly
6. Cached Daten für Performance

**Features:**
- **Übersicht:** KPIs, Statistiken, Trends
- **Zeitanalysen:** Artikel pro Tag/Woche/Monat
- **Autoren-Netzwerke:** NetworkX + Plotly
- **Keyword-Analysen:** Top Keywords, Trends
- **Word Clouds:** Häufigste Begriffe
- **AI-Analysen:** Topic Modeling, Sentiment
- **Suchfunktion:** Volltext-Suche
- **Export:** CSV, Excel, JSON, SQL

### Discord Bot

**Datei:** `heise/bot.py`

**Funktionsweise:**
1. Verbindet zu Discord
2. Jede 10 Minuten:
   - Zählt Artikel in beiden Tabellen
   - Zählt heutige Artikel
   - Zählt Autoren
3. Postet Embed-Message mit Statistiken

**Ausgabe:**
```
📊 News Mining Statistik

📰 Artikel heute: 45 (Heise: 25, Chip: 20)
📚 Artikel gesamt: 12.345 (Heise: 8.000, Chip: 4.345)
✍️ Autoren gesamt: 234

Stand: 02.10.2025 10:30
```

### Export-Tools

**Dateien:** `heise/export_articles.py`, `chip/export_articles.py`

**Funktionsweise:**
1. Verbindet zur Datenbank
2. Liest alle Artikel der jeweiligen Tabelle
3. Konvertiert zu gewünschtem Format
4. Speichert in `data/`-Verzeichnis

**Formate:**
- **CSV:** `data/heise_articles_YYYYMMDD.csv`
- **Excel:** `data/heise_articles_YYYYMMDD.xlsx`
- **JSON:** `data/heise_articles_YYYYMMDD.json`
- **SQL:** `data/heise_articles_YYYYMMDD.sql`

---

## 🐳 Docker-Architektur

### Docker Compose Services

```yaml
services:
  heise-archive-crawler:
    - Führt heise/main.py aus
    - Rückwärts-Crawling
    - Restart: unless-stopped
    
  heise-live-crawler:
    - Führt heise/current_crawler.py aus
    - Prüft alle 5 Minuten
    - Restart: unless-stopped
    
  chip-archive-crawler:
    - Führt chip/main.py aus
    - Vorwärts-Crawling
    - Restart: unless-stopped
    
  chip-live-crawler:
    - Führt chip/current_crawler.py aus
    - Prüft alle 10 Minuten
    - Restart: unless-stopped
    
  streamlit-dashboard:
    - Führt streamlit run aus
    - Port 8501 exposed
    - Volumes für Code-Updates
    
  discord-bot:
    - Führt heise/bot.py aus
    - Postet alle 10 Minuten
    - Restart: unless-stopped
```

### Docker-Netzwerk

```
crawler-network (bridge):
  ├─> heise-archive-crawler
  ├─> heise-live-crawler
  ├─> chip-archive-crawler
  ├─> chip-live-crawler
  ├─> streamlit-dashboard
  └─> discord-bot
```

Alle Container können sich über dieses Netzwerk erreichen und teilen die gleiche `.env`-Datei.

---

## 🔐 Sicherheit

### Umgebungsvariablen

Sensible Daten werden über `.env`-Datei verwaltet:
- Niemals in Git committen (`.gitignore`)
- Nur lesbar für Container
- Verschlüsselte Übertragung (SMTP SSL/TLS)

### Datenbank-Sicherheit

- PostgreSQL-Zugriff nur über Credentials
- Unique Constraints verhindern Duplikate
- Prepared Statements gegen SQL-Injection
- Index auf häufig abgefragte Spalten

### API-Sicherheit

- Keine Authentifizierung (lokaler Zugriff)
- Bei öffentlichem Deployment: OAuth/JWT empfohlen
- Rate-Limiting für API-Endpoints

---

## 📈 Skalierbarkeit

### Horizontale Skalierung

**Weitere Quellen hinzufügen:**
1. Neuen Ordner erstellen (z.B. `golem/`)
2. Crawler-Skripte kopieren und anpassen
3. Neue Tabelle in DB erstellen
4. Service zu `docker-compose.yml` hinzufügen
5. Streamlit lädt automatisch neue Tabelle

**Beispiel:**
```yaml
golem-live-crawler:
  build: .
  container_name: golem-live-crawler
  command: python3 golem/current_crawler.py
  ...
```

### Vertikale Skalierung

**Performance-Optimierungen:**
- Datenbank-Indizes auf häufig abgefragte Spalten
- Streamlit-Caching für große Datasets
- Batch-Inserts statt einzelner INSERTs
- Connection Pooling für Datenbank

### Load Balancing

**Bei hoher Last:**
- Mehrere Streamlit-Instanzen hinter Nginx
- PostgreSQL Read Replicas
- Redis für Session-Management
- CDN für statische Assets

---

## 🔄 Erweiterbarkeit

### Plugin-Architektur

Das System ist modular aufgebaut:

```
plugins/
├── crawlers/
│   ├── heise_crawler.py
│   ├── chip_crawler.py
│   └── custom_crawler.py  <- Neuer Crawler
│
├── exporters/
│   ├── csv_exporter.py
│   ├── json_exporter.py
│   └── pdf_exporter.py    <- Neuer Exporter
│
└── visualizations/
    ├── network_graph.py
    ├── time_series.py
    └── custom_viz.py      <- Neue Visualisierung
```

### API-Endpunkte

**Bestehende:**
- `/stats` - Gesamtstatistiken
- `/articles` - Alle Artikel

**Erweiterbar:**
- `/api/v1/heise/articles` - Nur Heise
- `/api/v1/chip/articles` - Nur Chip
- `/api/v1/search?q=keyword` - Suche
- `/api/v1/authors` - Autoren-Liste
- `/api/v1/keywords` - Keyword-Trends

---

## 🎯 Best Practices

### Crawler

1. **Rate Limiting:** Pause zwischen Requests (1-2 Sekunden)
2. **User-Agent:** Identifizierbar als Bot
3. **Robots.txt:** Respektieren der Crawling-Regeln
4. **Error Handling:** Graceful Degradation bei Fehlern
5. **Logging:** Ausführliche Logs für Debugging

### Datenbank

1. **Normalisierung:** Separate Tabellen für bessere Performance
2. **Indizes:** Auf häufig abgefragte Spalten
3. **Backups:** Regelmäßige Datenbank-Backups
4. **Constraints:** UNIQUE auf URL verhindert Duplikate
5. **Transactions:** ACID-Eigenschaften nutzen

### Streamlit

1. **Caching:** `@st.cache_data` für teure Operationen
2. **Lazy Loading:** Große Datasets erst bei Bedarf laden
3. **Pagination:** Bei sehr vielen Artikeln
4. **Responsive:** Mobile-freundliches Layout
5. **Error Handling:** Try-Except für alle DB-Queries

---

## 🛠️ Monitoring & Debugging

### Logs

```bash
# Docker Logs
docker-compose logs -f [service-name]

# Spezifischer Crawler
docker-compose logs -f heise-live-crawler

# Alle Services
docker-compose logs -f
```

### Metriken

**Wichtige KPIs:**
- Artikel pro Tag
- Crawler-Erfolgsrate
- Duplikate-Erkennungsrate
- API-Response-Zeit
- Streamlit-Load-Zeit

### Alerts

**E-Mail-Benachrichtigungen bei:**
- Weniger als 10 Artikel/Tag
- Datenbank-Verbindungsfehler
- Crawler-Crashes
- Disk Space < 10%

---

## 🚀 Deployment-Optionen

### Option 1: Lokales Deployment

```bash
# Crawlers manuell starten
python3 heise/main.py
python3 chip/main.py

# Streamlit starten
streamlit run visualization/streamlit_app.py
```

### Option 2: Docker Deployment

```bash
# Alle Services starten
docker-compose up -d

# Logs überwachen
docker-compose logs -f
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

**Empfohlene Architektur:**
- EC2/Compute Engine/VM für Container
- RDS/Cloud SQL/Azure DB für PostgreSQL
- CloudWatch/Logging für Monitoring
- S3/Cloud Storage für Exports
- Load Balancer für Streamlit

---

## 📚 Weiterführende Dokumentation

- **[README.md](README.md)** - Hauptdokumentation
- **[QUICKSTART.md](QUICKSTART.md)** - Schnellstart-Anleitung
- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker-Setup-Details
- **[SECURITY.md](SECURITY.md)** - Sicherheitsrichtlinien

---

## 🤝 Beiträge

Beiträge sind willkommen! Bitte öffnen Sie ein Issue oder Pull Request auf GitHub.

**Contribution Guidelines:**
1. Fork das Repository
2. Erstellen Sie einen Feature-Branch
3. Committen Sie Ihre Änderungen
4. Pushen Sie zum Branch
5. Öffnen Sie einen Pull Request

---

**Stand:** Oktober 2025  
**Version:** 2.0 (Separate Tables Architecture)  
**Status:** ✅ Production Ready

````