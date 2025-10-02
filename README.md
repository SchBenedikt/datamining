<a href="https://discord.com/invite/Q6Nn2z3tUP">
  <img src="https://discord.com/api/guilds/1346160903304773703/widget.png?style=banner2" width="250"/>
</a>
<a href="https://deepnote.com/app/schachner/Web-Crawler-d5025a36-3829-4c12-ad2d-b81aa84bd217?utm_source=app-settings&utm_medium=product-embed&utm_campaign=data-app&utm_content=d5025a36-3829-4c12-ad2d-b81aa84bd217&__embedded=true">
  <img src="https://img.shields.io/badge/Open%20in-Deepnote-blue?style=for-the-badge&logo=deepnote" width="250"/>
</a>

# 🗞️ Unified News Mining System

Ein umfassendes, einheitliches Crawler-System zum Sammeln und Analysieren von Nachrichtenartikeln von **Heise.de** und **Chip.de**.

---

## 📑 Inhaltsverzeichnis

- [🎯 Quick Links](#-quick-links)
- [✨ Key Features](#-key-features)
- [🌍 Zweck & Funktionalität](#-zweck--funktionalität)
- [🚀 Installation & Setup](#-installation--setup)
  - [Voraussetzungen](#1️⃣-voraussetzungen)
  - [Repository klonen](#2️⃣-repository-klonen)
  - [Dependencies installieren](#3️⃣-dependencies-installieren)
  - [Umgebungsvariablen konfigurieren](#4️⃣-umgebungsvariablen-konfigurieren)
  - [Datenbank Setup](#5️⃣-datenbank-setup)
- [🛠 Verwendung](#-verwendung)
  - [Crawler starten](#crawler-starten)
  - [Streamlit Dashboard](#streamlit-dashboard)
  - [Discord Bot](#discord-bot)
  - [API Endpoints](#api-endpoints)
  - [Daten exportieren](#daten-exportieren)
- [🐳 Docker Deployment](#-docker-deployment)
- [🏗 Datenbankschema](#-datenbankschema)
- [📊 Streamlit Features](#-streamlit-features)
- [📂 Projektstruktur](#-projektstruktur)
- [🔧 Verwaltung mit Docker-Tools](#-verwaltung-mit-docker-tools)
- [❗ Troubleshooting](#-troubleshooting)
- [🗂️ Beispiele & Screenshots](#️-beispiele--screenshots)
- [📜 Lizenz](#-lizenz)
- [🙋 Über uns](#-über-uns)

---

## 🎯 Quick Links

- 📖 **[Quick Start Guide](QUICKSTART.md)** - In 5 Minuten starten
- 🏗️ **[Architecture](ARCHITECTURE.md)** - Systemarchitektur und Datenfluss
- 🐳 **[Docker Setup](DOCKER_SETUP.md)** - Deployment mit Docker

---

## ✨ Key Features

- ✅ **Separate Tabellen** - Zwei dedizierte PostgreSQL-Tabellen (heise & chip)
- ✅ **Live Monitoring** - Automatische Prüfung alle 5-10 Minuten
- ✅ **Single Dashboard** - Eine Streamlit-App für beide Quellen
- ✅ **Source Filtering** - Anzeige von Heise, Chip oder beiden
- ✅ **Discord Bot** - Echtzeit-Statistik-Updates
- ✅ **Daten exportieren** - CSV, Excel, JSON für beide Quellen
- ✅ **Docker Ready** - Deployment mit einem Befehl
- ✅ **AI Analytics** - Powered by Google Generative AI
- ✅ **Autoren-Netzwerke** - Visualisierung von Autoren-Verbindungen
- ✅ **Keyword-Analysen** - Schlagwortverteilung und Trends
- ✅ **Zeitanalysen** - Zeitbasierte Diagramme und Graphen
- ✅ **Suchfunktion** - Volltext-Suche in allen Artikeln
- ✅ **Filterfunktionen** - Nach Quelle, Datum, Autor, Kategorie

---

## 🌍 Zweck & Funktionalität

Das **News Mining System** ist darauf ausgelegt, automatisch Nachrichtenartikel aus mehreren Quellen zu extrahieren und zu speichern. Die Hauptziele sind:

- 📡 **Datensammlung** - Erfassung historischer Nachrichtenartikel von Heise.de und Chip.de
- 🏛 **Strukturierte Speicherung** - Artikel beider Quellen in separaten PostgreSQL-Tabellen
- 🔍 **Metadaten-Extraktion** - Erfassung von Titel, Autor, Kategorie, Schlagwörtern, Wortanzahl und mehr
- 🔄 **Inkrementelles Crawling** - Erkennung von Duplikaten und Speicherung nur neuer Artikel
- 🔔 **Benachrichtigungen** - E-Mail-Benachrichtigung bei Fehlern während des Crawling-Prozesses
- 🎨 **Verbesserte Terminal-Ausgabe** - Nutzung von PyFiglet für bessere Lesbarkeit
- 📤 **Datenexport** - Export als CSV, JSON, XLSX mit Quellenfilterung
- 🖥 **API** - Bereitstellung von Statistiken und kompletten Datensätzen
- 🤖 **AI Analytics** - Erweiterte Analysen mit Google Generative AI für Topic Modeling, Sentiment Analysis und Trend Detection
- 🎯 **Einheitliches Dashboard** - Eine Streamlit-Anwendung für beide Quellen
- 🤖 **Discord Bot** - Echtzeit-Statistiken für beide Quellen in Discord
- 📊 **Umfangreiche Visualisierungen** - Über 20 verschiedene Diagramme, Graphen und Darstellungen
- 🕸️ **Autoren-Netzwerke** - Visualisierung von Verbindungen zwischen Autoren
- 📈 **Trend-Analysen** - Zeitbasierte Analysen und Vorhersagen

Ein API-Endpoint wird ebenfalls bereitgestellt, der die gecrawlten Daten und Statistiken anzeigen kann.

---

## 🚀 Installation & Setup

### 1️⃣ Voraussetzungen

🔹 **Python 3.8+** (empfohlen: Python 3.11)

🔹 **PostgreSQL 13+** (lokal oder remote)

🔹 **Git** (für das Klonen des Repositories)

🔹 **pip3** (Python Package Manager)

Optional:
- 🐳 **Docker & Docker Compose** (für containerisiertes Deployment)
- 🎮 **Discord Bot Token** (für Discord-Integration)
- 🤖 **Google API Key** (für KI-Analysen)

### 2️⃣ Repository klonen

```bash
git clone https://github.com/SchBenedikt/datamining.git
cd datamining
```

### 3️⃣ Dependencies installieren

Installieren Sie alle erforderlichen Python-Bibliotheken:

```bash
pip3 install -r requirements.txt
```

Für die Streamlit-Anwendung (erweiterte Visualisierungen):

```bash
cd visualization
pip3 install -r requirements_streamlit.txt
cd ..
```

### 4️⃣ Umgebungsvariablen konfigurieren

Erstellen Sie eine `.env`-Datei im Hauptverzeichnis mit folgenden Variablen:

```env
# Datenbank-Konfiguration
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432

# E-Mail-Benachrichtigungen (optional)
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ALERT_EMAIL=recipient@example.com

# Discord Bot (optional)
DISCORD_TOKEN=your_discord_bot_token
CHANNEL_ID=your_discord_channel_id

# Google AI (optional, für erweiterte Analysen)
GOOGLE_API_KEY=your_google_api_key
```

**Hinweise:**
- Für Gmail verwenden Sie ein [App-Passwort](https://support.google.com/accounts/answer/185833)
- Discord Token erhalten Sie im [Discord Developer Portal](https://discord.com/developers/applications)
- Google API Key erstellen Sie in der [Google Cloud Console](https://console.cloud.google.com)

### 5️⃣ Datenbank Setup

Erstellen Sie die PostgreSQL-Datenbank:

```bash
# PostgreSQL-Konsole öffnen
psql -U postgres

# Datenbank erstellen
CREATE DATABASE your_database_name;

# Beenden
\q
```

Die benötigten Tabellen werden automatisch beim ersten Start der Crawler erstellt.

**Manuelle Tabellenerstellung (optional):**

```sql
-- Heise-Tabelle
CREATE TABLE IF NOT EXISTS heise (
    id SERIAL PRIMARY KEY,
    title TEXT,
    url TEXT UNIQUE,
    date TEXT,
    author TEXT,
    category TEXT,
    keywords TEXT,
    word_count INTEGER,
    editor_abbr TEXT,
    site_name TEXT
);

-- Chip-Tabelle
CREATE TABLE IF NOT EXISTS chip (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
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
```

---

## 🛠 Verwendung

### Crawler starten

#### Heise Archive Crawler (crawlt rückwärts vom neuesten zum ältesten)

```bash
cd heise
python3 main.py
```

**Beispiel Terminal-Ausgabe:**

```
[INFO] Crawle URL: https://www.heise.de/newsticker/archiv/2025/10
[INFO] Gefundene Artikel (insgesamt): 55
2025-10-02 10:30:15 [INFO] Verarbeite 16 Artikel für den Tag 2025-10-02
2025-10-02 10:30:15 [INFO] 2025-10-02T20:00:00 - article-name
```

Falls weniger als 10 Artikel pro Tag gefunden werden, wird eine E-Mail gesendet.

#### Heise Live Crawler (prüft alle 5 Minuten auf neue Artikel)

```bash
cd heise
python3 current_crawler.py
```

**Beispiel Terminal-Ausgabe:**

```
[INFO] Crawle URL: https://www.heise.de/newsticker/archiv/2025/10
[INFO] Gefundene Artikel (insgesamt): 55
2025-10-02 10:35:00 [INFO] Aktueller Crawl-Durchlauf abgeschlossen.
2025-10-02 10:35:00 [INFO] Warte 300 Sekunden bis zum nächsten Crawl.
```

#### Chip Archive Crawler (crawlt von Seite 1 aufwärts)

```bash
cd chip
python3 main.py
```

#### Chip Live Crawler (prüft alle 10 Minuten auf neue Artikel)

```bash
cd chip
python3 current_crawler.py
```

---

### Streamlit Dashboard

Starten Sie das interaktive Streamlit-Dashboard mit Unterstützung für beide Quellen:

```bash
cd visualization
streamlit run streamlit_app.py
```

Das Dashboard wird auf `http://localhost:8501` geöffnet.

---

### Discord Bot

Starten Sie den Discord Bot für Echtzeit-Statistik-Updates:

```bash
cd heise
python3 bot.py
```

**Der Bot bietet:**
- Gesamtanzahl der Artikel beider Quellen
- Heutige Artikel-Anzahl beider Quellen
- Autoren-Statistiken
- Updates alle 10 Minuten

---

### API Endpoints

Der API-Server startet automatisch beim Ausführen von `heise/main.py`. Statistiken können hier abgerufen werden:

```
http://127.0.0.1:6600/stats
```

**Manueller Start der API:**

```bash
cd heise
python3 api.py
```

---

### Daten exportieren

Sie können die Daten für jede Quelle als CSV, JSON oder XLSX-Datei exportieren.

**Heise-Artikel exportieren:**

```bash
cd heise
python3 export_articles.py
```

**Chip-Artikel exportieren:**

```bash
cd chip
python3 export_articles.py
```

Exportierte Artikel werden im `data/`-Verzeichnis gespeichert.

---

## 🐳 Docker Deployment

### Alle Services mit einem Befehl starten

```bash
docker-compose up -d
```

### Einzelne Services verwalten

```bash
# Heise Archive Crawler starten
docker-compose up -d heise-archive-crawler

# Chip Live Crawler starten
docker-compose up -d chip-live-crawler

# Streamlit Dashboard starten
docker-compose up -d streamlit-dashboard

# Discord Bot starten
docker-compose up -d discord-bot
```

### Logs ansehen

```bash
# Alle Services
docker-compose logs -f

# Spezifischer Service
docker-compose logs -f heise-live-crawler
```

### Services stoppen

```bash
# Alle Services stoppen
docker-compose down

# Spezifischer Service
docker-compose stop heise-archive-crawler
```

### Dashboard aufrufen

Nach dem Start ist das Streamlit-Dashboard unter folgender Adresse erreichbar:

```
http://localhost:8501
```

---

## 🏗 Datenbankschema

Die Datenbank verwendet **zwei separate Tabellen** für bessere Organisation:

### Heise-Tabelle

| Spalte       | Typ    | Beschreibung                |
| ------------ | ------ | --------------------------- |
| id           | SERIAL | Eindeutige ID               |
| title        | TEXT   | Artikel-Titel               |
| url          | TEXT   | Artikel-URL (eindeutig)     |
| date         | TEXT   | Veröffentlichungsdatum      |
| author       | TEXT   | Autor(en)                   |
| category     | TEXT   | Kategorie                   |
| keywords     | TEXT   | Schlagwörter                |
| word\_count  | INT    | Wortanzahl                  |
| editor\_abbr | TEXT   | Redakteur-Kürzel            |
| site\_name   | TEXT   | Website-Name                |

### Chip-Tabelle

| Spalte         | Typ    | Beschreibung                |
| -------------- | ------ | --------------------------- |
| id             | SERIAL | Eindeutige ID               |
| url            | TEXT   | Artikel-URL (eindeutig)     |
| title          | TEXT   | Artikel-Titel               |
| author         | TEXT   | Autor(en)                   |
| date           | TEXT   | Veröffentlichungsdatum      |
| keywords       | TEXT   | Schlagwörter                |
| description    | TEXT   | Artikel-Beschreibung        |
| type           | TEXT   | Artikel-Typ                 |
| page\_level1   | TEXT   | Seitenebene 1               |
| page\_level2   | TEXT   | Seitenebene 2               |
| page\_level3   | TEXT   | Seitenebene 3               |
| page\_template | TEXT   | Seiten-Template             |

**Hinweis:** Das Streamlit-Dashboard führt Daten aus beiden Tabellen zusammen für einheitliche Ansicht.

---

## 📊 Streamlit Features

Das Dashboard bietet über **20 verschiedene Funktionen und Visualisierungen**:

### 📈 Visualisierungen

- **Autoren-Netzwerke** (🕸️) - Interaktive Netzwerkgraphen zeigen Verbindungen zwischen Autoren
- **Keyword-Analysen** (🔑) - Häufigkeitsverteilung der wichtigsten Schlagwörter
- **Word Clouds** - Visuelle Darstellung der häufigsten Begriffe
- **Zeitanalysen** (📅) - Artikel-Veröffentlichungen über Zeit
- **Trend-Analysen** - Vorhersagen und Mustererkennungen
- **KI-Analysen** (🤖) - Topic Modeling, Sentiment Analysis
- **Sentiment-Analyse** - Stimmungsanalyse der Artikel
- **Topic Clustering** - Automatische Themengruppierung
- **Content-Empfehlungen** - Ähnliche Artikel finden
- **Performance-Metriken** (⚡) - System-Statistiken

### 🔧 Interaktive Features

- **Quellenfilter** - Heise, Chip oder beide anzeigen
- **Suchfunktion** (🔍) - Volltext-Suche in Artikeln
- **Datumsbereich-Filter** - Zeitbasierte Filterung
- **Kategoriefilter** - Nach Kategorie filtern
- **Autorenfilter** - Nach Autor filtern
- **Export-Funktion** - CSV, Excel, JSON
- **SQL-Abfragen** (🔧) - Eigene Abfragen ausführen
- **Cache-Management** - Daten-Cache leeren

### 📥 Export-Optionen

- CSV-Export mit Quelleninfo
- Excel-Export (.xlsx)
- JSON-Export
- SQL-Export
- Gefilterte Exports möglich

---

## 📂 Projektstruktur

```
📂 datamining/
├── 📂 heise/                          # Heise-Crawler und verwandte Skripte
│   ├── 📄 main.py                     # Archive Crawler (rückwärts)
│   ├── 📄 current_crawler.py          # Live Crawler (alle 5 Minuten)
│   ├── 📄 bot.py                      # Discord Bot
│   ├── 📄 api.py                      # API-Funktionalitäten
│   ├── 📄 notification.py             # E-Mail-Benachrichtigungen
│   ├── 📄 export_articles.py          # Export-Funktionalität
│   ├── 📄 test_notification.py        # Benachrichtigungs-Test
│   └── 📂 templates/                  # HTML-Templates
│       ├── 📄 news_feed.html
│       └── 📄 query.html
├── 📂 chip/                           # Chip-Crawler und verwandte Skripte
│   ├── 📄 main.py                     # Archive Crawler (vorwärts)
│   ├── 📄 current_crawler.py          # Live Crawler (alle 10 Minuten)
│   ├── 📄 notification.py             # E-Mail-Benachrichtigungen
│   └── 📄 export_articles.py          # Export-Funktionalität
├── 📂 visualization/                  # Einheitliches Streamlit-Dashboard
│   ├── 📄 streamlit_app.py            # Haupt-Streamlit-Anwendung
│   └── 📄 requirements_streamlit.txt  # Streamlit-Dependencies
├── 📂 data/                           # Export-Verzeichnis
├── 📂 docker/                         # Docker-Konfigurationen (falls vorhanden)
├── 📄 docker-compose.yml              # Docker Compose Konfiguration
├── 📄 Dockerfile                      # Docker Image Definition
├── 📄 requirements.txt                # Python-Dependencies
├── 📄 .env                            # Umgebungsvariablen (manuell erstellen)
├── 📄 .gitignore                      # Git-Ignore-Datei
├── 📄 README.md                       # Diese Datei
├── 📄 QUICKSTART.md                   # Schnellstart-Anleitung
├── 📄 ARCHITECTURE.md                 # Systemarchitektur
├── 📄 DOCKER_SETUP.md                 # Docker-Setup-Anleitung
├── 📄 SECURITY.md                     # Sicherheitsrichtlinien
└── 📄 LICENSE                         # Lizenz (GNU GPL)
```

---

## 🔧 Verwaltung mit Docker-Tools

Für die zentrale Verwaltung Ihrer Docker-Container empfehlen wir folgende 3rd-Party-Lösungen:

### 🏆 Portainer (Empfohlen)

**Installation:**

```bash
docker volume create portainer_data

docker run -d \
  -p 9000:9000 \
  --name portainer \
  --restart always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest
```

**Zugriff:** `http://localhost:9000`

**Features:**
- Web-basierte GUI für Container-Management
- Logs in Echtzeit ansehen
- Container starten/stoppen/pausieren
- Ressourcen-Monitoring
- Stack-Management (Docker Compose)
- Benutzerfreundlich

### 🎨 Dockge (Alternative)

**Installation:**

```bash
docker run -d \
  -p 5001:5001 \
  --name dockge \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v dockge_data:/app/data \
  louislam/dockge:1
```

**Zugriff:** `http://localhost:5001`

**Features:**
- Moderne Alternative zu Portainer
- Docker Compose fokussiert
- Einfache Benutzeroberfläche
- Live-Logs

### 🚢 Yacht

**Installation:**

```bash
docker volume create yacht

docker run -d \
  -p 8000:8000 \
  --name yacht \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v yacht:/config \
  selfhostedpro/yacht
```

**Zugriff:** `http://localhost:8000`

**Features:**
- Self-hosted Docker-Management
- Template-basiert
- Clean UI

---

## ❗ Troubleshooting

### Problem: Verbindungsfehler zur Datenbank

**Lösung:**
1. Überprüfen Sie die `.env`-Datei auf korrekte Datenbank-Credentials
2. Stellen Sie sicher, dass PostgreSQL läuft:
   ```bash
   # macOS
   brew services list
   
   # Linux
   sudo systemctl status postgresql
   ```
3. Testen Sie die Verbindung:
   ```bash
   psql -U $DB_USER -d $DB_NAME -h $DB_HOST
   ```

### Problem: Keine Daten im Streamlit-Dashboard

**Lösung:**
1. Überprüfen Sie, ob Tabellen Daten enthalten:
   ```sql
   SELECT COUNT(*) FROM heise;
   SELECT COUNT(*) FROM chip;
   ```
2. Löschen Sie den Streamlit-Cache mit der Schaltfläche "🔄 Cache leeren"
3. Starten Sie die Streamlit-App neu

### Problem: E-Mail-Benachrichtigungen funktionieren nicht

**Lösung:**
1. Für Gmail: Verwenden Sie ein [App-Passwort](https://support.google.com/accounts/answer/185833)
2. Testen Sie die Benachrichtigungsfunktion:
   ```bash
   cd heise
   python3 test_notification.py
   ```
3. Überprüfen Sie SMTP-Einstellungen in `.env`

### Problem: Discord Bot antwortet nicht

**Lösung:**
1. Überprüfen Sie `DISCORD_TOKEN` und `CHANNEL_ID` in `.env`
2. Stellen Sie sicher, dass der Bot die richtigen Permissions hat
3. Überprüfen Sie die Bot-Logs auf Fehler

### Problem: Docker-Container starten nicht

**Lösung:**
1. Überprüfen Sie Docker-Logs:
   ```bash
   docker-compose logs
   ```
2. Stellen Sie sicher, dass alle Ports verfügbar sind
3. Überprüfen Sie die `.env`-Datei

### Problem: "Tabelle existiert nicht"

**Lösung:**
Führen Sie einen Crawler aus, um die Tabelle zu erstellen:
```bash
cd heise
python3 main.py
```

---

## 🗂️ Beispiele & Screenshots

(mit Tableau und DeepNote, Stand März 2025)

![image](https://github.com/user-attachments/assets/ce6ceae0-bdf4-499c-9577-973017bb1eff)

![image](https://github.com/user-attachments/assets/3affd472-8475-4534-99e6-54500493418c)

![image](https://github.com/user-attachments/assets/984babc4-d264-44be-8534-17fdae1f8d5f)

![image](https://github.com/user-attachments/assets/0c1d7a13-0f28-497c-afb3-048ee0a309e7)

![image](https://github.com/user-attachments/assets/ba9a3180-4ae8-4ab3-b4ae-3e81f4621c23)

![image](https://github.com/user-attachments/assets/85ecd8a3-1f31-49d0-ae3a-efdfd98bef21)

![image](https://github.com/user-attachments/assets/1d5c57f7-72be-4aca-8f03-d4fba8bfba9d)

![image](https://github.com/user-attachments/assets/cde65d2c-2b22-481d-9ba4-1c4086eb3f23)

![image](https://github.com/user-attachments/assets/10c87c9c-d444-487c-992f-73d3d4b4a185)

### Deepnote:

Wir haben auch einige Graphen mit [Deepnote](https://deepnote.com/app/schachner/Web-Crawler-d5025a36-3829-4c12-ad2d-b81aa84bd217?utm_source=app-settings&utm_medium=product-embed&utm_campaign=data-app&utm_content=d5025a36-3829-4c12-ad2d-b81aa84bd217&__embedded=true) generiert (❗ nur mit zufälligen 10.000 Zeilen ❗)

![image](https://github.com/user-attachments/assets/ea99ead8-0b48-47d0-8ddc-7c8ce3bd6b53)

Schauen Sie sich auch die [data/Datamining_Heise web crawler-3.twb](https://github.com/SchBenedikt/datamining/blob/3f3fe413aeff25a1ae024215745ed6fa82fc2add/data/Datamining_Heise%20web%20crawler-3.twb)-Datei mit einem Auszug von Analysen an.

---

## 📜 Lizenz

Dieses Programm ist lizenziert unter **GNU GENERAL PUBLIC LICENSE**

Siehe [LICENSE](LICENSE) für weitere Details.

---

## 🙋 Über uns

Dieses Projekt wurde von uns beiden innerhalb weniger Tage programmiert und wird ständig weiterentwickelt:
- https://github.com/schBenedikt
- https://github.com/schVinzenz

### 📬 Kontakt

Zögern Sie nicht, uns zu kontaktieren, wenn Sie Fragen, Feedback haben oder einfach nur Hallo sagen möchten!

📧 E-Mail: [server@schächner.de](mailto:server@schächner.de)

🌐 Website:
- https://technik.schächner.de
- https://benedikt.schächner.de
- https://vinzenz.schächner.de

### 💖 Besonderer Dank

Die Idee für unseren Heise News Crawler stammt von David Kriesel und seiner Präsentation "Spiegel Mining" auf dem 33c3.

---

**Happy Crawling! 🎉**

````
