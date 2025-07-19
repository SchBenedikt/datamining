# Heise Mining Streamlit Dashboard

Eine moderne, interaktive Streamlit-basierte Webanwendung für das Heise Mining Projekt. Diese App ersetzt die bisherige Flask/Dash-Lösung und bietet eine verbesserte Benutzeroberfläche mit besserer Performance.

## 🚀 Funktionen

### 📊 Dashboard
- **Übersicht über alle Artikel** mit Metriken und Statistiken
- **Zeitbasierte Visualisierungen** (Artikel pro Tag, Stunden, Wochentage)
- **Kategorien- und Autorenverteilung** mit interaktiven Charts
- **Neueste Artikel** mit direkten Links

### 🔍 Artikelsuche
- **Volltext-Suche** in Titeln und Schlagwörtern
- **Erweiterte Filter** (Kategorie, Autor, Datum)
- **Flexible Sortierung** (Datum, Titel)
- **Paginierung** für große Ergebnismengen
- **Artikel-Vorschau** direkt in der App

### 🕸️ Autoren-Netzwerk
- **Interaktive Netzwerk-Visualisierung** der Autoren-Kollaborationen
- **Netzwerk-Statistiken** (Anzahl Knoten, Kanten, Verbindungen)
- **Top-Autoren-Ranking** nach Verbindungen

### 📈 Erweiterte Analysen
- **Zeitliche Trends** (Artikel nach Stunden, Wochentagen)
- **Kategorien-Analyse** mit Trends über Zeit
- **Autoren-Aktivität** und Produktivitätsanalyse
- **Wort-Statistiken** mit Verteilungsanalyse

### 🔧 SQL-Abfragen
- **Sichere SQL-Abfragen** (nur SELECT erlaubt)
- **Beispiel-Abfragen** zum schnellen Einstieg
- **Ergebnis-Download** als CSV
- **Datenbank-Export** als SQLite-Datei

## 🛠️ Installation und Start

### Voraussetzungen
- Python 3.8+
- PostgreSQL-Datenbank (bereits konfiguriert)
- `.env`-Datei mit Datenbankverbindung

### Schnellstart

1. **Abhängigkeiten installieren und App starten:**
   ```bash
   python run_streamlit.py
   ```

2. **Oder manuell:**
   ```bash
   pip install -r requirements_streamlit.txt
   streamlit run streamlit_app.py
   ```

3. **Über main.py starten:**
   ```bash
   python main.py --streamlit
   ```

### Standardmäßige URLs
- **Streamlit-App:** http://localhost:8501
- **Original Flask-API:** http://localhost:6600 (wenn `main.py` ohne `--streamlit` gestartet wird)

## 🎯 Verbesserungen gegenüber der alten Lösung

### Performance
- **Caching** für Datenbankabfragen (5-15 Minuten TTL)
- **Lazy Loading** für große Datenmengen
- **Optimierte Queries** mit Pandas-Integration

### Benutzerfreundlichkeit
- **Responsive Design** für alle Bildschirmgrößen
- **Intuitive Navigation** mit Sidebar-Menü
- **Echtzeit-Feedback** mit Spinner und Fortschrittsanzeigen
- **Verbesserte Visualisierungen** mit Plotly

### Funktionalität
- **Erweiterte Suchfunktionen** mit Volltext-Suche
- **Artikel-Vorschau** ohne externe Links
- **Bessere Datenexporte** (CSV, SQLite)
- **Modulare Architektur** für einfache Erweiterungen

### Sicherheit
- **Input-Validierung** für alle Benutzereingaben
- **SQL-Injection-Schutz** durch Parameterisierung
- **Fehlerbehandlung** mit benutzerfreundlichen Nachrichten

## 📁 Dateistruktur

```
heise/
├── streamlit_app.py          # Hauptanwendung
├── run_streamlit.py          # Starter-Skript
├── requirements_streamlit.txt # Streamlit-Abhängigkeiten
├── main.py                   # Erweitert für Streamlit-Support
├── api.py                    # Original Flask/Dash-API
└── templates/                # Original HTML-Templates
```

## 🔧 Konfiguration

### Umgebungsvariablen (.env)
```env
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

### Streamlit-Konfiguration
Die App verwendet folgende Streamlit-Einstellungen:
- **Port:** 8501
- **Layout:** Wide
- **Sidebar:** Expandiert
- **Caching:** Aktiviert

## 🚀 Deployment

### Lokale Entwicklung
```bash
streamlit run streamlit_app.py --server.port=8501
```

### Produktionsumgebung
```bash
streamlit run streamlit_app.py --server.headless=true --server.port=8501
```

## 📊 Caching-Strategien

- **Datenbankverbindung:** 5 Minuten TTL
- **Artikel-Daten:** 10 Minuten TTL
- **Autoren-Netzwerk:** 15 Minuten TTL
- **Cache-Invalidierung:** Automatisch bei neuen Daten

## 🔄 Migration von Flask zu Streamlit

Die neue Streamlit-App ist vollständig kompatibel mit der bestehenden Datenbank und bietet alle Funktionen der alten Flask/Dash-Lösung:

- ✅ Alle API-Endpunkte als interaktive Seiten
- ✅ Verbesserte Visualisierungen
- ✅ Bessere Performance durch Caching
- ✅ Modernere Benutzeroberfläche
- ✅ Mobile-responsive Design

## 🤝 Beitragen

Zur Verbesserung der App:
1. Neue Funktionen in `streamlit_app.py` hinzufügen
2. Abhängigkeiten in `requirements_streamlit.txt` aktualisieren
3. Tests für neue Features schreiben
4. Performance-Optimierungen implementieren

## 📝 Lizenz

Dieses Projekt steht unter der gleichen Lizenz wie das Hauptprojekt.
