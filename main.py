"""
Crawling-Skript für Heise-News.

• Stellt sicher, dass eine articles-Tabelle (mit dynamischen Spalten für alternative URLs)
  und eine crawl_state-Tabelle existieren.
• Lädt und speichert den Fortschritt (Jahr, Monat und Artikel-Index) in der Datenbank.
• Extrahiert Artikeldetails über BeautifulSoup und fügt sie (bzw. aktualisiert sie) in die Datenbank ein.
• Gibt während des Crawlings farbige Statusmeldungen (inklusive Datum und Uhrzeit) aus.
• Bei Fehlern wird eine E-Mail über send_notification gesendet.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import Json
from pyfiglet import figlet_format
from notification import send_notification
from datetime import datetime

# PostgreSQL connection details
db_params = {
    'dbname': os.getenv('DB_NAME', 'web_crawler'),
    'user': os.getenv('DB_USER', 'schaechner'),
    'password': os.getenv('DB_PASSWORD', 'SchaechnerServer'),
    'host': os.getenv('DB_HOST', '192.168.188.36'),
    'port': os.getenv('DB_PORT', '6543')
}

def connect_db():
    """Stellt eine Verbindung zur Datenbank her."""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        print_status(f"DB-Verbindungsfehler: {e}", "ERROR")
        raise

def print_status(message, level="INFO"):
    """Gibt eine formatierte Statusmeldung (mit aktuellem Datum/Uhrzeit) in der Konsole aus."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{now} {colors.get(level, colors['INFO'])}[{level}] {message}{colors['RESET']}")

# ----------------------------------------------------------------
# DATABASE SETUP FUNCTIONS
# ----------------------------------------------------------------
def create_table(conn):
    """Erstellt die 'articles'-Tabelle, falls nicht vorhanden."""
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS articles (
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
            )
        ''')
        conn.commit()

def ensure_language_columns(conn, languages):
    """
    Stellt sicher, dass für jede Sprache in 'languages'
    eine entsprechende Spalte in der 'articles'-Tabelle existiert.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'articles';")
        existing = {row[0] for row in cur.fetchall()}
    for lang in languages:
        if lang not in existing:
            with conn.cursor() as cur:
                cur.execute(f'ALTER TABLE articles ADD COLUMN "{lang}" TEXT;')
            conn.commit()

def create_crawl_state_table(conn):
    """Erstellt die 'crawl_state'-Tabelle zur Fortschrittsspeicherung."""
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS crawl_state (
                id INTEGER PRIMARY KEY,
                year INTEGER,
                month INTEGER,
                article_index INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

def get_crawl_state(conn):
    """Liest den gespeicherten Crawl-Fortschritt; gibt (year, month, article_index) oder None zurück."""
    with conn.cursor() as cur:
        cur.execute("SELECT year, month, article_index FROM crawl_state WHERE id = 1;")
        row = cur.fetchone()
    return row if row else None

def update_crawl_state(conn, year, month, article_index):
    """Aktualisiert oder speichert den aktuellen Crawl-Status in der Datenbank."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO crawl_state (id, year, month, article_index)
               VALUES (1, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET year = EXCLUDED.year, month = EXCLUDED.month, article_index = EXCLUDED.article_index;""",
            (year, month, article_index)
        )
    conn.commit()

# ----------------------------------------------------------------
# ARTICLE PROCESSING FUNCTIONS
# ----------------------------------------------------------------
def get_article_details(url):
    """
    Extrahiert Details eines Artikels von der übergebenen URL:
      - Autor(en), Kategorie, Schlüsselwörter, Wortanzahl,
      - Editor-Abkürzung, Seitenname,
      - Alternative Links (als Dictionary, nach Sprache)
    Gibt ein Tuple zurück.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Editor-Abkürzung ermitteln
    editor_abbr = "N/A"
    editor_span = soup.find("span", class_="redakteurskuerzel")
    if editor_span:
        abbr_link = editor_span.find("a", class_="redakteurskuerzel__link")
        if abbr_link:
            if abbr_link.string:
                editor_abbr = abbr_link.string
            elif 'title' in abbr_link.attrs:
                email = abbr_link.get('href', '')
                if email.startswith('mailto:'):
                    editor_abbr = email.split('@')[0].replace('mailto:', '')
                else:
                    editor_abbr = abbr_link.text

    # Seitenname ermitteln
    site_name = "N/A"
    site_name_meta = soup.find("meta", property="og:site_name")
    if site_name_meta and site_name_meta.get("content"):
        site_name = site_name_meta["content"]

    # Alternative URLs extrahieren und nach Sprache sortieren
    alternate_links = soup.find_all("link", rel="alternate")
    alt_data = {}
    for link_tag in alternate_links:
        hreflang = link_tag.get("hreflang")
        href = link_tag.get("href")
        if not href:
            continue
        if not href.startswith("http"):
            href = "https://www.heise.de" + href
        if hreflang:
            lang = hreflang.strip().lower()
            if lang == "x-default":
                continue
            alt_data[lang] = href

    # LD+JSON Daten extrahieren
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        return "N/A", "N/A", "N/A", None, editor_abbr, site_name, alt_data

    try:
        data = json.loads(script_tag.string)
        author = ", ".join([a["name"] for a in data.get("author", [])]) if "author" in data else "N/A"
        category = data.get("articleSection", "N/A")
        keywords = ", ".join([t["name"] for t in data.get("about", [])]) if "about" in data else "N/A"
        word_count = data.get("wordCount", None)
        return author, category, keywords, word_count, editor_abbr, site_name, alt_data
    except json.JSONDecodeError:
        return "N/A", "N/A", "N/A", None, editor_abbr, site_name, alt_data

def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data):
    """
    Fügt einen Artikel in die 'articles'-Tabelle ein oder aktualisiert diesen,
    falls bereits ein Eintrag mit der gleichen URL existiert.
    Dynamisch werden Spalten für alternative URLs hinzugefügt.
    """
    replaced = False
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM articles WHERE url=%s", (url,))
        if cur.fetchone():
            replaced = True
        if alt_data:
            ensure_language_columns(conn, alt_data.keys())
        base_columns = ["title", "url", "date", "author", "category", "keywords", "word_count", "editor_abbr", "site_name"]
        base_values = [title, url, date, author, category, keywords, word_count, editor_abbr, site_name]
        extra_columns = []
        extra_values = []
        if alt_data:
            for lang, link in alt_data.items():
                extra_columns.append(lang)
                extra_values.append(link)
        columns = base_columns + extra_columns
        values = base_values + extra_values
        placeholders = ", ".join(["%s"] * len(values))
        columns_sql = ", ".join('"' + col + '"' for col in columns)
        update_set = ", ".join(f'"{col}" = EXCLUDED."{col}"' for col in columns if col != "url")
        query = f'''
            INSERT INTO articles ({columns_sql})
            VALUES ({placeholders})
            ON CONFLICT (url) DO UPDATE SET {update_set}
        '''
        cur.execute(query, values)
    conn.commit()
    return replaced

# ----------------------------------------------------------------
# MAIN CRAWLING FUNCTION
# ----------------------------------------------------------------
def crawl_heise(initial_year=2025, initial_month=3):
    """
    Startet den Crawl-Prozess für Heise-News.
    Lädt den gespeicherten Fortschritt (Jahr, Monat, Artikel-Index) und setzt dort fort.
    Extrahiert alle Artikel einer Archivseite, überprüft auf doppelte URLs und 
    warnt, wenn an einem Tag weniger als 10 Artikel gefunden wurden.
    Navigiert danach zur vorherigen Archivseite.
    """
    conn = connect_db()
    create_table(conn)
    create_crawl_state_table(conn)
    state = get_crawl_state(conn)
    if state:
        year, month, article_index = state
        print_status(f"Wiederaufnahme Crawling ab: {year}/{month:02d}, Artikel-ID {article_index}", "INFO")
    else:
        year, month, article_index = initial_year, initial_month, 0
        update_crawl_state(conn, year, month, article_index)
        print_status(f"Startcrawl ab: {year}/{month:02d}", "INFO")
        # Sende Benachrichtigung, dass das Crawling erfolgreich gestartet wurde
        send_notification("Crawling gestartet", f"Crawling erfolgreich gestartet: {year}/{month:02d}", os.getenv('ALERT_EMAIL'))
    conn.close()

    while True:
        conn = connect_db()
        archive_url = f"https://www.heise.de/newsticker/archiv/{year}/{month:02d}"
        print_status(f"Crawle URL: {archive_url}", "INFO")
        try:
            response = requests.get(archive_url)
            response.raise_for_status()
        except Exception as e:
            print_status(f"HTTP-Fehler beim Abrufen von {archive_url}: {e}", "ERROR")
            send_notification("Crawling Fehler", f"HTTP-Fehler: {e}", os.getenv('ALERT_EMAIL'))
            conn.close()
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')

        # Integrity Checks: Gruppiere Artikel nach Datum
        articles_by_date = {}
        date_counts = {}
        for a in articles:
            time_elem = a.find('time')
            if time_elem and time_elem.has_attr('datetime'):
                d = time_elem['datetime'][:10]
            else:
                d = 'N/A'
            articles_by_date.setdefault(d, []).append(a)
            date_counts[d] = date_counts.get(d, 0) + 1

        # Überprüfe jeden Tag auf Anzahl der Artikel
        for d, count in date_counts.items():
            if count < 10:
                warn_msg = f"Warnung: Am {d} wurden nur {count} Artikel gefunden!"
                print_status(warn_msg, "WARNING")
                if d != datetime.now().strftime("%Y-%m-%d"):
                    send_notification("Crawling Warnung", warn_msg, os.getenv('ALERT_EMAIL', 'admin@example.com'))

        if not articles:
            print_status(f"Keine Artikel gefunden für {year}-{month:02d}. Beende Crawl.", "WARNING")
            update_crawl_state(conn, year, month, 0)
            conn.close()
            break

        print_status(f"Gefundene Artikel (insgesamt): {len(articles)}", "INFO")
        
        # Verarbeite Artikel tagweise
        for day in sorted(articles_by_date.keys()):
            day_articles = articles_by_date[day]
            print_status(f"Verarbeite {len(day_articles)} Artikel für den Tag {day}", "INFO")
            for article in day_articles:
                try:
                    title = article.find('h3').get_text(strip=True)
                    a_link = article.find('a')
                    link = a_link['href'] if a_link else ""
                    if not link.startswith("http"):
                        link = "https://www.heise.de" + link
                    if "${" in link:
                        print_status(f"Überspringe Artikel mit ungültigem Link: {link}", "WARNING")
                        continue

                    time_element = article.find('time')
                    date = time_element['datetime'] if time_element else 'N/A'
                    author, category, keywords, word_count, editor_abbr, site_name, alt_data = get_article_details(link)
                    replaced = insert_article(conn, title, link, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data)
                    if replaced:
                        print_status(f"{date} - {title} bereits vorhanden", "INFO")
                    else:
                        print_status(f"{date} - {title}", "INFO")
                except Exception as e:
                    error_msg = f"Fehler bei Artikel {link}: {e}"
                    print_status(error_msg, "ERROR")
                    send_notification("Crawling Fehler", error_msg, os.getenv('ALERT_EMAIL', 'server@xn--schchner-2za.de'))
                    continue
            # Nach Abschluss des Tages: Fortschritt aktualisieren
            update_crawl_state(conn, year, month, 0)
            
        conn.close()

        # Navigationslogik: Wechsle zur vorherigen Archivseite
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        try:
            formatted_text = figlet_format(f"{year}, {month:02d}")
            print(f"\033[1m{formatted_text}\033[0m")
        except Exception as e:
            print_status(f"Fehler bei der Anzeige: {e}", "ERROR")

if __name__ == '__main__':
    import threading
    # Startet die API in einem separaten Daemon-Thread
    threading.Thread(
        target=lambda: __import__('api').app.run(debug=True, port=6600, use_reloader=False),
        daemon=True
    ).start()
    try:
        crawl_heise()
    except KeyboardInterrupt:
        print_status("Crawling unterbrochen.", "WARNING")
