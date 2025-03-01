import requests
from bs4 import BeautifulSoup
import psycopg2
import json
from pyfiglet import figlet_format
from notification import send_notification
import os
from datetime import datetime
from psycopg2.extras import Json

# PostgreSQL connection details
db_params = {
    'dbname': os.getenv('DB_NAME', 'web_crawler'),
    'user': os.getenv('DB_USER', 'schaechner'),
    'password': os.getenv('DB_PASSWORD', 'SchaechnerServer'),
    'host': os.getenv('DB_HOST', '192.168.188.36'),
    'port': os.getenv('DB_PORT', '6543')
}

def connect_db():
    return psycopg2.connect(**db_params)

# Modify create_table to remove hard-coded language columns:
def create_table(conn):
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

# NEW: Add helper function to add language columns dynamically
def ensure_language_columns(conn, languages):
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'articles';")
        existing = {row[0] for row in cur.fetchall()}
    for lang in languages:
        if lang not in existing:
            with conn.cursor() as cur:
                cur.execute(f'ALTER TABLE articles ADD COLUMN "{lang}" TEXT;')
            conn.commit()

# NEW: Create crawl_state table with article_index
def create_crawl_state_table(conn):
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

# NEW: Get saved crawl state. Returns (year, month, article_index) or None.
def get_crawl_state(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT year, month, article_index FROM crawl_state WHERE id = 1;")
        row = cur.fetchone()
    return row if row else None

# NEW: Update crawl state.
def update_crawl_state(conn, year, month, article_index):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO crawl_state (id, year, month, article_index)
               VALUES (1, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET year = EXCLUDED.year, month = EXCLUDED.month, article_index = EXCLUDED.article_index;""",
            (year, month, article_index)
        )
    conn.commit()

# Modify get_article_details to build alt_data as dict keyed by language code:
def get_article_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract editor abbreviation
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

    # Extract site name
    site_name = "N/A"
    site_name_meta = soup.find("meta", property="og:site_name")
    if site_name_meta and site_name_meta.get("content"):
        site_name = site_name_meta["content"]

    # Extract alternate URLs and organize them by language
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
            if lang == "x-default":  # Skip x-default
                continue
            alt_data[lang] = href

    # Extract LD+JSON data
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

# Modify insert_article to dynamically include language columns:
def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data):
    replaced = False
    with conn.cursor() as cur:
        # Check if the URL exists
        cur.execute("SELECT id FROM articles WHERE url=%s", (url,))
        if cur.fetchone():
            replaced = True
        # Ensure all language columns exist
        if alt_data:
            ensure_language_columns(conn, alt_data.keys())
        # Base columns and values
        base_columns = ["title", "url", "date", "author", "category", "keywords", "word_count", "editor_abbr", "site_name"]
        base_values = [title, url, date, author, category, keywords, word_count, editor_abbr, site_name]
        # Append dynamic language columns and values
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

# Helper function for formatted printing
def print_status(message, level="INFO"):
    colors = {
        "INFO": "\033[92m",    
        "WARNING": "\033[93m", 
        "ERROR": "\033[91m",   
        "RESET": "\033[0m"
    }
    print(f"{colors.get(level, colors['INFO'])}[{level}] {message}{colors['RESET']}")

def crawl_heise(initial_year=None, initial_month=None):
    """
    Startet den Crawl-Prozess:
    - Wird ein gespeicherter Fortschritt gefunden, so wird genau dort fortgesetzt.
    - Fehlt der gespeicherte Fortschritt, werden als Startpunkt das aktuelle Datum verwendet.
    - Wird die aktuelle (heutige) Archivseite verarbeitet, so wird nach dem Durchlauf gewartet und 
      die Seite erneut abgefragt, um neu hinzugekommene Artikel zu erfassen.
    - Bereits in der DB vorhandene Artikel (unique URL) werden übersprungen.
    """
    conn = connect_db()
    create_table(conn)
    create_crawl_state_table(conn)
    
    # State: Wenn vorhanden, verwende ihn, sonst aktuelles Datum
    state = get_crawl_state(conn)
    if state:
        year, month, article_index = state
        print_status(f"Resuming crawl from saved state: {year}/{month:02d}, article index {article_index}", "INFO")
    else:
        now = datetime.now()
        year, month, article_index = now.year, now.month, 0
        update_crawl_state(conn, year, month, article_index)
        print_status(f"Starting crawl from current date: {year}/{month:02d}", "INFO")
    conn.close()

    import time
    while True:
        conn = connect_db()
        # Zum Start jeder Archivseite den aktuellen Fortschritt der Seite neu auslesen.
        state = get_crawl_state(conn)
        if state and state[0] == year and state[1] == month:
            article_index = state[2]
        archive_url = f"https://www.heise.de/newsticker/archiv/{year}/{month:02d}"
        print_status(f"Crawling archive page: {archive_url} (ab Artikel {article_index})", "INFO")
        response = requests.get(archive_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')

        if not articles:
            print_status(f"Keine Artikel gefunden für {year}-{month:02d}. Beende Crawl.", "WARNING")
            update_crawl_state(conn, year, month, 0)
            conn.close()
            break

        print_status(f"Gefundene Artikel: {len(articles)}", "INFO")

        # Verarbeitung ab dem gespeicherten article_index
        for i in range(article_index, len(articles)):
            try:
                article = articles[i]
                title = article.find('h3').get_text(strip=True)
                link = article.find('a')['href']
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
                    print_status(f"{title} bereits vorhanden", "INFO")
                else:
                    print_status(title, "INFO")
            except Exception as e:
                error_msg = f"Fehler bei Artikel {link}: {e}"
                print_status(error_msg, "ERROR")
                send_notification("Crawling Fehler", error_msg, os.getenv('ALERT_EMAIL', 'admin@example.com'))
                continue
            # Aktualisiere Fortschritt nach jedem Artikel
            update_crawl_state(conn, year, month, i + 1)
        
        # Seite vollständig verarbeitet: Setze article_index zurück
        update_crawl_state(conn, year, month, 0)
        conn.close()
        
        now = datetime.now()
        # Wenn die verarbeitete Archivseite gleich dem aktuellen Datum ist, warte auf neue Artikel.
        if year == now.year and month == now.month:
            print_status("Überprüfe aktuelle Archivseite auf neue Artikel (warte 5 Minuten)", "INFO")
            time.sleep(300)  # 5 Minuten warten
            continue
        # Andernfalls wechsle zur älteren Archivseite
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1
        
        try:
            formatted_text = figlet_format(f"{year}, {month:02d}")
            print(f"\033[1m{formatted_text}\033[0m")
        except Exception as e:
            print_status(f"Fehler bei Ausgabe: {e}", "ERROR")

if __name__ == '__main__':
    import threading
    # Start the API in a separate daemon thread
    threading.Thread(
        target=lambda: __import__('api').app.run(debug=True, use_reloader=False),
        daemon=True
    ).start()
    crawl_heise()
