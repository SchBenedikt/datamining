import os
import time
import requests
import psycopg2
import json
from datetime import datetime
from bs4 import BeautifulSoup
from notification import send_notification
from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Status message function
def print_status(message, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{now} {colors.get(level, colors['INFO'])}[{level}] {message}{colors['RESET']}")

# Database connection function
def connect_db():
    try:
        db_params = {
            'dbname': os.getenv('DB_NAME', 'datamining'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        return psycopg2.connect(**db_params)
    except Exception as e:
        print_status(f"Database connection error: {e}", "ERROR")
        raise

# Create table function (like in main.py)
def create_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
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
            )
        ''')
        conn.commit()

# Neue Funktion: Sicherstellen, dass Sprachspalten existieren
def ensure_language_columns(conn, languages):
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'heise';")
        existing = {row[0] for row in cur.fetchall()}
    for lang in languages:
        if lang not in existing:
            with conn.cursor() as cur:
                cur.execute(f'ALTER TABLE heise ADD COLUMN "{lang}" TEXT;')
            conn.commit()

# Neue Funktion: Artikel einfügen/aktualisieren (wie in main.py)
def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data):
    replaced = False
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM heise WHERE url=%s", (url,))
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
            INSERT INTO heise ({columns_sql})
            VALUES ({placeholders})
            ON CONFLICT (url) DO UPDATE SET {update_set}
        '''
        cur.execute(query, values)
    conn.commit()
    return replaced

# Die bestehende get_article_details-Funktion (wie in main.py)
def get_article_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
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
    site_name = "N/A"
    site_name_meta = soup.find("meta", property="og:site_name")
    if site_name_meta and site_name_meta.get("content"):
        site_name = site_name_meta["content"]
    alt_data = {}
    for link_tag in soup.find_all("link", rel="alternate"):
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

def article_exists(conn, url):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM heise WHERE url=%s", (url,))
        return cur.fetchone() is not None

# Angepasste Funktion: Crawl-Prozess nur für heutige Artikel, jedoch mit Integritätschecks und E-Mail-Benachrichtigung wie in main.py
def crawl_current():
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now()
    archive_url = f"https://www.heise.de/newsticker/archiv/{now.year}/{now.month:02d}"
    print_status(f"Crawle neue Artikel vom {today} über: {archive_url}", "INFO")
    
    try:
        response = requests.get(archive_url)
        response.raise_for_status()
    except Exception as e:
        print_status(f"HTTP-Fehler beim Abrufen von {archive_url}: {e}", "ERROR")
        send_notification("Crawling Fehler", f"HTTP-Fehler: {e}", os.getenv('ALERT_EMAIL'))
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    
    # Gruppierung und Integrity-Check (wie in main.py)
    articles_by_date = {}
    date_counts = {}
    for a in articles:
        time_elem = a.find('time')
        d = time_elem['datetime'][:10] if (time_elem and time_elem.has_attr('datetime')) else 'N/A'
        articles_by_date.setdefault(d, []).append(a)
        date_counts[d] = date_counts.get(d, 0) + 1
    for d, count in date_counts.items():
        if count < 10:
            warn_msg = f"Warnung: Am {d} wurden nur {count} Artikel gefunden!"
            print_status(warn_msg, "WARNING")
            if d != today:
                send_notification("Crawling Warnung", warn_msg, os.getenv('ALERT_EMAIL', 'admin@example.com'))
    
    if today not in articles_by_date:
        print_status(f"Keine Artikel gefunden für {today}.", "WARNING")
        return

    day_articles = articles_by_date[today]
    print_status(f"Verarbeite {len(day_articles)} Artikel für den Tag {today}", "INFO")
    
    # Zähler für neue Artikel einführen
    new_articles = 0
    
    # Öffne eine DB-Verbindung ein Mal für den heutigen Tag
    conn = connect_db()
    create_table(conn)
    
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
            # Prüfe in der DB, ob Artikel bereits vorhanden ist
            if article_exists(conn, link):
                print_status(f"{link} bereits vorhanden", "INFO")
                continue
            
            time_element = article.find('time')
            date_val = time_element['datetime'] if time_element else 'N/A'
            author, category, keywords, word_count, editor_abbr, site_name, alt_data = get_article_details(link)
            
            replaced = insert_article(conn, title, link, date_val, author, category, keywords, word_count, editor_abbr, site_name, alt_data)
            if not replaced:
                new_articles += 1
                print_status(f"{date_val} - {title} wurde neu eingefügt", "INFO")
            else:
                print_status(f"{date_val} - {title} bereits vorhanden", "INFO")
        except Exception as e:
            error_msg = f"Fehler bei Artikel {link}: {e}"
            print_status(error_msg, "ERROR")
            send_notification("Crawling Fehler", error_msg, os.getenv('ALERT_EMAIL'))
            continue

    if new_articles == 0:
        print_status("Keine neuen Artikel gefunden.", "INFO")
    
    print_status("Aktueller Crawl-Durchlauf abgeschlossen.", "INFO")
    conn.close()

def main():
    print("Press CTRL+C to quit")  # entsprechende Anweisung aus main.py
    while True:
        crawl_current()
        print_status("Warte 300 Sekunden bis zum nächsten Crawl. (Press CTRL+C to quit)", "INFO")
        time.sleep(300)

if __name__ == '__main__':
    import threading
    # Starte den API-Endpoint in einem Daemon-Thread
    threading.Thread(
        target=lambda: __import__('api').app.run(debug=True, port=6600, use_reloader=False),
        daemon=True
    ).start()
    main()
