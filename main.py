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

def crawl_heise(initial_year=2025, initial_month=3):
    conn = connect_db()
    create_table(conn)
    create_crawl_state_table(conn)
    
    # Get current year and month
    current = datetime.now()
    current_year = current.year
    current_month = current.month

    state = get_crawl_state(conn)
    if state:
        saved_year, saved_month, article_index = state
        # If current date is newer than saved state, update state to current date
        if (current_year > saved_year) or (current_year == saved_year and current_month > saved_month):
            saved_year, saved_month, article_index = current_year, current_month, 0
            update_crawl_state(conn, saved_year, saved_month, 0)
        print_status(f"Wiederaufnahme Crawling ab: {saved_year}/{saved_month:02d}, Artikel-ID {article_index}", "INFO")
        year, month = saved_year, saved_month
    else:
        # No saved state, start with current date
        year, month, article_index = current_year, current_month, 0
    conn.close()

    while True:
        conn = connect_db()
        print_status(f"Crawle URL: https://www.heise.de/newsticker/archiv/{year}/{month:02d}", "INFO")
        response = requests.get(f"https://www.heise.de/newsticker/archiv/{year}/{month:02d}")
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')

        # NEW: Integrity Checks
        titles = []
        date_set = set()
        for a in articles:
            h3 = a.find('h3')
            if h3:
                titles.append(h3.get_text(strip=True))
            time_elem = a.find('time')
            if time_elem and time_elem.has_attr('datetime'):
                date_set.add(time_elem['datetime'][:10])  # only the date part
        if len(titles) != len(set(titles)):
            print_status("Warnung: Duplikate in Artikeltiteln gefunden!", "WARNING")
        if not date_set:
            print_status("Warnung: Keine Veröffentlichungsdaten in den Artikeln gefunden!", "WARNING")

        if not articles:
            print_status(f"Keine Artikel gefunden für {year}-{month:02d}. Beende Crawl.", "WARNING")
            update_crawl_state(conn, year, month, 0)
            conn.close()
            break

        print_status(f"Gefundene Artikel: {len(articles)}", "INFO")

        # Start processing from saved article_index
        for i in range(article_index, len(articles)):
            try:
                article = articles[i]
                title = article.find('h3').get_text(strip=True)
                link = article.find('a')['href']
                if not link.startswith("http"):
                    link = "https://www.heise.de" + link
                if "${" in link:
                    # Display the URL of the skipped article
                    print_status(f"Überspringe Artikel mit ungültigem Link: {link}", "WARNING")
                    continue

                time_element = article.find('time')
                date = time_element['datetime'] if time_element else 'N/A'

                author, category, keywords, word_count, editor_abbr, site_name, alt_data = get_article_details(link)
                replaced = insert_article(conn, title, link, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data)
                # Instead of printing an ERROR when the article is already in the database,
                # show an informational message so that duplicates are not flagged as an error.
                if replaced:
                    print_status(f"{title} bereits vorhanden", "INFO")
                else:
                    print_status(title, "INFO")
            except Exception as e:
                error_msg = f"Fehler bei Artikel {link}: {e}"
                print_status(error_msg, "ERROR")
                # Send an email alert (recipient from env variable ALERT_EMAIL)
                send_notification("Crawling Fehler", error_msg, os.getenv('ALERT_EMAIL', 'admin@example.com'))
                continue
            # Update state after each article
            update_crawl_state(conn, year, month, i + 1)

        # Finished current archive page; reset article_index to 0 for next iteration.
        update_crawl_state(conn, year, month, 0)
        conn.close()

        # Navigation: move to previous month
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
