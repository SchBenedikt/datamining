import requests
from bs4 import BeautifulSoup
import psycopg2
import json
from pyfiglet import figlet_format
from notification import send_notification
import os
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
    return psycopg2.connect(**db_params)

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
                site_name TEXT,
                alternate_urls TEXT
            )
        ''')
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

def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alternate_urls):
    replaced = False
    with conn.cursor() as cur:
        # Check if the URL already exists
        cur.execute("SELECT id FROM articles WHERE url=%s", (url,))
        if cur.fetchone():
            replaced = True
        cur.execute('''
            INSERT INTO articles (title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alternate_urls)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                date = EXCLUDED.date,
                author = EXCLUDED.author,
                category = EXCLUDED.category,
                keywords = EXCLUDED.keywords,
                word_count = EXCLUDED.word_count,
                editor_abbr = EXCLUDED.editor_abbr,
                site_name = EXCLUDED.site_name,
                alternate_urls = EXCLUDED.alternate_urls
        ''', (title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alternate_urls))
    conn.commit()
    return replaced

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

    # Extract alternate URLs with hreflang values from <link rel="alternate">
    alternate_links = soup.find_all("link", rel="alternate")
    alternates = []
    for link_tag in alternate_links:
        hreflang = link_tag.get("hreflang", "x-default")
        href = link_tag.get("href")
        if href and not href.startswith("http"):
            href = "https://www.heise.de" + href
        alternates.append({"hreflang": hreflang, "href": href})
    alternate_urls = json.dumps(alternates)

    # Extract LD+JSON data
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        return "N/A", "N/A", "N/A", None, editor_abbr, site_name, alternate_urls

    try:
        data = json.loads(script_tag.string)
        author = ", ".join([a["name"] for a in data.get("author", [])]) if "author" in data else "N/A"
        category = data.get("articleSection", "N/A")
        keywords = ", ".join([t["name"] for t in data.get("about", [])]) if "about" in data else "N/A"
        word_count = data.get("wordCount", None)
        return author, category, keywords, word_count, editor_abbr, site_name, alternate_urls
    except json.JSONDecodeError:
        return "N/A", "N/A", "N/A", None, editor_abbr, site_name, alternate_urls

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
    # Initialize tables and load saved state:
    conn = connect_db()
    create_table(conn)
    create_crawl_state_table(conn)
    state = get_crawl_state(conn)
    if state:
        year, month, article_index = state
        print_status(f"Wiederaufnahme Crawling ab: {year}/{month:02d}, Artikel-ID {article_index}", "INFO")
    else:
        year, month, article_index = initial_year, initial_month, 0
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
                    print_status(f"Überspringe Artikel mit ungültigem Link: {link}", "WARNING")
                    continue

                time_element = article.find('time')
                date = time_element['datetime'] if time_element else 'N/A'

                author, category, keywords, word_count, editor_abbr, site_name, alternate_urls = get_article_details(link)
                replaced = insert_article(conn, title, link, date, author, category, keywords, word_count, editor_abbr, site_name, alternate_urls)
                # Instead of printing an ERROR when the article is already in the database,
                # show an informational message so that duplicates are not flagged as an error.
                if replaced:
                    print_status(f"{title} bereits vorhanden", "INFO")
                else:
                    print_status(title, "INFO")
            except Exception as e:
                print_status(f"Fehler bei Artikel {link}: {e}", "ERROR")
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
