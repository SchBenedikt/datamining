"""
Crawling script for Heise News.

• Ensures that an "articles" table (with dynamic columns for alternative URLs)
  and a "crawl_state" table exist.
• Loads and saves the progress (year, month, and article index) in the database.
• Extracts article details using BeautifulSoup and inserts (or updates) them into the database.
• Outputs colorful status messages (including date and time) during the crawl.
• Sends an email via send_notification in case of errors.
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
from dotenv import load_dotenv  # added

load_dotenv()  # added

# PostgreSQL connection details
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def connect_db():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        print_status(f"DB connection error: {e}", "ERROR")
        raise

def print_status(message, level="INFO"):
    """Prints a formatted status message (with current date/time) to the console."""
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
    """Creates the 'articles' table if it does not exist."""
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
    Ensures that for each language in 'languages'
    a corresponding column exists in the 'articles' table.
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
    """Creates the 'crawl_state' table for storing the crawling progress."""
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
    """Reads the stored crawl progress; returns (year, month, article_index) or None."""
    with conn.cursor() as cur:
        cur.execute("SELECT year, month, article_index FROM crawl_state WHERE id = 1;")
        row = cur.fetchone()
    return row if row else None

def update_crawl_state(conn, year, month, article_index):
    """Updates or saves the current crawl status in the database."""
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
    Extracts details of an article from the given URL:
      - Author(s), category, keywords, word count,
      - Editor abbreviation, site name,
      - Alternative links (as a dictionary, keyed by language)
    Returns a tuple.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Determine editor abbreviation
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

    # Determine site name
    site_name = "N/A"
    site_name_meta = soup.find("meta", property="og:site_name")
    if site_name_meta and site_name_meta.get("content"):
        site_name = site_name_meta["content"]

    # Extract alternative URLs and sort by language
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

def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data):
    """
    Inserts an article into the 'articles' table or updates it,
    if an entry with the same URL already exists.
    Dynamically adds columns for alternative URLs.
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
    Starts the crawl process for Heise News.
    Loads the stored progress (year, month, article index) and resumes from there.
    Extracts all articles from an archive page, checks for duplicate URLs and 
    warns when fewer than 10 articles are found on a day.
    Then navigates to the previous archive page.
    """
    conn = connect_db()
    create_table(conn)
    create_crawl_state_table(conn)
    state = get_crawl_state(conn)
    if state:
        year, month, article_index = state
        print_status(f"Resuming crawl from: {year}/{month:02d}, article index {article_index}", "INFO")
    else:
        year, month, article_index = initial_year, initial_month, 0
        update_crawl_state(conn, year, month, article_index)
        print_status(f"Starting crawl from: {year}/{month:02d}", "INFO")
        # Send notification that crawling has started successfully
        send_notification("Crawling started", f"Crawling successfully started: {year}/{month:02d}", os.getenv('ALERT_EMAIL'))
    conn.close()

    while True:
        conn = connect_db()
        archive_url = f"https://www.heise.de/newsticker/archiv/{year}/{month:02d}"
        print_status(f"Crawling URL: {archive_url}", "INFO")
        try:
            response = requests.get(archive_url)
            response.raise_for_status()
        except Exception as e:
            print_status(f"HTTP error fetching {archive_url}: {e}", "ERROR")
            send_notification("Crawling Error", f"HTTP error: {e}", os.getenv('ALERT_EMAIL'))
            conn.close()
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')

        # Integrity Checks: Group articles by date
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

        # Check each day for article count
        for d, count in date_counts.items():
            if count < 10:
                warn_msg = f"Warning: Only {count} articles found on {d}!"
                print_status(warn_msg, "WARNING")
                if d != datetime.now().strftime("%Y-%m-%d"):
                    send_notification("Crawling Warning", warn_msg, os.getenv('ALERT_EMAIL', 'admin@example.com'))

        if not articles:
            print_status(f"No articles found for {year}-{month:02d}. Stopping crawl.", "WARNING")
            update_crawl_state(conn, year, month, 0)
            conn.close()
            break

        print_status(f"Found articles (total): {len(articles)}", "INFO")
        
        # Process articles day by day
        for day in sorted(articles_by_date.keys()):
            day_articles = articles_by_date[day]
            print_status(f"Processing {len(day_articles)} articles for date {day}", "INFO")
            for article in day_articles:
                try:
                    title = article.find('h3').get_text(strip=True)
                    a_link = article.find('a')
                    link = a_link['href'] if a_link else ""
                    if not link.startswith("http"):
                        link = "https://www.heise.de" + link
                    if "${" in link:
                        print_status(f"Skipping article with invalid link: {link}", "WARNING")
                        continue

                    time_element = article.find('time')
                    date = time_element['datetime'] if time_element else 'N/A'
                    author, category, keywords, word_count, editor_abbr, site_name, alt_data = get_article_details(link)
                    replaced = insert_article(conn, title, link, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data)
                    if replaced:
                        print_status(f"{date} - {title} already exists", "INFO")
                    else:
                        print_status(f"{date} - {title}", "INFO")
                except Exception as e:
                    error_msg = f"Error processing article {link}: {e}"
                    print_status(error_msg, "ERROR")
                    send_notification("Crawling Error", error_msg, os.getenv('ALERT_EMAIL', 'server@xn--schchner-2za.de'))
                    continue
            # After finishing the day's articles: update progress
            update_crawl_state(conn, year, month, 0)
            
        conn.close()

        # Navigation logic: Move to the previous archive page
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        try:
            formatted_text = figlet_format(f"{year}, {month:02d}")
            print(f"\033[1m{formatted_text}\033[0m")
        except Exception as e:
            print_status(f"Error displaying text: {e}", "ERROR")

if __name__ == '__main__':
    import threading
    # Start the API in a separate daemon thread
    threading.Thread(
        target=lambda: __import__('api').app.run(debug=True, port=6600, use_reloader=False),
        daemon=True
    ).start()
    try:
        crawl_heise()
    except KeyboardInterrupt:
        print_status("Crawling interrupted.", "WARNING")