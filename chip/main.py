import os
import json
import requests
import psycopg2
from psycopg2.extras import Json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pyfiglet import figlet_format
from notification import send_notification
from datetime import datetime

load_dotenv()

# PostgreSQL connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def connect_db():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        print_status(f"DB connection error: {e}", "ERROR")
        raise

def print_status(message, level="INFO"):
    """Outputs a colored status message with current date and time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{now} {colors.get(level, colors['INFO'])}[{level}] {message}{colors['RESET']}")

# -----------------------------------------------------------------------------
# TABLE AND CRAWL STATE FUNCTIONS
# -----------------------------------------------------------------------------
def create_table():
    """Creates the articles table if it doesn't exist yet."""
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
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
            """)
            
            # Check if title column exists, add it if not (for existing databases)
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'articles' AND column_name = 'title';
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE articles ADD COLUMN title TEXT;")
                print_status("Added 'title' column to existing articles table.", "INFO")
            
            conn.commit()
        conn.close()
        print_status("Table 'articles' was created or already exists.", "INFO")
    except Exception as e:
        print_status(f"Error creating table: {e}", "ERROR")
        send_notification("Chip Mining Error", f"Error creating table: {e}", os.getenv('ALERT_EMAIL'))

def create_crawl_state_table_chip(conn):
    """Creates the crawl_state table for storing crawl progress."""
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS crawl_state (
                id INTEGER PRIMARY KEY,
                page INTEGER,
                article_index INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

def get_crawl_state(conn):
    """Reads the saved crawl progress; returns (page, article_index) or None."""
    with conn.cursor() as cur:
        cur.execute("SELECT page, article_index FROM crawl_state WHERE id = 1;")
        row = cur.fetchone()
    return row if row else None

def update_crawl_state(conn, page, article_index):
    """Updates or saves the current crawl state in the database."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO crawl_state (id, page, article_index)
               VALUES (1, %s, %s)
               ON CONFLICT (id) DO UPDATE SET page = EXCLUDED.page, article_index = EXCLUDED.article_index;""",
            (page, article_index)
        )
    conn.commit()

# -----------------------------------------------------------------------------
# ARTICLE PROCESSING
# -----------------------------------------------------------------------------
def insert_chip_article(conn, url, title, author, date, keywords, description, type_, 
                        page_level1, page_level2, page_level3, page_template):
    """
    Inserts an article into the articles table or updates it
    if an entry with the same URL already exists.
    """
    replaced = False
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM articles WHERE url=%s", (url,))
        if cur.fetchone():
            replaced = True
        base_columns = ["url", "title", "author", "date", "keywords", "description", "type",
                        "page_level1", "page_level2", "page_level3", "page_template"]
        base_values = [url, title, author, date, keywords, description, type_,
                       page_level1, page_level2, page_level3, page_template]
        placeholders = ", ".join(["%s"] * len(base_values))
        columns_sql = ", ".join('"' + col + '"' for col in base_columns)
        update_set = ", ".join(f'"{col}" = EXCLUDED."{col}"' for col in base_columns if col != "url")
        query = f'''
            INSERT INTO articles ({columns_sql})
            VALUES ({placeholders})
            ON CONFLICT (url) DO UPDATE SET {update_set}
        '''
        cur.execute(query, base_values)
    conn.commit()
    return replaced

def scrape_article_details(conn, url):
    """
    Scrapes the details of a single article from CHIP and stores it in the database.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract standard metadata
            title_meta = soup.find("meta", attrs={"property": "og:title"})
            author_meta = soup.find("meta", attrs={"name": "author"})
            date_elem = soup.find("time")
            description_meta = soup.find("meta", attrs={"name": "description"})
            type_meta = soup.find("meta", attrs={"property": "og:type"})
            
            title = title_meta["content"].strip() if title_meta and title_meta.get("content") else "Unknown"
            author = author_meta["content"].strip() if author_meta and author_meta.get("content") else "Unknown"
            date = date_elem["content"].strip() if date_elem and date_elem.get("content") else "Unknown"
            description = description_meta["content"].strip() if description_meta and description_meta.get("content") else "Unknown"
            type_ = type_meta["content"].strip() if type_meta and type_meta.get("content") else "Unknown"
            
            # Extract utag_data from a script tag
            script_tag = soup.find("script", string=lambda t: t and "utag_data" in t)
            page_level1 = page_level2 = page_level3 = page_template = "Unknown"
            keywords = "Unknown"
            if script_tag:
                try:
                    json_text = script_tag.string.split("var utag_data =", 1)[1].split(";", 1)[0].strip()
                    utag_data = json.loads(json_text)
                    page_level1 = utag_data.get("pageLevel1", "Unknown")
                    page_level2 = utag_data.get("pageLevel2", "Unknown")
                    page_level3 = utag_data.get("pageLevel3", "Unknown")
                    page_template = utag_data.get("pageTemplate", "Unknown")
                    keywords = ", ".join(utag_data.get("pageAdKeyword", []))
                except Exception as e:
                    print_status(f"Error parsing utag_data in {url}: {e}", "ERROR")
                    send_notification("Chip Mining Error", f"Error parsing utag_data in {url}: {e}", os.getenv('ALERT_EMAIL'))
            
            replaced = insert_chip_article(conn, url, title, author, date, keywords, description, 
                                           type_, page_level1, page_level2, page_level3, page_template)
            if replaced:
                print_status(f"{date} - {url} already exists.", "INFO")
            else:
                print_status(f"{date} - {url} has been inserted.", "INFO")
        else:
            print_status(f"Error retrieving article {url}: {response.status_code}", "ERROR")
            send_notification("Chip Mining Error", f"Error retrieving article {url}: {response.status_code}", os.getenv('ALERT_EMAIL'))
    except Exception as e:
        print_status(f"Error scraping article {url}: {e}", "ERROR")
        send_notification("Chip Mining Error", f"Error scraping article {url}: {e}", os.getenv('ALERT_EMAIL'))

# -----------------------------------------------------------------------------
# MAIN FUNCTION: CHIP NEWS SCRAPING
# -----------------------------------------------------------------------------
def scrape_chip_news():
    """
    Retrieves article URLs from CHIP and processes them.
    Stores crawl progress in the database,
    outputs status messages (with banner) and reports errors via email.
    """
    create_table()  # Ensure that the articles table exists.
    conn = connect_db()
    create_crawl_state_table_chip(conn)
    
    state = get_crawl_state(conn)
    if state:
        page, article_index = state
        print_status(f"Resuming Chip Mining from page {page}, article index {article_index}", "INFO")
    else:
        page = 1
        article_index = 0
        update_crawl_state(conn, page, article_index)
        print_status(f"Starting Chip Mining from page {page}", "INFO")
        send_notification("Chip Mining Started", f"Chip Mining started on page {page}", os.getenv('ALERT_EMAIL'))
    
    while True:
        url = f"https://www.chip.de/nachrichten/{page}"
        print_status(f"Crawling URL: {url}", "INFO")
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception as e:
            print_status(f"HTTP error retrieving {url}: {e}", "ERROR")
            send_notification("Chip Mining Error", f"HTTP error retrieving {url}: {e}", os.getenv('ALERT_EMAIL'))
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        # Filter all links containing "/news/"
        links = [link['href'] for link in soup.find_all('a', href=True) if '/news/' in link['href']]
        if not links:
            print_status(f"No articles found on page {page}. Crawl ending.", "WARNING")
            update_crawl_state(conn, page, 0)
            break
        print_status(f"Found {len(links)} article links on page {page}.", "INFO")
        
        # Process each article link
        for i, link in enumerate(links, start=article_index):
            full_url = requests.compat.urljoin("https://www.chip.de", link)
            if "${" in full_url:
                print_status(f"Skipping article with invalid link: {full_url}", "WARNING")
                continue
            scrape_article_details(conn, full_url)
            article_index = i + 1  # Update article index after processing each article
            update_crawl_state(conn, page, article_index)  # Update after each article

        # Show a banner for the current page
        try:
            formatted_text = figlet_format(f"Page {page}")
            print(f"\033[1m{formatted_text}\033[0m")
        except Exception as e:
            print_status(f"Error displaying banner: {e}", "ERROR")
        
        # If there are more articles to crawl, increment the page number
        page += 1
        update_crawl_state(conn, page, 0)  # Reset article index for the next page
    
    conn.close()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        scrape_chip_news()
    except KeyboardInterrupt:
        print_status("Chip Mining was interrupted.", "WARNING")
