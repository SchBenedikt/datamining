import os
import time
import requests
import psycopg2
import json
from datetime import datetime
from bs4 import BeautifulSoup
from notification import send_notification
from dotenv import load_dotenv

load_dotenv()

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
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        return psycopg2.connect(**db_params)
    except Exception as e:
        print_status(f"DB connection error: {e}", "ERROR")
        raise

# Create table function
def create_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
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
            )
        ''')
        conn.commit()

# Check if article exists
def article_exists(conn, url):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM chip WHERE url=%s", (url,))
        return cur.fetchone() is not None

# Insert article function
def insert_chip_article(conn, url, title, author, date, keywords, description, type_, 
                        page_level1, page_level2, page_level3, page_template):
    replaced = False
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM chip WHERE url=%s", (url,))
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
            INSERT INTO chip ({columns_sql})
            VALUES ({placeholders})
            ON CONFLICT (url) DO UPDATE SET {update_set}
        '''
        cur.execute(query, base_values)
    conn.commit()
    return replaced

# Get article details function
def get_article_details(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract standard metadata
            title_meta = soup.find("meta", attrs={"property": "og:title"})
            author_meta = soup.find("meta", attrs={"name": "author"})
            date_elem = soup.find("time")
            description_meta = soup.find("meta", attrs={"name": "description"})
            
            title = title_meta['content'] if title_meta else "N/A"
            author = author_meta['content'] if author_meta else "N/A"
            date = date_elem['datetime'] if date_elem and date_elem.has_attr('datetime') else "N/A"
            description = description_meta['content'] if description_meta else "N/A"
            
            # Extract keywords
            keywords_meta = soup.find("meta", attrs={"name": "keywords"})
            keywords = keywords_meta['content'] if keywords_meta else "N/A"
            
            # Extract JSON-LD structured data
            type_ = "N/A"
            page_level1 = "N/A"
            page_level2 = "N/A"
            page_level3 = "N/A"
            page_template = "N/A"
            
            json_ld = soup.find("script", type="application/ld+json")
            if json_ld:
                try:
                    data = json.loads(json_ld.string)
                    type_ = data.get("@type", "N/A")
                    breadcrumb = data.get("breadcrumb", {})
                    if isinstance(breadcrumb, dict):
                        items = breadcrumb.get("itemListElement", [])
                        if len(items) > 0:
                            page_level1 = items[0].get("name", "N/A")
                        if len(items) > 1:
                            page_level2 = items[1].get("name", "N/A")
                        if len(items) > 2:
                            page_level3 = items[2].get("name", "N/A")
                    page_template = data.get("pageTemplate", "N/A")
                except:
                    pass
            
            return title, author, date, keywords, description, type_, page_level1, page_level2, page_level3, page_template
        else:
            return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
    except Exception as e:
        print_status(f"Error getting article details from {url}: {e}", "ERROR")
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

# Crawl current articles from Chip
def crawl_current():
    print_status("Starting crawl for current Chip news articles", "INFO")
    
    # Open database connection
    conn = connect_db()
    create_table(conn)
    
    # Get articles from page 1 (newest articles)
    url = "https://www.chip.de/nachrichten/1"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print_status(f"HTTP error fetching {url}: {e}", "ERROR")
        send_notification("Chip Crawling Error", f"HTTP error: {e}", os.getenv('ALERT_EMAIL'))
        conn.close()
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    # Filter all links containing "/news/"
    links = [link['href'] for link in soup.find_all('a', href=True) if '/news/' in link['href']]
    
    # Make links absolute and deduplicate
    absolute_links = []
    seen = set()
    for link in links:
        if not link.startswith("http"):
            link = "https://www.chip.de" + link
        if link not in seen:
            absolute_links.append(link)
            seen.add(link)
    
    if not absolute_links:
        print_status("No articles found on page 1", "WARNING")
        conn.close()
        return
    
    new_articles = 0
    for article_url in absolute_links:
        # Check if article already exists
        if article_exists(conn, article_url):
            print_status(f"{article_url} already exists", "INFO")
            continue
        
        # Get article details
        title, author, date, keywords, description, type_, page_level1, page_level2, page_level3, page_template = get_article_details(article_url)
        
        # Insert article
        try:
            insert_chip_article(conn, article_url, title, author, date, keywords, description, type_,
                              page_level1, page_level2, page_level3, page_template)
            print_status(f"Inserted new article: {title} - {article_url}", "INFO")
            new_articles += 1
        except Exception as e:
            print_status(f"Error inserting article {article_url}: {e}", "ERROR")
            send_notification("Chip Crawling Error", f"Error inserting article {article_url}: {e}", os.getenv('ALERT_EMAIL'))
    
    conn.close()
    print_status(f"Crawl complete. Found {new_articles} new articles.", "INFO")

def main():
    print("Press CTRL+C to quit")
    while True:
        crawl_current()
        print_status("Waiting 600 seconds until next crawl. (Press CTRL+C to quit)", "INFO")
        time.sleep(600)  # 10 minutes

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("Chip current crawler was interrupted.", "WARNING")
