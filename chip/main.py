import os
import sys
import json
import requests
import psycopg2
from psycopg2.extras import Json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pyfiglet import figlet_format
from notification import send_notification
from datetime import datetime
from urllib.parse import urljoin

# Add parent directory to path to import init_database
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from init_database import initialize_database

# Load environment variables from root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# PostgreSQL connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME', 'datamining'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
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

def send_notification_safe(subject, body, to_email):
    """Sends notification only if email is configured."""
    if to_email and os.getenv('EMAIL_USER') and os.getenv('EMAIL_PASSWORD'):
        try:
            send_notification(subject, body, to_email)
        except Exception as e:
            print_status(f"Failed to send notification: {e}", "WARNING")

# -----------------------------------------------------------------------------
# TABLE AND CRAWL STATE FUNCTIONS
# -----------------------------------------------------------------------------
def create_table():
    """Creates the chip table if it doesn't exist yet."""
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chip (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE,
                    headline TEXT,
                    description TEXT,
                    author TEXT,
                    date_published TEXT,
                    date_modified TEXT,
                    article_type TEXT,
                    image_url TEXT,
                    image_caption TEXT,
                    video_url TEXT,
                    video_duration TEXT,
                    category TEXT,
                    page_level1 TEXT,
                    page_level2 TEXT,
                    page_level3 TEXT
                );
            """)
            
            conn.commit()
        conn.close()
        print_status("Table 'chip' was created or already exists.", "INFO")
    except Exception as e:
        print_status(f"Error creating table: {e}", "ERROR")
        send_notification_safe("Chip Mining Error", f"Error creating table: {e}", os.getenv('ALERT_EMAIL'))

def create_crawl_state_table_chip(conn):
    """Creates the crawl_state table for storing crawl progress."""
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chip_crawl_state (
                id SERIAL PRIMARY KEY,
                page INTEGER DEFAULT 1,
                article_index INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def get_crawl_state(conn):
    """Reads the saved crawl progress; returns (page, article_index) or None."""
    with conn.cursor() as cur:
        cur.execute("SELECT page, article_index FROM chip_crawl_state ORDER BY id DESC LIMIT 1;")
        row = cur.fetchone()
    return row if row else None

def update_crawl_state(conn, page, article_index):
    """Updates or saves the current crawl state in the database."""
    with conn.cursor() as cur:
        # Check if a record exists
        cur.execute("SELECT id FROM chip_crawl_state ORDER BY id DESC LIMIT 1;")
        existing = cur.fetchone()
        
        if existing:
            cur.execute(
                """UPDATE chip_crawl_state SET page = %s, article_index = %s, last_updated = CURRENT_TIMESTAMP
                   WHERE id = %s;""",
                (page, article_index, existing[0])
            )
        else:
            cur.execute(
                """INSERT INTO chip_crawl_state (page, article_index) VALUES (%s, %s);""",
                (page, article_index)
            )
    conn.commit()

# -----------------------------------------------------------------------------
# ARTICLE PROCESSING
# -----------------------------------------------------------------------------
def insert_chip_article(conn, url, headline, description, author, date_published, date_modified,
                        article_type, image_url, image_caption, video_url, video_duration,
                        category, page_level1, page_level2, page_level3):
    """
    Inserts an article into the chip table or updates it
    if an entry with the same URL already exists.
    """
    replaced = False
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM chip WHERE url=%s", (url,))
        if cur.fetchone():
            replaced = True
        base_columns = ["url", "headline", "description", "author", "date_published", "date_modified",
                        "article_type", "image_url", "image_caption", "video_url", "video_duration",
                        "category", "page_level1", "page_level2", "page_level3"]
        base_values = [url, headline, description, author, date_published, date_modified,
                       article_type, image_url, image_caption, video_url, video_duration,
                       category, page_level1, page_level2, page_level3]
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

def scrape_article_details(conn, url):
    """
    Scrapes the details of a single article from CHIP using JSON-LD structured data.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize default values
            headline = "Unknown"
            description = "Unknown"
            author = "Unknown"
            date_published = "Unknown"
            date_modified = "Unknown"
            article_type = "Unknown"
            image_url = None
            image_caption = None
            video_url = None
            video_duration = None
            category = "Unknown"
            page_level1 = None
            page_level2 = None
            page_level3 = None
            
            # Extract JSON-LD data (NewsArticle)
            json_ld_scripts = soup.find_all("script", {"type": "application/ld+json"})
            
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    
                    # NewsArticle data
                    if isinstance(data, dict) and data.get("@type") == "NewsArticle":
                        headline = data.get("headline", headline)
                        description = data.get("description", description)
                        date_published = data.get("datePublished", date_published)
                        date_modified = data.get("dateModified", date_modified)
                        
                        # Author
                        author_data = data.get("author", {})
                        if isinstance(author_data, dict):
                            author = author_data.get("name", author)
                        elif isinstance(author_data, list) and len(author_data) > 0:
                            if isinstance(author_data[0], dict):
                                author = author_data[0].get("name", author)
                            elif isinstance(author_data[0], str):
                                author = author_data[0]
                        
                        # Image data
                        images = data.get("image", [])
                        if isinstance(images, list) and len(images) > 0:
                            first_image = images[0]
                            if isinstance(first_image, dict):
                                image_url = first_image.get("url")
                                image_caption = first_image.get("caption") or first_image.get("alternateName")
                        elif isinstance(images, dict):
                            image_url = images.get("url")
                            image_caption = images.get("caption") or images.get("alternateName")
                        
                        # Video data
                        video = data.get("video", {})
                        if isinstance(video, dict):
                            video_url = video.get("contentUrl")
                            video_duration = video.get("duration")
                    
                    # BreadcrumbList data (for category)
                    elif isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
                        items = data.get("itemListElement", [])
                        if isinstance(items, list) and len(items) >= 3:
                            # items[0] = Home, items[1] = News, items[2+] = categories
                            if len(items) > 1 and isinstance(items[1], dict):
                                item_data = items[1].get("item", {})
                                page_level1 = item_data.get("name") if isinstance(item_data, dict) else None
                            if len(items) > 2 and isinstance(items[2], dict):
                                item_data = items[2].get("item", {})
                                page_level2 = item_data.get("name") if isinstance(item_data, dict) else None
                            if len(items) > 3 and isinstance(items[3], dict):
                                item_data = items[3].get("item", {})
                                page_level3 = item_data.get("name") if isinstance(item_data, dict) else None
                            category = page_level2 or page_level1 or "Unknown"
                
                except json.JSONDecodeError as e:
                    print_status(f"Warning: Could not parse JSON-LD in {url}: {e}", "WARNING")
                    continue
                except Exception as e:
                    print_status(f"Warning: Error processing JSON-LD in {url}: {e}", "WARNING")
                    continue
            
            # Fallback: Extract from meta tags if JSON-LD failed
            if headline == "Unknown":
                # Try og:title
                title_meta = soup.find("meta", attrs={"property": "og:title"})
                if title_meta and title_meta.get("content"):
                    headline = title_meta["content"]
                else:
                    # Try title tag with itemprop
                    title_tag = soup.find("title", itemprop="name")
                    if title_tag:
                        # Remove " - CHIP" suffix if present
                        headline = title_tag.get_text().replace(" - CHIP", "").strip()
            
            if description == "Unknown":
                # Try meta description with itemprop first
                desc_meta = soup.find("meta", attrs={"name": "description", "itemprop": "description"})
                if desc_meta and desc_meta.get("content"):
                    description = desc_meta["content"]
                else:
                    # Try og:description
                    og_desc = soup.find("meta", attrs={"property": "og:description"})
                    if og_desc and og_desc.get("content"):
                        description = og_desc["content"]
                    else:
                        # Try twitter:description
                        twitter_desc = soup.find("meta", attrs={"name": "twitter:description"})
                        if twitter_desc and twitter_desc.get("content"):
                            description = twitter_desc["content"]
            
            if author == "Unknown":
                author_meta = soup.find("meta", attrs={"name": "author"})
                if author_meta and author_meta.get("content"):
                    author = author_meta["content"]
            
            if article_type == "Unknown":
                type_meta = soup.find("meta", attrs={"property": "og:type"})
                if type_meta and type_meta.get("content"):
                    article_type = type_meta["content"]
            
            if not image_url:
                # Try og:image
                og_image = soup.find("meta", attrs={"property": "og:image"})
                if og_image and og_image.get("content"):
                    image_url = og_image["content"]
                else:
                    # Try twitter:image
                    twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
                    if twitter_image and twitter_image.get("content"):
                        image_url = twitter_image["content"]
            
            if date_published == "Unknown":
                # Try <time> tag with itemprop="datePublished"
                time_tag = soup.find("time", itemprop="datePublished")
                if time_tag and time_tag.get("content"):
                    date_published = time_tag["content"]
                else:
                    # Try meta tag
                    date_meta = soup.find("meta", attrs={"name": "date"})
                    if date_meta and date_meta.get("content"):
                        date_published = date_meta["content"]
            
            # Validiere Datumswert - setze auf None wenn ungültig
            if date_published in ["Unknown", "N/A", "", None]:
                date_published = None
            elif date_published:
                # Versuche das Datum zu parsen um Gültigkeit zu prüfen
                try:
                    from datetime import datetime
                    # ISO 8601 Format prüfen
                    datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    print_status(f"Ungültiges Datumsformat in {url}: {date_published}", "WARNING")
                    date_published = None
            
            if date_modified == "Unknown":
                # Try <meta itemprop="dateModified">
                modified_meta = soup.find("meta", itemprop="dateModified")
                if modified_meta and modified_meta.get("content"):
                    date_modified = modified_meta["content"]
            
            # Validiere date_modified
            if date_modified in ["Unknown", "N/A", "", None]:
                date_modified = None
            elif date_modified:
                try:
                    from datetime import datetime
                    datetime.fromisoformat(date_modified.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    print_status(f"Ungültiges dateModified Format in {url}: {date_modified}", "WARNING")
                    date_modified = None
            
            if author == "Unknown":
                # Try <span itemprop="name"> inside author section
                author_span = soup.find("span", itemprop="author")
                if author_span:
                    name_span = author_span.find("span", itemprop="name")
                    if name_span:
                        author_link = name_span.find("a")
                        if author_link:
                            author = author_link.get_text().strip()
                        else:
                            author = name_span.get_text().strip()
            
            # Fallback for category: Parse JavaScript globalOptions
            if category == "Unknown" or not page_level1:
                try:
                    # Find script tag containing globalOptions
                    scripts = soup.find_all("script")
                    for script in scripts:
                        if script.string and "var globalOptions" in script.string:
                            script_content = script.string
                            
                            # Extract category using regex
                            import re
                            category_match = re.search(r'"category"\s*:\s*"([^"]+)"', script_content)
                            keyword_match = re.search(r'"keyword"\s*:\s*"([^"]+)"', script_content)
                            adunit2_match = re.search(r'"adunit2"\s*:\s*"([^"]+)"', script_content)
                            
                            # Build category hierarchy
                            extracted_category = None
                            if category_match:
                                extracted_category = category_match.group(1)
                                if not page_level1 or page_level1 == "Unknown":
                                    page_level1 = extracted_category
                                if category == "Unknown":
                                    category = extracted_category
                            
                            if adunit2_match:
                                adunit2_value = adunit2_match.group(1)
                                # Only use if different from main category
                                if extracted_category is None or adunit2_value != extracted_category:
                                    if not page_level2:
                                        page_level2 = adunit2_value
                            
                            if keyword_match:
                                keyword_value = keyword_match.group(1)
                                # Keywords often use underscore format like "Geld_Finanzen_Recht"
                                # Replace underscores with spaces for readability
                                keyword_readable = keyword_value.replace("_", " ")
                                if not page_level2:
                                    page_level2 = keyword_readable
                                elif not page_level3:
                                    page_level3 = keyword_readable
                                
                                # Update main category if still unknown
                                if category == "Unknown":
                                    category = keyword_readable
                            
                            break
                except Exception as e:
                    print_status(f"Warning: Could not parse JavaScript globalOptions in {url}: {e}", "WARNING")
            
            # Insert into database
            replaced = insert_chip_article(
                conn, url, headline, description, author, date_published, date_modified,
                article_type, image_url, image_caption, video_url, video_duration,
                category, page_level1, page_level2, page_level3
            )
            
            if replaced:
                print_status(f"{date_published[:10] if date_published and date_published != 'Unknown' else 'Unknown'} - {url} already exists.", "INFO")
            else:
                print_status(f"{date_published[:10] if date_published and date_published != 'Unknown' else 'Unknown'} - {url} has been inserted.", "INFO")
        else:
            print_status(f"Error retrieving article {url}: {response.status_code}", "ERROR")
            send_notification_safe("Chip Mining Error", f"Error retrieving article {url}: {response.status_code}", os.getenv('ALERT_EMAIL'))
    except Exception as e:
        print_status(f"Error scraping article {url}: {e}", "ERROR")
        send_notification_safe("Chip Mining Error", f"Error scraping article {url}: {e}", os.getenv('ALERT_EMAIL'))

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
        send_notification_safe("Chip Mining Started", f"Chip Mining started on page {page}", os.getenv('ALERT_EMAIL'))
    
    while True:
        url = f"https://www.chip.de/nachrichten/{page}"
        print_status(f"Crawling URL: {url}", "INFO")
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception as e:
            print_status(f"HTTP error retrieving {url}: {e}", "ERROR")
            send_notification_safe("Chip Mining Error", f"HTTP error retrieving {url}: {e}", os.getenv('ALERT_EMAIL'))
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
            full_url = urljoin("https://www.chip.de", link)
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
    # Initialize database on startup
    print_status("Initializing database...", "INFO")
    initialize_database()
    
    try:
        scrape_chip_news()
    except KeyboardInterrupt:
        print_status("Chip Mining was interrupted.", "WARNING")
