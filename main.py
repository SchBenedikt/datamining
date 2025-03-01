import requests
from bs4 import BeautifulSoup
import psycopg2
import json
from pyfiglet import figlet_format

# PostgreSQL connection details
db_params = {
    'dbname': 'web_crawler',
    'user': 'schaechner',
    'password': 'SchaechnerServer',
    'host': '192.168.188.36',
    'port': '6543'
}

def connect_db():
    return psycopg2.connect(**db_params)

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                title TEXT,
                url TEXT,
                date TEXT,
                author TEXT,
                category TEXT,
                keywords TEXT,
                word_count INTEGER,
                editor_abbr TEXT
            )
        ''')
        conn.commit()

def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr):
    with conn.cursor() as cur:
        cur.execute('''
            INSERT INTO articles (title, url, date, author, category, keywords, word_count, editor_abbr)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (title, url, date, author, category, keywords, word_count, editor_abbr))
        conn.commit()

def get_article_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract editor abbreviation
    editor_abbr = "N/A"
    editor_span = soup.find("span", class_="redakteurskuerzel")
    if editor_span:
        abbr_link = editor_span.find("a", class_="redakteurskuerzel__link")
        if abbr_link:
            # Extract the abbreviation from the link text or title
            if abbr_link.string:
                editor_abbr = abbr_link.string
            elif 'title' in abbr_link.attrs:
                # Extract the abbreviation from the email in href or from the title
                email = abbr_link.get('href', '')
                if email.startswith('mailto:'):
                    editor_abbr = email.split('@')[0].replace('mailto:', '')
                else:
                    # Extract from title which might be in format "Full Name (abbr)"
                    title = abbr_link.get('title', '')
                    editor_abbr = abbr_link.text

    # LDJSON extrahieren
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        return "N/A", "N/A", "N/A", None, editor_abbr

    try:
        data = json.loads(script_tag.string)
        author = ", ".join([a["name"] for a in data.get("author", [])]) if "author" in data else "N/A"
        category = data.get("articleSection", "N/A")
        keywords = ", ".join([t["name"] for t in data.get("about", [])]) if "about" in data else "N/A"
        word_count = data.get("wordCount", None)
        return author, category, keywords, word_count, editor_abbr
    except json.JSONDecodeError:
        return "N/A", "N/A", "N/A", None, editor_abbr

def crawl_heise(initial_year=2025, initial_month=3):
    year = initial_year
    month = initial_month

    while True:
        url = f"https://www.heise.de/newsticker/archiv/{year}/{month:02d}"
        print(f"Crawle URL: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')

        if not articles:
            print(f"Keine Artikel gefunden für {year}-{month:02d}. Beende Crawl.")
            break

        print(f"Gefundene Artikel: {len(articles)}")

        conn = connect_db()
        create_table(conn)

        for article in articles:
            try:
                title = article.find('h3').get_text(strip=True)
                link = article.find('a')['href']
                if not link.startswith("http"):
                    link = "https://www.heise.de" + link

                time_element = article.find('time')
                date = time_element['datetime'] if time_element else 'N/A'

                # Artikel öffnen und Details auslesen
                author, category, keywords, word_count, editor_abbr = get_article_details(link)

                insert_article(conn, title, link, date, author, category, keywords, word_count, editor_abbr)
                print(f"Gespeichert: {title} | {author} | {category} | {keywords} | Word Count: {word_count} | Editor: {editor_abbr}")

            except Exception as e:
                print(f"Fehler bei Artikel: {e}")

        conn.close()

        # Wechsel zum Vormonat
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        # Anzeige im Terminal, wenn Monat/Jahr gewechselt wird
        try:
            formatted_text = figlet_format(f"Wechsel zu: Jahr {year}, Monat {month:02d}")
            print(f"\033[1m{formatted_text}\033[0m")
        except ImportError:
            print(f"\033[1m{year}, Monat {month:02d}\033[0m")

if __name__ == '__main__':
    crawl_heise()
