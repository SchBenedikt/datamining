import os
import time
import requests
import psycopg2
import json
from datetime import datetime
from bs4 import BeautifulSoup
from notification import send_notification

# DB-Verbindungsdetails (aus Umgebungsvariablen)
db_params = {
    'dbname': os.getenv('DB_NAME', 'web_crawler'),
    'user': os.getenv('DB_USER', 'schaechner'),
    'password': os.getenv('DB_PASSWORD', 'SchaechnerServer'),
    'host': os.getenv('DB_HOST', '192.168.188.36'),
    'port': os.getenv('DB_PORT', '6543')
}

def connect_db():
    try:
        return psycopg2.connect(**db_params)
    except Exception as e:
        print(f"DB-Verbindungsfehler: {e}")
        raise

def article_exists(conn, url):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM articles WHERE url=%s", (url,))
        return cur.fetchone() is not None

def get_article_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Ähnlich wie in main.py:
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
    except Exception as e:
        return "N/A", "N/A", "N/A", None, editor_abbr, site_name, alt_data

def insert_article(conn, title, url, date, author, category, keywords, word_count, editor_abbr, site_name, alt_data):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO articles (title, url, date, author, category, keywords, word_count, editor_abbr, site_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET title = EXCLUDED.title;
        """, (title, url, date, author, category, keywords, word_count, editor_abbr, site_name))
    conn.commit()

def crawl_current():
    # Aktuelles Datum im Format YYYY-MM-DD
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now()
    # Verwende das aktuelle Archiv (aktueller Monat/Jahr) – hier werden aber nur Artikel des heutigen Tages verarbeitet
    archive_url = f"https://www.heise.de/newsticker/archiv/{now.year}/{now.month:02d}"
    print(f"Crawle neue Artikel vom {today} über: {archive_url}")
    
    try:
        response = requests.get(archive_url)
        response.raise_for_status()
    except Exception as e:
        print(f"HTTP-Fehler: {e}")
        send_notification("Crawling Fehler", f"HTTP-Fehler: {e}", os.getenv('ALERT_EMAIL'))
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    for a in articles:
        time_elem = a.find('time')
        if time_elem and time_elem.has_attr('datetime'):
            d = time_elem['datetime'][:10]
        else:
            continue
        # Verarbeite nur Artikel, die heute veröffentlicht wurden
        if d != today:
            continue
        a_link = a.find('a')
        if not a_link:
            continue
        link = a_link['href']
        if not link.startswith("http"):
            link = "https://www.heise.de" + link
        if "${" in link:
            continue
        # Prüfe, ob der Artikel bereits existiert
        conn = connect_db()
        if article_exists(conn, link):
            conn.close()
            continue
        # Extrahiere Details und speichere den Artikel
        title = a.find('h3').get_text(strip=True) if a.find('h3') else "Ohne Titel"
        details = get_article_details(link)
        author, category, keywords, word_count, editor_abbr, site_name, alt_data = details
        insert_article(conn, title, link, d, author, category, keywords, word_count, editor_abbr, site_name, alt_data)
        conn.close()
        print(f"Inserted: {title} ({d})")
    
    print("Aktueller Crawl-Durchlauf abgeschlossen.\n")

def main():
    while True:
        crawl_current()
        # Warte 5 Minuten
        time.sleep(300)

if __name__ == '__main__':
    main()
