from flask import Flask, jsonify
import psycopg2
import os

app = Flask(__name__)

# Database connection details using environment variables
db_params = {
    'dbname': os.getenv('DB_NAME', 'web_crawler'),
    'user': os.getenv('DB_USER', 'schaechner'),
    'password': os.getenv('DB_PASSWORD', 'SchaechnerServer'),
    'host': os.getenv('DB_HOST', '192.168.188.36'),
    'port': os.getenv('DB_PORT', '6543')
}

def get_db_connection():
    return psycopg2.connect(**db_params)

@app.route('/api/stats')
def stats():
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Total articles
        cur.execute("SELECT COUNT(*) FROM articles")
        total = cur.fetchone()[0]
        # Articles with alternate URLs (non-empty JSON after conversion)
        cur.execute("SELECT COUNT(*) FROM articles WHERE alternate_urls IS NOT NULL AND alternate_urls != '[]'")
        with_alternate = cur.fetchone()[0]
        # Articles without alternate URLs
        cur.execute("SELECT COUNT(*) FROM articles WHERE alternate_urls IS NULL OR alternate_urls = '[]'")
        without_alternate = cur.fetchone()[0]
        # Articles with valid keywords (not equal to 'N/A')
        cur.execute("SELECT COUNT(*) FROM articles WHERE keywords IS NOT NULL AND keywords != 'N/A'")
        with_keywords = cur.fetchone()[0]
        # Articles without keywords
        cur.execute("SELECT COUNT(*) FROM articles WHERE keywords IS NULL OR keywords = 'N/A'")
        without_keywords = cur.fetchone()[0]
        # Articles with word count present
        cur.execute("SELECT COUNT(*) FROM articles WHERE word_count IS NOT NULL")
        with_word_count = cur.fetchone()[0]
        # Articles without word count data
        cur.execute("SELECT COUNT(*) FROM articles WHERE word_count IS NULL")
        without_word_count = cur.fetchone()[0]
    conn.close()

    stats_data = {
        "total": total,
        "withAlternate": with_alternate,
        "withoutAlternate": without_alternate,
        "withKeywords": with_keywords,
        "withoutKeywords": without_keywords,
        "withWordCount": with_word_count,
        "withoutWordCount": without_word_count
    }
    return jsonify(stats_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
