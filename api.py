import os
import re  # New import for validation
import psycopg2
from flask import Flask, jsonify, render_template, redirect, url_for, request

db_params = {
    'dbname': os.getenv('DB_NAME', 'web_crawler'),
    'user': os.getenv('DB_USER', 'schaechner'),
    'password': os.getenv('DB_PASSWORD', 'SchaechnerServer'),
    'host': os.getenv('DB_HOST', '192.168.188.36'),
    'port': os.getenv('DB_PORT', '6543')
}

def get_db_statistics():
    """
    Berechnet Statistikdaten aus der Datenbank web_crawler:
    Ermittelt alle Tabellen im Schema 'public' und zählt die Anzahl der Zeilen.
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        # Alle Tabellen im public-Schema abrufen
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        stats = {}
        for (table_name,) in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cur.fetchone()[0]
            stats[table_name] = count
        cur.close()
        conn.close()
        return stats
    except Exception as e:
        print("Error calculating database statistics:", e)
        return None

def get_db_data():
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        # Alle Tabellen im public-Schema abrufen
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        data = {}
        for (table_name,) in tables:
            cur.execute(f"SELECT * FROM {table_name};")
            rows = cur.fetchall()
            # Get column names
            colnames = [desc[0] for desc in cur.description]
            # Convert rows to list of dicts
            data[table_name] = [dict(zip(colnames, row)) for row in rows]
        cur.close()
        conn.close()
        return data
    except Exception as e:
        print("Error fetching all data:", e)
        return None

def get_data_quality_stats(data):
    stats = {}
    for table_name, rows in data.items():
        total = len(rows)
        col_stats = {}
        if total > 0:
            for col in rows[0].keys():
                missing = 0
                for row in rows:
                    value = row[col]
                    if value is None or value == "" or (isinstance(value, str) and value.strip().upper() == "N/A"):
                        missing += 1
                col_stats[col] = missing
        stats[table_name] = {"total": total, "missing": col_stats}
    return stats

app = Flask(__name__)

@app.route("/")
def home():
    return "Visit /stats for database statistics."  # New default route

@app.route('/stats')
def stats():
    data = get_db_data()
    quality_stats = get_data_quality_stats(data)
    return render_template('stats.html', data=data, quality_stats=quality_stats)

# Remove /delete_articles route and add a generic route
@app.route('/delete_table', methods=['POST'])
def delete_table():
    table_name = request.form.get("table_name", "")
    # Validate table name (only letters, numbers, and underscores allowed)
    if not re.match(r'^[A-Za-z0-9_]+$', table_name):
        return "Ungültiger Tabellenname", 400
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return f"Fehler beim Löschen der Tabelle: {e}", 500
    return redirect(url_for('stats'))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6600)