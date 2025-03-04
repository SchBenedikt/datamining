import os
import re  # New import for validation
import psycopg2
import sqlite3  # For SQLite support
import tempfile   # For temporary file storage
import pandas as pd   # New import for Excel processing
from flask import Flask, jsonify, render_template, redirect, url_for, request, send_file  # send_file hinzugefügt
from dotenv import load_dotenv  # hinzugefügt

load_dotenv()  # hinzugefügt

db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
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

@app.route('/query', methods=["GET", "POST"])
def query_page():
    if request.method == "GET":
        return render_template("query.html", executed_query="")
    # POST method
    sql_query = request.form.get("sql")
    if not sql_query:
        return render_template("query.html", executed_query="")
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Nur SELECT-Abfragen sind erlaubt.", 400
    db_file = request.files.get("db_file")
    try:
        if db_file and db_file.filename:
            filename = db_file.filename.lower()
            if filename.endswith(".xlsx"):
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    db_file.save(tmp.name)
                    tmp_filename = tmp.name
                df = pd.read_excel(tmp_filename)
                # Load the Excel data into an in-memory SQLite table named "articles"
                conn = sqlite3.connect(":memory:")
                df.to_sql("articles", conn, if_exists="replace", index=False)
                cur = conn.cursor()
                cur.execute(sql_query)
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                rows = [dict(zip(colnames, row)) for row in rows]
                cur.close()
                conn.close()
                os.remove(tmp_filename)
            elif filename.endswith(".db"):
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    db_file.save(tmp.name)
                    tmp_filename = tmp.name
                conn = sqlite3.connect(tmp_filename)
                cur = conn.cursor()
                cur.execute(sql_query)
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                rows = [dict(zip(colnames, row)) for row in rows]
                cur.close()
                conn.close()
                os.remove(tmp_filename)
            else:
                return "Ungültiger Dateityp. Bitte laden Sie eine .xlsx oder .db Datei hoch.", 400
        else:
            conn = psycopg2.connect(**db_params)
            cur = conn.cursor()
            cur.execute(sql_query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description] if cur.description else []
            rows = [dict(zip(colnames, row)) for row in rows]
            cur.close()
            conn.close()
    except Exception as e:
        return f"Fehler beim Ausführen der Abfrage: {e}", 500
    return render_template("query.html", query_result={"cols": colnames, "rows": rows}, executed_query=sql_query)

@app.route('/export_db', methods=["GET"])
def export_db():
    # Daten aus PostgreSQL abfragen
    data = get_db_data()
    try:
        # Temporäre SQLite-Datei anlegen
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_filename = tmp.name
        conn = sqlite3.connect(tmp_filename)
        # Jede Tabelle in die SQLite-Datenbank exportieren
        for table_name, rows in data.items():
            if rows:
                import pandas as pd
                pd.DataFrame(rows).to_sql(table_name, conn, if_exists='replace', index=False)
            else:
                # Leere Tabelle: Spaltennamen ermitteln
                # Annahme: falls Tabelle leer ist, wird sie übersprungen oder mit einer Dummy-Spalte angelegt.
                conn.execute(f"CREATE TABLE {table_name} (dummy TEXT);")
        conn.commit()
        conn.close()
        return send_file(tmp_filename, as_attachment=True, download_name="export.db")
    except Exception as e:
        return f"Fehler beim Exportieren der Datenbank: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6600)