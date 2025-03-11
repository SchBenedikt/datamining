import os
import psycopg2
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sqlite3
import tempfile
import requests
import re
from bs4 import BeautifulSoup
from dash import Dash, html, dcc
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv  # hinzugefügt
from datetime import datetime

load_dotenv()  # hinzugefügt


# Datenbank und Autoren-Netzwerk
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def get_author_network():
    """Liest Autoren aus der 'articles'-Tabelle und baut ein Netzwerk auf."""
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute("SELECT author FROM articles;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print("Error fetching authors:", e)
        rows = []

    authors_dict = {}
    all_authors = set()
    
    for (author_string,) in rows:
        if author_string:
            authors = [a.strip() for a in author_string.split(',') if a.strip()]
            authors = [author for author in authors if author.lower() != 'dpa']
            all_authors.update(authors)
            for author in authors:
                if author not in authors_dict:
                    authors_dict[author] = set()
                authors_dict[author].update(authors)

    for author in all_authors:
        if author not in authors_dict:
            authors_dict[author] = set()

    return authors_dict

authors_dict = get_author_network()
G = nx.Graph()

for author, coauthors in authors_dict.items():
    G.add_node(author)
    for coauthor in coauthors:
        if author != coauthor:
            G.add_edge(author, coauthor)

# Dynamischer k-Wert für ein natürlicheres Layout
k_value = 1 / np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 0.1
pos = nx.spring_layout(G, k=k_value, iterations=50)

# Leichte zufällige Verschiebung für isolierte Autoren
for node in G.nodes():
    if len(list(G.adj[node])) == 0:
        pos[node] = (pos[node][0] + np.random.uniform(-0.3, 0.3),
                     pos[node][1] + np.random.uniform(-0.3, 0.3))

# Kantenkoordinaten für Plotly
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

# Knotenkoodinaten für Plotly
node_x, node_y, node_text, node_adjacencies = [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_adjacencies.append(len(list(G.adj[node])))

# Kanten-Trace für Plotly
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Knoten-Trace für Plotly
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=node_adjacencies,
        size=10,
        colorbar=dict(
            thickness=15,
            title='Anzahl Verbindungen',
            xanchor='left',
        ),
        line_width=1)
)

# Netzwerk-Plot erstellen
fig_network = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Autoren Netzwerk',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

# Excel-Daten und Scatter-Plot
df = pd.read_excel("data/articles_export.xlsx")
df["date"] = pd.to_datetime(df["date"], format='ISO8601', utc=True)
df = df.assign(author=df['author'].str.split(', ')).explode('author')
fig_scatter = px.scatter(
    df, 
    x="date", 
    y="author", 
    title="Autoren pro Tag",
    labels={'date': 'Tag', 'author': 'Autor'},
    height=600
)

# Flask Server erstellen - wird sowohl für Dash als auch für Flask-Routen verwendet
server = Flask(__name__, template_folder='templates')
app = Dash(__name__, server=server)

# Dash Layout beibehalten
app.layout = html.Div([
    html.H1("Dashboard"),
    html.Div([
        html.H2("Autoren Netzwerk Diagramm"),
        dcc.Graph(figure=fig_network, style={"height": "100vh"}),
    ], style={"margin-bottom": "50px"}),
    html.Div([
        html.H2("Autoren vs. Tage"),
        dcc.Graph(figure=fig_scatter)
    ])
], style={"height": "100vh"})

# Hilfsfunktionen für den News-Feed
def get_all_categories():
    """Holt alle vorhandenen Kategorien aus der Datenbank"""
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT category FROM articles WHERE category != 'N/A' ORDER BY category;")
        categories = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return categories
    except Exception as e:
        print(f"Fehler beim Abrufen der Kategorien: {e}")
        return []

def search_articles(search_term=None, category=None, author=None, date_from=None, date_to=None, 
                   sort='date_desc', page=1, per_page=20):
    """
    Sucht Artikel basierend auf den angegebenen Kriterien
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Basis-Query
        query = "SELECT title, url, date, author, category, keywords FROM articles WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM articles WHERE 1=1"
        params = []
        
        # Suchbedingungen hinzufügen
        if search_term:
            query += " AND (title ILIKE %s OR keywords ILIKE %s)"
            count_query += " AND (title ILIKE %s OR keywords ILIKE %s)"
            search_param = f"%{search_term}%"
            params.extend([search_param, search_param])
            
        if category:
            query += " AND category = %s"
            count_query += " AND category = %s"
            params.append(category)
            
        if author:
            query += " AND author ILIKE %s"
            count_query += " AND author ILIKE %s"
            params.append(f"%{author}%")
            
        if date_from:
            query += " AND date >= %s"
            count_query += " AND date >= %s"
            params.append(date_from)
            
        if date_to:
            query += " AND date <= %s"
            count_query += " AND date <= %s"
            params.append(date_to)
        
        # Sortierung
        if sort == 'date_desc':
            query += " ORDER BY date DESC"
        elif sort == 'date_asc':
            query += " ORDER BY date ASC"
        elif sort == 'title_asc':
            query += " ORDER BY title ASC"
        elif sort == 'title_desc':
            query += " ORDER BY title DESC"
        
        # Gesamtanzahl der Treffer
        cur.execute(count_query, params)
        total_count = cur.fetchone()[0]
        
        # Paginierung
        offset = (page - 1) * per_page
        query += f" LIMIT {per_page} OFFSET {offset}"
        
        # Abfrage ausführen
        cur.execute(query, params)
        rows = cur.fetchall()
        
        articles = []
        for row in rows:
            article = {
                'title': row[0],
                'url': row[1],
                'date': row[2],
                'author': row[3],
                'category': row[4],
                'keywords': row[5]
            }
            articles.append(article)
            
        cur.close()
        conn.close()
        
        return articles, total_count
        
    except Exception as e:
        print(f"Fehler bei der Artikelsuche: {e}")
        return [], 0

# Berechnung des Paginierungs-Bereichs
def get_pagination_range(page, total_pages, window=2):
    """
    Berechnet einen Bereich von Seitenzahlen für die Paginierung
    
    Args:
        page: Aktuelle Seite
        total_pages: Gesamtanzahl der Seiten
        window: Anzahl der anzuzeigenden Seiten vor/nach der aktuellen Seite
        
    Returns:
        Liste mit Seitenzahlen
    """
    start_page = max(1, page - window)
    end_page = min(total_pages, page + window)
    
    return list(range(start_page, end_page + 1))

# Neue Flask-Routen für den News-Feed
@server.route('/')
def index():
    # Umleitung zur News Feed-Seite als Hauptseite
    return redirect('/news')

@server.route('/news')
def news_feed():
    # Parameter aus der Anfrage holen
    page = int(request.args.get('page', 1))
    search_term = request.args.get('search', '')
    category = request.args.get('category', '')
    author = request.args.get('author', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    sort = request.args.get('sort', 'date_desc')
    
    # Artikel suchen (feste Anzahl von 20 Artikeln pro Seite)
    per_page = 20
    articles, total_count = search_articles(
        search_term=search_term,
        category=category,
        author=author,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        page=page,
        per_page=per_page
    )
    
    # Alle verfügbaren Kategorien für das Dropdown-Menü holen
    categories = get_all_categories()
    
    # Paginierungs-Informationen
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    # Berechne den Paginierungsbereich hier anstatt im Template
    pagination_range = get_pagination_range(page, total_pages)
    
    return render_template(
        'news_feed.html',
        articles=articles,
        page=page,
        total_pages=total_pages,
        total_articles=total_count,
        search_term=search_term,
        category=category,
        author=author,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        categories=categories,
        pagination_range=pagination_range
    )

# Bestehende Flask-Routen für query.html
@server.route('/query', methods=['GET', 'POST'])
def query():
    query_result = None
    executed_query = None
    
    if request.method == 'POST':
        # SQL-Abfrage aus dem Formular holen
        sql = request.form.get('sql')
        executed_query = sql
        
        # Prüfen ob eine Datei hochgeladen wurde
        if 'db_file' in request.files and request.files['db_file'].filename:
            file = request.files['db_file']
            filename = secure_filename(file.filename)
            
            if filename.endswith('.db'):
                # SQLite-Datei verarbeiten
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    file.save(tmp.name)
                    conn = sqlite3.connect(tmp.name)
                    try:
                        df = pd.read_sql_query(sql, conn)
                        query_result = {"cols": df.columns.tolist(), "rows": df.to_dict('records')}
                    except Exception as e:
                        query_result = {"error": str(e)}
                    finally:
                        conn.close()
                        os.unlink(tmp.name)
            
            elif filename.endswith('.xlsx'):
                # Excel-Datei verarbeiten
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    file.save(tmp.name)
                    try:
                        df = pd.read_excel(tmp.name)
                        # SQL über Pandas ausführen
                        result_df = df.query(sql) if sql.strip().lower().startswith('select') else df
                        query_result = {"cols": result_df.columns.tolist(), "rows": result_df.to_dict('records')}
                    except Exception as e:
                        query_result = {"error": str(e)}
                    finally:
                        os.unlink(tmp.name)
        else:
            # Verwende die PostgreSQL-Verbindung wenn keine Datei hochgeladen wurde
            try:
                conn = psycopg2.connect(**db_params)
                df = pd.read_sql_query(sql, conn)
                query_result = {"cols": df.columns.tolist(), "rows": df.to_dict('records')}
                conn.close()
            except Exception as e:
                query_result = {"error": str(e)}
    
    return render_template('query.html', query_result=query_result, executed_query=executed_query)

@server.route('/export_db')
def export_db():
    # Erstellen einer temporären SQLite-Datenbank mit den Artikeln
    try:
        conn = psycopg2.connect(**db_params)
        df = pd.read_sql_query("SELECT * FROM articles", conn)
        conn.close()
        
        # Temporäre SQLite-DB erstellen
        db_path = tempfile.mktemp(suffix='.db')
        sqlite_conn = sqlite3.connect(db_path)
        df.to_sql('articles', sqlite_conn, if_exists='replace', index=False)
        sqlite_conn.close()
        
        # DB als Download anbieten
        return send_file(
            db_path,
            as_attachment=True,
            download_name='articles_export.db',
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return str(e)

# Neue Route für die Artikel-Vorschau
@server.route('/article_preview')
def article_preview():
    url = request.args.get('url', '')
    if not url:
        return jsonify({'success': False, 'error': 'Keine URL angegeben'})
    
    try:
        # Artikel-Seite abrufen
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'HTTP Fehler: {response.status_code}'})
        
        # HTML parsen
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Den Hauptinhalt des Artikels extrahieren
        article_content = soup.find('div', class_='article-content')
        
        if not article_content:
            return jsonify({'success': False, 'error': 'Artikel-Inhalt konnte nicht gefunden werden'})
        
        # Unerwünschte Elemente entfernen (Werbung, etc.)
        for ad in article_content.find_all(['div'], class_=['ad', 'ad-label', 'ad--sticky', 'ad--inread', 'inread-cls-reduc']):
            ad.decompose()
        
        for paternoster in article_content.find_all('a-paternoster'):
            paternoster.decompose()
        
        # Nur den eigentlichen Text des Artikels behalten
        content_html = ""
        for element in article_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']):
            content_html += str(element)
        
        # Falls der Inhalt leer ist, versuchen wir es mit einer anderen Methode
        if not content_html.strip():
            # Alternative: Den Text zwischen RSPEAK_START und RSPEAK_STOP finden
            html_str = str(soup)
            content_parts = []
            
            start_markers = [m.start() for m in re.finditer('<!-- RSPEAK_START -->', html_str)]
            end_markers = [m.start() for m in re.finditer('<!-- RSPEAK_STOP -->', html_str)]
            
            if start_markers and end_markers and len(start_markers) == len(end_markers):
                for i in range(len(start_markers)):
                    if i < len(end_markers):
                        start_pos = start_markers[i] + len('<!-- RSPEAK_START -->')
                        end_pos = end_markers[i]
                        if start_pos < end_pos:
                            content_parts.append(html_str[start_pos:end_pos].strip())
                
                content_html = ''.join(content_parts)
                # HTML-Parser nochmal anwenden, um nur die relevanten Elemente zu behalten
                content_soup = BeautifulSoup(content_html, 'html.parser')
                content_html = ''.join(str(el) for el in content_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']))
        
        if not content_html.strip():
            return jsonify({'success': False, 'error': 'Artikel-Inhalt konnte nicht extrahiert werden'})
        
        return jsonify({
            'success': True,
            'content': content_html,
            'url': url
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=6800)
