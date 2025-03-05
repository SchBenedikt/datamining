import os
import psycopg2
<<<<<<< Updated upstream
from flask import Flask, jsonify, render_template, redirect, url_for, request
from dotenv import load_dotenv  # hinzugefügt

load_dotenv()  # hinzugefügt
=======
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sqlite3
import tempfile
from dash import Dash, html, dcc
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
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

# Neue Flask-Routen für query.html
@server.route('/')
def index():
    # Umleitung zur Dash-App
    return redirect('/dash/')

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

>>>>>>> Stashed changes
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=6800)
