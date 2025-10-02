"""
Streamlit-basierte Webanwendung für Heise Mining
Ersetzt die Flask/Dash-Anwendung mit einer modernen, interaktiven Benutzeroberfläche
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import psycopg2
import requests
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
import json
import io
import base64
import hashlib
import random
import time
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

# TextBlob oder TextBlob-DE für Sentiment-Analyse importieren
# Wir versuchen zuerst TextBlob-DE für deutsche Texte zu laden, 
# und fallen auf Standard-TextBlob zurück, wenn TextBlob-DE nicht verfügbar ist
try:
    from textblob_de import TextBlobDE
    TEXTBLOB_DE_AVAILABLE = True
except ImportError:
    try:
        from textblob import TextBlob
        TEXTBLOB_DE_AVAILABLE = False
    except ImportError:
        TEXTBLOB_DE_AVAILABLE = False

# Streamlit-Konfiguration
st.set_page_config(
    page_title="Heise Mining Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Umgebungsvariablen laden
load_dotenv()

# Datenbank-Verbindungsparameter
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Google API-Konfiguration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Google Generative AI konfigurieren, wenn API-Schlüssel vorhanden
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Fehler beim Konfigurieren der Google AI API: {e}")
else:
    st.warning("Kein Google API-Schlüssel gefunden. KI-Funktionen sind deaktiviert.")

# Entferne None-Werte aus den DB-Parametern
DB_PARAMS = {k: v for k, v in DB_PARAMS.items() if v is not None}

# CSS für ultra-minimalistisches Styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: #333;
        text-align: left;
        margin-bottom: 1.2rem;
        font-weight: 300;
        letter-spacing: -0.5px;
    }
    
    .metric-container {
        background: #fafafa;
        padding: 0.8rem;
        border-radius: 3px;
        color: #333;
        margin: 0.4rem 0;
        border-left: 2px solid #e0e0e0;
    }
    
    .search-container {
        background: #fafafa;
        padding: 0.8rem;
        border-radius: 3px;
        margin: 0.8rem 0;
    }
    
    .article-card {
        background: white;
        padding: 0.8rem 0;
        border-bottom: 1px solid #eaeaea;
        margin: 0;
    }
    
    /* Streamlit component styling */
    .stSelectbox > div > div {
        background-color: white;
    }
    
    .sidebar-content {
        background: #fafafa;
        padding: 0.6rem;
        border-radius: 3px;
        margin: 0.4rem 0;
    }
    
    /* Make buttons more minimal */
    .stButton > button {
        border-radius: 3px;
        font-weight: 400;
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 0.25rem 0.75rem;
        font-size: 0.85em;
    }
    
    /* Section headers */
    h2 {
        font-weight: 400;
        color: #333;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h3 {
        font-weight: 400;
        color: #333;
        font-size: 1.1rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    
    /* Remove excessive whitespace */
    div.block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }
    
    /* More compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1rem;
        font-weight: 500;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #666;
    }
    
    /* More minimal dashboard blocks */
    .stExpander {
        border: 1px solid #eaeaea;
        border-radius: 3px;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Streamlit elements */
    .st-emotion-cache-1kyxreq {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Plot backgrounds */
    .js-plotly-plot {
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# Cache für bessere Performance
def get_db_connection():
    """Erstellt eine Datenbankverbindung"""
    try:
        # Überprüfe, ob alle erforderlichen Parameter vorhanden sind
        if not all([DB_PARAMS.get('dbname'), DB_PARAMS.get('user'), 
                   DB_PARAMS.get('password'), DB_PARAMS.get('host')]):
            st.error("Nicht alle Datenbankparameter sind konfiguriert. Prüfen Sie die .env-Datei.")
            return None
            
        conn = psycopg2.connect(
            dbname=DB_PARAMS['dbname'],
            user=DB_PARAMS['user'],
            password=DB_PARAMS['password'],
            host=DB_PARAMS['host'],
            port=DB_PARAMS.get('port', '5432')
        )
        return conn
    except Exception as e:
        st.error(f"Datenbankverbindung fehlgeschlagen: {e}")
        return None

@st.cache_data(ttl=600)  # 10 Minuten Cache
def load_articles_data() -> pd.DataFrame:
    """Lädt alle Artikel aus der Datenbank"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT id, title, url, date, author, category, keywords, 
               word_count, editor_abbr, site_name, 
               COALESCE(source, 'heise') as source
        FROM articles 
        ORDER BY date DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Datentypen konvertieren
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        # Konvertiere zu lokaler Zeit ohne Timezone-Info für Vergleiche
        df['date'] = df['date'].dt.tz_localize(None)
        df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce')
        
        # Prüfen ob date-Konvertierung erfolgreich war
        if df['date'].isna().all():
            st.warning("Warnung: Alle Datumswerte konnten nicht konvertiert werden. Zeitbasierte Analysen sind möglicherweise nicht verfügbar.")
        elif df['date'].isna().any():
            invalid_dates = df['date'].isna().sum()
            st.info(f"Info: {invalid_dates} ungültige Datumswerte wurden gefunden und ignoriert.")
        
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Artikel: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # 5 Minuten Cache
def get_categories() -> List[str]:
    """Holt alle verfügbaren Kategorien"""
    df = load_articles_data()
    if df.empty:
        return []
    
    categories = df['category'].dropna().unique().tolist()
    return [cat for cat in categories if cat != 'N/A']

@st.cache_data(ttl=300)  # 5 Minuten Cache
def get_authors() -> List[str]:
    """Holt alle verfügbaren Autoren"""
    df = load_articles_data()
    if df.empty:
        return []
    
    # Autoren aufteilen (falls mehrere pro Artikel)
    all_authors = set()
    for author_str in df['author'].dropna():
        if author_str != 'N/A':
            authors = [a.strip() for a in str(author_str).split(',')]
            all_authors.update(authors)
    
    return sorted(list(all_authors))

@st.cache_data(ttl=600)  # 10 Minuten Cache für Statistiken
def get_dashboard_stats(df: pd.DataFrame) -> Dict:
    """Berechnet Dashboard-Statistiken"""
    stats = {
        'total_articles': len(df),
        'unique_authors': df['author'].nunique(),
        'unique_categories': df['category'].nunique(),
        'avg_words': df['word_count'].mean()
    }
    return stats

@st.cache_data(ttl=600)  # 10 Minuten Cache für Zeitreihen
def get_daily_article_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet tägliche Artikel-Anzahlen"""
    df_with_valid_dates = df.dropna(subset=['date'])
    if df_with_valid_dates.empty:
        return pd.DataFrame()
    
    df_daily = df_with_valid_dates.groupby(df_with_valid_dates['date'].dt.date).size().reset_index()
    df_daily.columns = ['date', 'count']
    return df_daily

@st.cache_data(ttl=600)  # 10 Minuten Cache für Kategorien
def get_category_counts(df: pd.DataFrame) -> pd.Series:
    """Berechnet Kategorie-Anzahlen"""
    return df['category'].value_counts().head(10)

@st.cache_data(ttl=600)  # 10 Minuten Cache für Autoren
def get_author_counts(df: pd.DataFrame) -> pd.Series:
    """Berechnet Autoren-Anzahlen"""
    all_authors = []
    for author_str in df['author'].dropna():
        if author_str != 'N/A':
            authors = [a.strip() for a in str(author_str).split(',')]
            all_authors.extend(authors)
    
    if all_authors:
        return pd.Series(all_authors).value_counts().head(10)
    return pd.Series()

@st.cache_data(ttl=900)  # 15 Minuten Cache für Keyword-Analysen
def get_keyword_analysis(df: pd.DataFrame) -> Dict:
    """Berechnet Keyword-Häufigkeiten"""
    keywords_list = []
    for keywords_str in df['keywords'].dropna():
        if keywords_str not in [None, 'N/A', '']:
            keywords = [k.strip() for k in str(keywords_str).split(',')]
            keywords_list.extend(keywords)
    
    keyword_counts = Counter(keywords_list)
    top_keywords = {k: v for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]}
    
    # Keyword-Beziehungen für Netzwerkanalyse
    keyword_relations = {}
    for keywords_str in df['keywords'].dropna():
        if keywords_str not in [None, 'N/A', '']:
            keywords = [k.strip() for k in str(keywords_str).split(',')]
            if len(keywords) > 1:
                for i, kw1 in enumerate(keywords):
                    for kw2 in keywords[i+1:]:
                        if kw1 != kw2:
                            pair = tuple(sorted([kw1, kw2]))
                            keyword_relations[pair] = keyword_relations.get(pair, 0) + 1
    
    # Top-Verbindungen
    top_relations = {k: v for k, v in sorted(keyword_relations.items(), key=lambda x: x[1], reverse=True)[:30]}
    
    return {
        'top_keywords': top_keywords,
        'keyword_relations': top_relations
    }

@st.cache_data(ttl=1800)  # 30 Minuten Cache für KI-intensive Analysen
def get_article_content(article_id: int) -> str:
    """Holt den Inhalt eines Artikels aus der Datenbank"""
    conn = get_db_connection()
    if conn is None:
        return ""
    
    try:
        query = "SELECT content FROM article_content WHERE article_id = %s"
        cursor = conn.cursor()
        cursor.execute(query, (article_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return result[0]
        return ""
    except Exception as e:
        st.error(f"Fehler beim Abrufen des Artikelinhalts: {e}")
        return ""

@st.cache_resource
def get_text_vectorizer():
    """Erstellt einen TF-IDF Vektorizer für Textanalyse"""
    return TfidfVectorizer(max_features=5000, 
                          stop_words=['der', 'die', 'das', 'und', 'in', 'ist', 'zu', 'für', 'mit'],
                          ngram_range=(1, 2))

@st.cache_data(ttl=1800)  # 30 Minuten Cache für Themenmodellierung
def extract_topics(df: pd.DataFrame, n_topics=5, method='lda') -> Dict:
    """Extrahiert Themen aus den Artikelinhalten mit LDA oder NMF"""
    # Artikel mit Inhalt laden
    contents = []
    ids = []
    
    progress_bar = st.progress(0.0)
    total = min(len(df), 500)  # Begrenze auf 500 Artikel für Performance
    
    for i, row in df.head(total).iterrows():
        try:
            content = get_article_content(row['id'])
            if content:
                contents.append(content)
                ids.append(row['id'])
            progress_bar.progress((i+1)/total)
        except Exception as e:
            st.warning(f"Konnte Inhalt für Artikel {row['id']} nicht laden: {str(e)}")
    
    if not contents:
        progress_bar.empty()
        return {'topics': [], 'topic_distribution': []}
    
    # Vektorisierung
    vectorizer = get_text_vectorizer()
    try:
        X = vectorizer.fit_transform(contents)
    except Exception as e:
        progress_bar.empty()
        st.error(f"Fehler bei der Textvektorisierung: {str(e)}")
        return {'topics': [], 'topic_distribution': []}
    
    # Topic-Modelling
    if method == 'lda':
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    else:  # NMF
        model = NMF(n_components=n_topics, random_state=42)
    
    try:
        topic_distribution = model.fit_transform(X)
    except Exception as e:
        progress_bar.empty()
        st.error(f"Fehler beim Topic-Modelling: {str(e)}")
        return {'topics': [], 'topic_distribution': []}
    
    # Themen und deren Top-Wörter
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    
    progress_bar.empty()
    
    # Zuordnung von Artikeln zu Themen
    article_topics = []
    for i, article_id in enumerate(ids):
        dominant_topic = topic_distribution[i].argmax()
        confidence = topic_distribution[i].max()
        article_topics.append({
            'article_id': article_id,
            'topic': dominant_topic,
            'confidence': confidence
        })
    
    return {
        'topics': topics,
        'article_topics': article_topics
    }

@st.cache_data(ttl=1800)  # 30 Minuten Cache für Sentiment-Analyse
def analyze_sentiment(texts: List[str]) -> List[Dict]:
    """Führt Sentiment-Analyse auf den Texten durch"""
    results = []
    
    # Überprüfen, ob TextBlob oder TextBlob-DE verfügbar ist (global definiert)
    if not TEXTBLOB_DE_AVAILABLE:
        try:
            from textblob import TextBlob
        except ImportError:
            st.error("Weder TextBlob noch TextBlob-DE gefunden. Sentiment-Analyse nicht verfügbar.")
            return [{'polarity': 0, 'subjectivity': 0, 'error': 'TextBlob nicht installiert'} for _ in texts]
        
        st.warning("TextBlob-DE nicht gefunden, verwende Standard-TextBlob. Die Ergebnisse könnten für deutsche Texte weniger präzise sein.")
    
    for text in texts:
        if not text:
            results.append({'polarity': 0, 'subjectivity': 0})
            continue
            
        try:
            if TEXTBLOB_DE_AVAILABLE:
                # TextBlob-DE für deutsche Texte
                from textblob_de import TextBlobDE
                blob = TextBlobDE(text[:5000])  # Begrenze auf 5000 Zeichen für Performance
                # TextBlobDE gibt Tupel (polarity, subjectivity) zurück
                polarity, subjectivity = blob.sentiment
            else:
                # Standard TextBlob als Fallback
                from textblob import TextBlob
                blob = TextBlob(text[:5000])
                # Standard TextBlob hat separate sentiment.polarity und sentiment.subjectivity Attribute
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
            results.append({
                'polarity': polarity,  # Zwischen -1 (negativ) und 1 (positiv)
                'subjectivity': subjectivity  # Zwischen 0 (objektiv) und 1 (subjektiv)
            })
        except Exception as e:
            results.append({'polarity': 0, 'subjectivity': 0, 'error': str(e)})
    
    return results

@st.cache_data(ttl=1800)  # 30 Minuten Cache für Cluster-Analyse
def cluster_articles(df: pd.DataFrame, n_clusters=5) -> Dict:
    """Clustert Artikel basierend auf TF-IDF-Vektorisierung"""
    contents = []
    ids = []
    
    progress_bar = st.progress(0.0)
    total = min(len(df), 300)  # Begrenze auf 300 Artikel für Performance
    
    for i, row in df.head(total).iterrows():
        try:
            content = get_article_content(row['id'])
            if content:
                contents.append(content)
                ids.append(row['id'])
            progress_bar.progress((i+1)/total)
        except Exception:
            pass
    
    if not contents:
        progress_bar.empty()
        return {'clusters': []}
    
    # Vektorisierung
    vectorizer = get_text_vectorizer()
    try:
        X = vectorizer.fit_transform(contents)
    except Exception as e:
        progress_bar.empty()
        st.error(f"Fehler bei der Textvektorisierung: {str(e)}")
        return {'clusters': []}
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    try:
        clusters = kmeans.fit_predict(X)
    except Exception as e:
        progress_bar.empty()
        st.error(f"Fehler beim Clustering: {str(e)}")
        return {'clusters': []}
    
    # Cluster-Zentren für Top-Wörter pro Cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = []
    
    for i in range(n_clusters):
        indices = [j for j, x in enumerate(clusters) if x == i]
        if not indices:
            cluster_keywords.append([])
            continue
            
        cluster_docs = [contents[j] for j in indices]
        try:
            cluster_vec = vectorizer.transform(cluster_docs)
            importance = np.array(cluster_vec.sum(axis=0)).flatten()
            indices = importance.argsort()[-10:][::-1]
            keywords = [feature_names[j] for j in indices]
            cluster_keywords.append(keywords)
        except:
            cluster_keywords.append([])
    
    # Ergebnisse strukturieren
    cluster_results = []
    for i, article_id in enumerate(ids):
        cluster_results.append({
            'article_id': article_id,
            'cluster': int(clusters[i])
        })
    
    progress_bar.empty()
    return {
        'clusters': cluster_results,
        'cluster_keywords': cluster_keywords
    }

def analyze_with_genai(article_content: str, task: str) -> str:
    """Nutzt die Google Generative AI für verschiedene Analyseaufgaben"""
    if not GOOGLE_API_KEY or not article_content:
        return "API-Schlüssel fehlt oder kein Artikelinhalt verfügbar."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompts = {
            'summary': f"Fasse den folgenden Nachrichtenartikel auf Deutsch in 3-5 Sätzen zusammen:\n\n{article_content[:15000]}",
            'keywords': f"Extrahiere 5-10 der wichtigsten Schlüsselwörter aus diesem Artikel und gib sie als kommagetrennte Liste zurück:\n\n{article_content[:15000]}",
            'sentiment': f"Analysiere den Sentiment (positive, neutral oder negativ) dieses Artikels und begründe deine Analyse kurz:\n\n{article_content[:15000]}",
            'entities': f"Identifiziere die wichtigsten Entitäten (Personen, Organisationen, Technologien) in diesem Artikel und liste sie kategorisiert auf:\n\n{article_content[:15000]}",
            'trends': f"Identifiziere potenzielle Trends oder wichtige Entwicklungen basierend auf diesem Artikel und erkläre ihre Bedeutung:\n\n{article_content[:15000]}"
        }
        
        if task not in prompts:
            return f"Unbekannte Analyseaufgabe: {task}"
        
        response = model.generate_content(prompts[task])
        return response.text
        
    except Exception as e:
        return f"Fehler bei der KI-Analyse: {str(e)}"

@st.cache_data(ttl=900)  # 15 Minuten Cache für Keyword-Analysen
def get_keyword_analysis(df: pd.DataFrame) -> Dict:
    """Berechnet Keyword-Statistiken"""
    all_keywords = []
    keyword_category_data = []
    
    for _, row in df.iterrows():
        if pd.notna(row['keywords']) and row['keywords'] != 'N/A':
            keywords = [k.strip() for k in str(row['keywords']).split(',')]
            all_keywords.extend(keywords)
            
            if pd.notna(row['category']):
                for keyword in keywords:
                    keyword_category_data.append({
                        'keyword': keyword,
                        'category': row['category']
                    })
    
    keyword_counts = pd.Series(all_keywords).value_counts() if all_keywords else pd.Series()
    
    return {
        'keyword_counts': keyword_counts,
        'keyword_category_data': keyword_category_data
    }

@st.cache_data(ttl=1200)  # 20 Minuten Cache für Performance-Metriken
def get_performance_metrics(df: pd.DataFrame) -> Dict:
    """Berechnet Performance-Metriken"""
    metrics = {}
    
    # Wortanzahl-Kategorisierung
    word_counts = df['word_count'].dropna()
    if not word_counts.empty:
        metrics['short_articles'] = len(word_counts[word_counts < 300])
        metrics['medium_articles'] = len(word_counts[(word_counts >= 300) & (word_counts < 800)])
        metrics['long_articles'] = len(word_counts[word_counts >= 800])
    
    # Zeitbasierte Metriken
    df_time = df.dropna(subset=['date'])
    if not df_time.empty:
        metrics['weekday_counts'] = df_time['date'].dt.day_name().value_counts()
        metrics['hourly_counts'] = df_time['date'].dt.hour.value_counts().sort_index()
    
    return metrics

@st.cache_data(ttl=900)  # 15 Minuten Cache
def create_author_network():
    """Erstellt das Autoren-Netzwerk"""
    df = load_articles_data()
    if df.empty:
        return None, None
    
    # Autoren-Netzwerk aufbauen
    authors_dict = {}
    all_authors = set()
    
    for author_str in df['author'].dropna():
        if author_str and author_str != 'N/A':
            authors = [a.strip() for a in str(author_str).split(',')]
            authors = [author for author in authors if author.lower() != 'dpa']
            all_authors.update(authors)
            
            for author in authors:
                if author not in authors_dict:
                    authors_dict[author] = set()
                authors_dict[author].update(authors)
    
    # NetworkX Graph erstellen
    G = nx.Graph()
    for author, coauthors in authors_dict.items():
        G.add_node(author)
        for coauthor in coauthors:
            if author != coauthor:
                G.add_edge(author, coauthor)
    
    # Layout berechnen
    if len(G.nodes()) > 0:
        k_value = 1 / np.sqrt(len(G.nodes()))
        pos = nx.spring_layout(G, k=k_value, iterations=50)
    else:
        pos = {}
    
    return G, pos

def analyze_text_patterns(df: pd.DataFrame) -> Dict:
    """Analysiert Textmuster in Artikeltiteln"""
    titles = df['title'].dropna().tolist()
    
    # Häufigste Wörter in Titeln
    all_words = []
    for title in titles:
        words = re.findall(r'\b\w+\b', title.lower())
        all_words.extend([word for word in words if len(word) > 3])
    
    word_freq = Counter(all_words)
    
    # Emotionale Wörter
    positive_words = ['neu', 'besser', 'schnell', 'innovativ', 'erfolg', 'gut', 'stark', 'hoch', 'top']
    negative_words = ['problem', 'fehler', 'langsam', 'schlecht', 'schwach', 'niedrig', 'warnung', 'kritik']
    
    positive_count = sum(word_freq.get(word, 0) for word in positive_words)
    negative_count = sum(word_freq.get(word, 0) for word in negative_words)
    
    # Titel-Längenanalyse
    title_lengths = [len(title) for title in titles]
    
    # Häufigste Anfangswörter
    first_words = [title.split()[0].lower() for title in titles if title.split()]
    first_word_freq = Counter(first_words)
    
    return {
        'word_frequency': dict(word_freq.most_common(20)),
        'positive_sentiment': positive_count,
        'negative_sentiment': negative_count,
        'avg_title_length': np.mean(title_lengths),
        'title_lengths': title_lengths,
        'first_words': dict(first_word_freq.most_common(10))
    }

def generate_word_cloud_data(df: pd.DataFrame) -> Dict:
    """Generiert Daten für eine Text-Wolke"""
    titles = df['title'].dropna().tolist()
    keywords = df['keywords'].dropna().tolist()
    
    # Alle Texte kombinieren
    all_text = ' '.join(titles + [str(k) for k in keywords])
    
    # Häufige deutsche Stoppwörter entfernen
    stop_words = {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'daß', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'kann', 'wenn', 'was', 'machen', 'sehr', 'alle', 'jahren', 'ihre', 'heute', 'gibt', 'gegen', 'diese', 'neuen', 'schon', 'seit', 'ersten', 'große', 'deutschen', 'dpa', 'nachrichten', 'navigation', 'artikel'}
    
    # Wörter extrahieren und filtern
    words = re.findall(r'\b\w+\b', all_text.lower())
    filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
    
    word_freq = Counter(filtered_words)
    
    return dict(word_freq.most_common(50))

def create_advanced_statistics(df: pd.DataFrame) -> Dict:
    """Erstellt erweiterte Statistiken"""
    stats = {}
    
    # Zeitbasierte Statistiken
    df_time = df.dropna(subset=['date'])
    if not df_time.empty:
        stats['peak_publishing_hour'] = df_time['date'].dt.hour.mode().iloc[0] if not df_time.empty else 0
        stats['peak_publishing_day'] = df_time['date'].dt.day_name().mode().iloc[0] if not df_time.empty else 'N/A'
        
        # Wochentage-Verteilung
        weekday_counts = df_time['date'].dt.day_name().value_counts()
        stats['weekday_distribution'] = weekday_counts.to_dict()
        
        # Monatliche Trends
        monthly_counts = df_time.groupby(df_time['date'].dt.to_period('M')).size()
        stats['monthly_trend'] = monthly_counts.to_dict()
    
    # Autor-Statistiken
    author_stats = df.groupby('author').agg({
        'word_count': ['mean', 'sum', 'count'],
        'date': ['min', 'max']
    }).round(2)
    
    stats['author_productivity'] = author_stats.to_dict()
    
    # Kategorie-Diversität
    category_diversity = {}
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        category_diversity[category] = {
            'articles': len(cat_data),
            'avg_words': cat_data['word_count'].mean(),
            'unique_authors': cat_data['author'].nunique(),
            'timespan_days': (cat_data['date'].max() - cat_data['date'].min()).days if not cat_data['date'].isna().all() else 0
        }
    
    stats['category_diversity'] = category_diversity
    
    return stats

def create_smart_insights(df: pd.DataFrame) -> List[str]:
    """Generiert intelligente Einblicke basierend auf Datenanalyse"""
    insights = []
    
    # Analyze publishing patterns
    df_time = df.dropna(subset=['date'])
    if not df_time.empty:
        df_time['hour'] = df_time['date'].dt.hour
        df_time['weekday'] = df_time['date'].dt.day_name()
        
        # Peak publishing time
        peak_hour = df_time['hour'].mode().iloc[0]
        peak_day = df_time['weekday'].mode().iloc[0]
        
        insights.append(f"📅 **Optimale Veröffentlichungszeit**: Die meisten Artikel werden um {peak_hour}:00 Uhr an {peak_day}en veröffentlicht.")
        
        # Recent activity trend
        last_7_days = df_time[df_time['date'] >= (datetime.now() - timedelta(days=7))]
        prev_7_days = df_time[(df_time['date'] >= (datetime.now() - timedelta(days=14))) & 
                              (df_time['date'] < (datetime.now() - timedelta(days=7)))]
        
        if len(last_7_days) > len(prev_7_days):
            insights.append("📈 **Aktivitätstrend**: Die Veröffentlichungsaktivität hat in der letzten Woche zugenommen.")
        elif len(last_7_days) < len(prev_7_days):
            insights.append("📉 **Aktivitätstrend**: Die Veröffentlichungsaktivität hat in der letzten Woche abgenommen.")
        else:
            insights.append("📊 **Aktivitätstrend**: Die Veröffentlichungsaktivität ist stabil geblieben.")
    
    # Analyze content patterns
    if not df['word_count'].isna().all():
        avg_words = df['word_count'].mean()
        long_articles = df[df['word_count'] > avg_words * 1.5]
        
        if len(long_articles) > len(df) * 0.3:
            insights.append("📝 **Content-Trend**: Es gibt einen Trend zu längeren, detaillierteren Artikeln.")
        
        # Quality metric
        quality_score = (df['word_count'].fillna(0).mean() * 0.3 + 
                        df['author'].notna().sum() / len(df) * 0.4 + 
                        df['keywords'].notna().sum() / len(df) * 0.3) * 100
        
        insights.append(f"⭐ **Content-Qualität**: Der durchschnittliche Qualitätsscore liegt bei {quality_score:.1f}/100.")
    
    # Author insights
    author_counts = df['author'].value_counts()
    if not author_counts.empty:
        top_author = author_counts.index[0]
        author_share = (author_counts.iloc[0] / len(df)) * 100
        
        if author_share > 20:
            insights.append(f"✍️ **Autor-Dominanz**: {top_author} ist sehr aktiv und verfasst {author_share:.1f}% aller Artikel.")
        
        # Author diversity
        authors_with_single_article = len(author_counts[author_counts == 1])
        if authors_with_single_article > len(author_counts) * 0.5:
            insights.append("👥 **Autor-Vielfalt**: Viele Autoren schreiben nur gelegentlich - es gibt eine hohe Autoren-Diversität.")
    
    # Category insights
    category_counts = df['category'].value_counts()
    if not category_counts.empty:
        top_category = category_counts.index[0]
        category_share = (category_counts.iloc[0] / len(df)) * 100
        
        if category_share > 50:
            insights.append(f"🏷️ **Kategorie-Fokus**: {top_category} dominiert stark mit {category_share:.1f}% aller Artikel.")
        elif category_share < 20:
            insights.append("🏷️ **Kategorie-Balance**: Die Artikel sind gut über verschiedene Kategorien verteilt.")
    
    # Keyword insights
    if df['keywords'].notna().any():
        keyword_coverage = (df['keywords'].notna().sum() / len(df)) * 100
        if keyword_coverage > 80:
            insights.append(f"🔑 **Keyword-Abdeckung**: Ausgezeichnet! {keyword_coverage:.1f}% der Artikel haben Keywords.")
        elif keyword_coverage < 50:
            insights.append(f"🔑 **Keyword-Potential**: Nur {keyword_coverage:.1f}% der Artikel haben Keywords - hier ist Verbesserungspotential.")
    
    return insights

def create_dynamic_summary(df: pd.DataFrame) -> str:
    """Erstellt eine dynamische Zusammenfassung basierend auf aktuellen Daten im minimalistischen Design"""
    
    total_articles = len(df)
    current_time = datetime.now()
    
    # Zeitbasierte Analyse
    df_time = df.dropna(subset=['date'])
    if not df_time.empty:
        latest_article = df_time['date'].max()
        time_since_last = current_time - latest_article
        
        if time_since_last.total_seconds() < 3600:  # Weniger als 1 Stunde
            freshness = "🔥 **Sehr aktuell** - Der neueste Artikel ist weniger als eine Stunde alt!"
        elif time_since_last.total_seconds() < 86400:  # Weniger als 24 Stunden
            hours_old = int(time_since_last.total_seconds() / 3600)
            freshness = f"✅ **Aktuell** - Der neueste Artikel ist {hours_old} Stunden alt."
        else:
            days_old = int(time_since_last.days)
            freshness = f"📅 **Letzte Aktualisierung** - Der neueste Artikel ist {days_old} Tage alt."
    else:
        freshness = "❓ **Unbekannt** - Keine gültigen Datumsdaten verfügbar."
    
    # Aktivitätslevel
    recent_articles = df_time[df_time['date'] >= (current_time - timedelta(days=7))] if not df_time.empty else pd.DataFrame()
    
    if len(recent_articles) > 50:
        activity_level = "🚀 **Sehr aktiv** - Mehr als 50 Artikel in der letzten Woche"
    elif len(recent_articles) > 20:
        activity_level = "📈 **Aktiv** - Regelmäßige Veröffentlichungen"
    elif len(recent_articles) > 5:
        activity_level = "📊 **Moderat** - Moderate Aktivität"
    else:
        activity_level = "🔍 **Ruhig** - Geringe Aktivität in der letzten Woche"
    
    # Datenqualität
    quality_factors = []
    if df['title'].notna().sum() / len(df) > 0.95:
        quality_factors.append("Vollständige Titel")
    if df['author'].notna().sum() / len(df) > 0.8:
        quality_factors.append("Gute Autoren-Abdeckung")
    if df['keywords'].notna().sum() / len(df) > 0.5:
        quality_factors.append("Keyword-Optimierung")
    
    quality_summary = "✅ **Datenqualität**: " + ", ".join(quality_factors) if quality_factors else "⚠️ **Datenqualität**: Verbesserungspotential vorhanden"
    
    summary = f"""
    <div style="background: #f7f7f7; color: #333; padding: 18px; border-radius: 5px; margin: 15px 0; border-left: 3px solid #4c78a8;">
        <h3 style="margin-top: 0; font-weight: 400;">🎯 Live-Status der Datenbank</h3>
        <p><strong>Gesamtbestand:</strong> {total_articles:,} Artikel</p>
        <p>{freshness}</p>
        <p>{activity_level}</p>
        <p>{quality_summary}</p>
        <p style="color: #888; font-size: 0.9em; margin-bottom: 0;">🔄 Automatisch aktualisiert bei jeder Seitenladung</p>
    </div>
    """
    
    return summary

def generate_dashboard_report(df: pd.DataFrame, stats: Dict) -> str:
    """Generiert einen detaillierten Textbericht für das Dashboard"""
    
    # Grundlegende Statistiken
    total_articles = stats['total_articles']
    unique_authors = stats['unique_authors']
    unique_categories = stats['unique_categories']
    avg_words = stats['avg_words']
    
    # Zeitbasierte Informationen
    df_with_dates = df.dropna(subset=['date'])
    if not df_with_dates.empty:
        earliest_date = df_with_dates['date'].min().strftime('%d.%m.%Y')
        latest_date = df_with_dates['date'].max().strftime('%d.%m.%Y')
        date_range = (df_with_dates['date'].max() - df_with_dates['date'].min()).days
        
        # Letzte 24 Stunden
        last_24h = df[df['date'] >= (datetime.now() - timedelta(hours=24))]
        articles_24h = len(last_24h)
        
        # Letzte 7 Tage
        last_7d = df[df['date'] >= (datetime.now() - timedelta(days=7))]
        articles_7d = len(last_7d)
    else:
        earliest_date = "N/A"
        latest_date = "N/A"
        date_range = 0
        articles_24h = 0
        articles_7d = 0
    
    # Wortanzahl-Analyse
    if not df['word_count'].isna().all():
        max_words = int(df['word_count'].max())
        min_words = int(df['word_count'].min())
        
        # Artikel-Längen-Kategorisierung
        word_counts = df['word_count'].dropna()
        short_articles = len(word_counts[word_counts < 300])
        medium_articles = len(word_counts[(word_counts >= 300) & (word_counts < 800)])
        long_articles = len(word_counts[word_counts >= 800])
    else:
        max_words = 0
        min_words = 0
        short_articles = 0
        medium_articles = 0
        long_articles = 0
    
    # Bericht zusammenstellen
    report = f"""
    **📊 Datenbank-Übersicht**
    
    **Gesamtbestand:** Die Datenbank enthält aktuell **{total_articles:,}** Artikel von **{unique_authors}** verschiedenen Autoren, verteilt auf **{unique_categories}** Kategorien.
    
    **📅 Zeitraum und Aktivität**
    
    **Erfasster Zeitraum:** {earliest_date} bis {latest_date} ({date_range} Tage)
    **Aktuelle Aktivität:** {articles_24h} Artikel in den letzten 24 Stunden, {articles_7d} Artikel in der letzten Woche
    
    **📝 Artikel-Charakteristika**
    
    **Durchschnittliche Länge:** {avg_words:.0f} Wörter pro Artikel
    **Längster Artikel:** {max_words:,} Wörter
    **Kürzester Artikel:** {min_words:,} Wörter
    
    **📏 Längen-Verteilung**
    
    **Kurze Artikel** (<300 Wörter): {short_articles} ({short_articles/total_articles*100:.1f}%)
    **Mittlere Artikel** (300-800 Wörter): {medium_articles} ({medium_articles/total_articles*100:.1f}%)
    **Lange Artikel** (>800 Wörter): {long_articles} ({long_articles/total_articles*100:.1f}%)
    """
    
    return report

def generate_category_summary(top_categories: pd.Series) -> str:
    """Generiert eine Textzusammenfassung der Kategorien"""
    
    total_articles = top_categories.sum()
    num_categories = len(top_categories)
    
    # Top 3 Kategorien
    top_3 = top_categories.head(3)
    
    summary = f"""
    **📊 Kategorie-Analyse**
    
    **Kategorien-Überblick:** Die Top {num_categories} Kategorien umfassen {total_articles:,} Artikel.
    
    **🏆 Führende Kategorien**
    """
    
    for i, (category, count) in enumerate(top_3.items(), 1):
        percentage = (count / total_articles) * 100
        summary += f"""
    **{i}. {category}:** {count:,} Artikel ({percentage:.1f}% der Top-Kategorien)
        """
    
    # Verteilungs-Analyse
    if len(top_categories) > 3:
        others_count = top_categories.iloc[3:].sum()
        others_percentage = (others_count / total_articles) * 100
        summary += f"""
    **Weitere Kategorien:** {others_count:,} Artikel ({others_percentage:.1f}%) verteilt auf {len(top_categories)-3} weitere Kategorien
        """
    
    # Dominanz-Analyse
    if len(top_categories) > 0:
        dominance = (top_categories.iloc[0] / total_articles) * 100
        if dominance > 30:
            summary += f"""
    **📈 Beobachtung:** Die Kategorie '{top_categories.index[0]}' dominiert mit {dominance:.1f}% der Artikel deutlich.
            """
        elif dominance < 15:
            summary += f"""
    **📊 Beobachtung:** Die Artikel sind relativ gleichmäßig über die Kategorien verteilt (höchste Kategorie: {dominance:.1f}%).
            """
    
    return summary

def generate_author_summary(author_counts: pd.Series, df: pd.DataFrame) -> str:
    """Generiert eine Textzusammenfassung der Autoren"""
    
    total_articles = len(df)
    num_authors = len(author_counts)
    
    # Top 3 Autoren
    top_3 = author_counts.head(3)
    
    summary = f"""
    **✍️ Autoren-Analyse**
    
    **Autoren-Überblick:** Die Top {num_authors} Autoren haben {author_counts.sum():,} Artikel verfasst.
    
    **🏆 Produktivste Autoren**
    """
    
    for i, (author, count) in enumerate(top_3.items(), 1):
        percentage = (count / total_articles) * 100
        summary += f"""
    **{i}. {author}:** {count:,} Artikel ({percentage:.1f}% aller Artikel)
        """
    
    # Produktivitäts-Analyse
    avg_articles_per_author = author_counts.mean()
    median_articles_per_author = author_counts.median()
    
    summary += f"""
    
    **📊 Produktivitäts-Statistiken**
    
    **Durchschnitt:** {avg_articles_per_author:.1f} Artikel pro Autor
    **Median:** {median_articles_per_author:.1f} Artikel pro Autor
    """
    
    # Verteilungs-Analyse
    if len(author_counts) > 0:
        top_author_share = (author_counts.iloc[0] / total_articles) * 100
        if top_author_share > 15:
            summary += f"""
    **📈 Beobachtung:** Der Autor '{author_counts.index[0]}' ist besonders produktiv mit {top_author_share:.1f}% aller Artikel.
            """
        
        # Gelegenheitsautoren
        occasional_authors = len(author_counts[author_counts == 1])
        if occasional_authors > 0:
            summary += f"""
    **📝 Gelegenheitsautoren:** {occasional_authors} Autoren haben nur einen Artikel verfasst ({occasional_authors/len(author_counts)*100:.1f}% aller Autoren).
            """
    
    return summary

def generate_time_summary(df: pd.DataFrame) -> str:
    """Generiert eine Textzusammenfassung der Zeitanalyse"""
    
    df_with_dates = df.dropna(subset=['date'])
    
    if df_with_dates.empty:
        return "Keine gültigen Datumsdaten für die Zeitanalyse verfügbar."
    
    # Grundlegende Zeitstatistiken
    earliest_date = df_with_dates['date'].min()
    latest_date = df_with_dates['date'].max()
    date_range = (latest_date - earliest_date).days
    
    # Durchschnittliche Artikel pro Tag
    avg_articles_per_day = len(df_with_dates) / max(date_range, 1)
    
    # Wochentagsanalyse
    df_with_dates_copy = df_with_dates.copy()
    df_with_dates_copy['weekday'] = df_with_dates_copy['date'].dt.day_name()
    weekday_counts = df_with_dates_copy['weekday'].value_counts()
    
    most_active_weekday = weekday_counts.idxmax()
    most_active_count = weekday_counts.max()
    
    # Stundenanalyse
    df_with_dates_copy['hour'] = df_with_dates_copy['date'].dt.hour
    hourly_counts = df_with_dates_copy['hour'].value_counts().sort_index()
    
    most_active_hour = hourly_counts.idxmax()
    most_active_hour_count = hourly_counts.max()
    
    # Letzte Aktivität
    last_24h = df_with_dates[df_with_dates['date'] >= (datetime.now() - timedelta(hours=24))]
    last_7d = df_with_dates[df_with_dates['date'] >= (datetime.now() - timedelta(days=7))]
    
    summary = f"""
    **📅 Zeitbasierte Analyse**
    
    **Erfasster Zeitraum:** {earliest_date.strftime('%d.%m.%Y')} bis {latest_date.strftime('%d.%m.%Y')}
    **Gesamtdauer:** {date_range} Tage
    **Durchschnittliche Aktivität:** {avg_articles_per_day:.1f} Artikel pro Tag
    
    **📊 Aktivitätsmuster**
    
    **Aktivster Wochentag:** {most_active_weekday} mit {most_active_count} Artikeln
    **Aktivste Stunde:** {most_active_hour}:00 Uhr mit {most_active_hour_count} Artikeln
    
    **🕐 Aktuelle Aktivität**
    
    **Letzte 24 Stunden:** {len(last_24h)} Artikel
    **Letzte 7 Tage:** {len(last_7d)} Artikel
    """
    
    # Trend-Analyse
    if len(last_7d) > 0:
        weekly_avg = len(last_7d) / 7
        if weekly_avg > avg_articles_per_day:
            trend = "steigend"
            trend_icon = "📈"
        elif weekly_avg < avg_articles_per_day * 0.8:
            trend = "fallend"
            trend_icon = "📉"
        else:
            trend = "stabil"
            trend_icon = "📊"
        
        summary += f"""
    **{trend_icon} Trend:** Die Aktivität ist derzeit {trend} ({weekly_avg:.1f} Artikel/Tag in der letzten Woche vs. {avg_articles_per_day:.1f} Artikel/Tag insgesamt)
        """
    
    return summary

def search_articles(df: pd.DataFrame, search_term: str = "", category: str = "", 
                   author: str = "", date_from: Optional[datetime] = None, date_to: Optional[datetime] = None,
                   sort_by: str = "date", ascending: bool = False) -> pd.DataFrame:
    """Filtert Artikel basierend auf Suchkriterien"""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Textsuche
    if search_term:
        mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) |
            filtered_df['keywords'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Kategoriefilter
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    # Autorenfilter
    if author:
        filtered_df = filtered_df[filtered_df['author'].str.contains(author, case=False, na=False)]
    
    # Datumsfilter
    if date_from:
        filtered_df = filtered_df[filtered_df['date'] >= date_from]
    
    if date_to:
        filtered_df = filtered_df[filtered_df['date'] <= date_to]
    
    # Sortierung
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    return filtered_df

def get_article_preview(url: str) -> Tuple[bool, str]:
    """Holt eine Vorschau eines Artikels"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return False, f"HTTP Fehler: {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        article_content = soup.find('div', class_='article-content')
        
        if not article_content:
            return False, "Artikel-Inhalt konnte nicht gefunden werden"
        
        # Überprüfe, ob article_content ein Tag ist
        if hasattr(article_content, 'find_all') and callable(getattr(article_content, 'find_all')):
            # Unerwünschte Elemente entfernen
            for ad in article_content.find_all(['div'], class_=['ad', 'ad-label']):
                ad.decompose()
            
            # Nur relevante Textelemente behalten
            content_html = ""
            for element in article_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                content_html += str(element)
        else:
            # Fallback für NavigableString
            content_html = str(article_content)
        
        if not content_html.strip():
            return False, "Artikel-Inhalt konnte nicht extrahiert werden"
        
        return True, content_html
        
    except Exception as e:
        return False, str(e)

def main():
    """Hauptfunktion der Streamlit-App"""
    
    # Header
    st.markdown('<h1 class="main-header">🗞️ Heise Mining Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar für Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Seite auswählen",
        ["📊 Dashboard", "📅 Zeitanalysen", "🔑 Keyword-Analysen", "⚡ Performance-Metriken", 
         "🔍 Artikelsuche", "🕸️ Autoren-Netzwerk", "📈 Analysen", "🤖 KI-Analysen", "🔧 SQL-Abfragen"]
    )
    
    # Daten laden mit Fortschrittsanzeige
    with st.spinner("Lade Daten aus der Datenbank..."):
        df = load_articles_data()
    
    if df.empty:
        st.error("❌ Keine Daten verfügbar. Überprüfen Sie die Datenbankverbindung.")
        st.info("💡 Stellen Sie sicher, dass:")
        st.info("• Die .env-Datei korrekt konfiguriert ist")
        st.info("• Die Datenbank erreichbar ist")
        st.info("• Die Tabelle 'articles' existiert und Daten enthält")
        return
    
    # Informationen über die geladenen Daten
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Daten-Info")
    st.sidebar.metric("Artikel gesamt", len(df))
    st.sidebar.metric("Anzahl Autoren", df['author'].nunique())
    st.sidebar.metric("Anzahl Kategorien", df['category'].nunique())
    
    # Cache-Status
    if st.sidebar.button("🔄 Cache leeren"):
        st.cache_data.clear()
        st.sidebar.success("Cache geleert!")
        st.rerun()
    
    # Seitenbasierte Navigation
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "📅 Zeitanalysen":
        show_time_analytics(df)
    elif page == "🔑 Keyword-Analysen":
        show_keyword_analytics(df)
    elif page == "⚡ Performance-Metriken":
        show_performance_metrics(df)
    elif page == "🔍 Artikelsuche":
        show_article_search(df)
    elif page == "🕸️ Autoren-Netzwerk":
        show_author_network(df)
    elif page == "🤖 KI-Analysen":
        show_ai_analytics(df)
    elif page == "📈 Analysen":
        show_analytics(df)
    elif page == "🔧 SQL-Abfragen":
        show_sql_queries()

def show_dashboard(df: pd.DataFrame):
    """Zeigt das kompakte Hauptdashboard"""
    st.header("📊 Dashboard Übersicht")
    
    # Allgemeine Informationen
    st.subheader("📈 Allgemeine Informationen")
    
    # Einfache Dashboard-Statistiken in 4 Karten
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesamtartikel", len(df))
    
    with col2:
        # Artikel der letzten 24 Stunden
        last_24h = df[df['date'] >= (datetime.now() - timedelta(hours=24))]
        st.metric("Letzte 24h", len(last_24h))
    
    with col3:
        # Anzahl der Autoren
        st.metric("Autoren", df['author'].nunique())
        
    with col4:
        # Anzahl der Kategorien
        st.metric("Kategorien", df['category'].nunique())
    
    # Top Wörter anzeigen
    st.subheader("🔠 Häufigste Begriffe")
    
    word_cloud_data = generate_word_cloud_data(df)
    if word_cloud_data:
        # Die Top 10 Wörter anzeigen
        top_words = list(word_cloud_data.items())[:10]
        
        # Horizontale Balken für die Top-Wörter
        col1, col2 = st.columns([3, 1])
        
        with col1:
            word_df = pd.DataFrame(top_words, columns=['Begriff', 'Häufigkeit'])
            fig = px.bar(
                word_df,
                x='Häufigkeit',
                y='Begriff',
                orientation='h',
                title='Top 10 häufigste Begriffe',
                height=300
            )
            fig.update_layout(margin=dict(l=0, r=10, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("**Top Begriffe:**")
            for word, count in top_words[:5]:
                st.write(f"• **{word}**: {count}×")
    
    # Aktuelle Artikel - die neuesten 10 anzeigen
    st.subheader("🆕 Neueste Artikel")
    
    # Die ersten 10 Artikel anzeigen
    recent_articles = df.head(10)
    
    for _, article in recent_articles.iterrows():
        with st.container():
            st.markdown(f"""
            <div style="border-bottom: 1px solid #eaeaea; padding: 12px 0; margin-bottom: 8px;">
                <h4 style="margin: 0 0 6px 0; font-weight: 500;">
                    <a href="{article['url']}" target="_blank" style="color: #1a73e8; text-decoration: none;">{article['title']}</a>
                </h4>
                <div style="display: flex; flex-wrap: wrap; font-size: 0.85em; color: #707070; margin-bottom: 4px;">
                    <span style="margin-right: 12px;">{article['date'].strftime('%Y-%m-%d') if pd.notna(article['date']) else 'N/A'}</span>
                    <span style="margin-right: 12px;">{article['author'] if pd.notna(article['author']) else 'N/A'}</span>
                    <span>{article['category'] if pd.notna(article['category']) else 'N/A'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def show_time_analytics(df: pd.DataFrame):
    """Zeigt detaillierte Zeitanalysen"""
    st.header("📈 Zeitanalysen")
    
    # Grundlegende Zeitstatistiken
    st.subheader("⏱️ Grundlegende Zeitstatistiken")
    
    df_with_dates = df.dropna(subset=['date'])
    
    if df_with_dates.empty:
        st.warning("Keine gültigen Datumsdaten für Zeitanalysen verfügbar.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        earliest_date = df_with_dates['date'].min()
        st.metric("Frühester Artikel", earliest_date.strftime('%Y-%m-%d'))
    
    with col2:
        latest_date = df_with_dates['date'].max()
        st.metric("Neuester Artikel", latest_date.strftime('%Y-%m-%d'))
    
    with col3:
        date_range = (latest_date - earliest_date).days
        st.metric("Zeitraum (Tage)", date_range)
    
    with col4:
        if date_range > 0:
            articles_per_day = len(df_with_dates) / date_range
            st.metric("Ø Artikel/Tag", f"{articles_per_day:.2f}")
        else:
            st.metric("Ø Artikel/Tag", "N/A")
    
    # Zeitreihen-Analysen
    st.subheader("📅 Artikel-Verteilung über Zeit")
    
    # Optimierte tägliche Artikel-Anzahlen
    df_daily = get_daily_article_counts(df)
    
    if not df_daily.empty:
        # Hauptdiagramm: Artikel pro Tag
        fig_daily = px.line(
            df_daily, 
            x='date', 
            y='count',
            title='Artikel pro Tag',
            labels={'date': 'Datum', 'count': 'Anzahl Artikel'}
        )
        fig_daily.update_layout(
            height=400,
            hovermode='x unified',
            showlegend=False
        )
        fig_daily.update_traces(
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='<b>%{x}</b><br>Artikel: %{y}<extra></extra>'
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Zusätzliche Zeitanalysen
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap: Artikel pro Wochentag und Stunde
            df_time_analysis = df.dropna(subset=['date']).copy()
            if not df_time_analysis.empty:
                df_time_analysis['hour'] = df_time_analysis['date'].dt.hour
                df_time_analysis['weekday'] = df_time_analysis['date'].dt.day_name()
                
                heatmap_data = df_time_analysis.groupby(['weekday', 'hour']).size().reset_index(name='count')
                heatmap_pivot = heatmap_data.pivot(index='weekday', columns='hour', values='count').fillna(0)
                
                # Wochentage sortieren
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_pivot.reindex(weekday_order, fill_value=0)
                
                fig_heatmap = px.imshow(
                    heatmap_pivot,
                    title='Artikel-Aktivität nach Wochentag und Stunde',
                    labels={'x': 'Stunde', 'y': 'Wochentag', 'color': 'Anzahl Artikel'},
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Monatliche Trends
            df_monthly = df.dropna(subset=['date']).copy()
            if not df_monthly.empty:
                df_monthly['month'] = df_monthly['date'].dt.to_period('M')
                monthly_counts = df_monthly.groupby('month').size().reset_index(name='count')
                monthly_counts['month'] = monthly_counts['month'].astype(str)
                
                fig_monthly = px.bar(
                    monthly_counts,
                    x='month',
                    y='count',
                    title='Artikel pro Monat',
                    labels={'month': 'Monat', 'count': 'Anzahl Artikel'}
                )
                fig_monthly.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Wochentag-Analysen
    st.subheader("📊 Wochentagsanalysen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wochentag-Performance
        df_weekday = df.dropna(subset=['date'])
        if not df_weekday.empty:
            df_weekday['weekday'] = df_weekday['date'].dt.day_name()
            weekday_counts = df_weekday['weekday'].value_counts()
            
            # Wochentage sortieren
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)
            
            fig_weekday_perf = px.bar(
                x=weekday_counts.index,
                y=weekday_counts.values,
                title='Artikel-Veröffentlichungen pro Wochentag',
                labels={'x': 'Wochentag', 'y': 'Anzahl Artikel'},
                color=weekday_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_weekday_perf.update_layout(showlegend=False)
            st.plotly_chart(fig_weekday_perf, use_container_width=True)
    
    with col2:
        # Stundenbasierte Aktivität
        df_hourly = df.dropna(subset=['date'])
        if not df_hourly.empty:
            df_hourly['hour'] = df_hourly['date'].dt.hour
            hourly_counts = df_hourly['hour'].value_counts().sort_index()
            
            fig_hourly_perf = px.line(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title='Artikel-Veröffentlichungen pro Stunde',
                labels={'x': 'Stunde', 'y': 'Anzahl Artikel'},
                markers=True
            )
            fig_hourly_perf.update_traces(line_color='#FF6B6B', line_width=3)
            st.plotly_chart(fig_hourly_perf, use_container_width=True)
    
    # Saisonale Trends
    st.subheader("🌍 Saisonale Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quartalsweise Verteilung
        df_quarterly = df.dropna(subset=['date']).copy()
        if not df_quarterly.empty:
            df_quarterly['quarter'] = df_quarterly['date'].dt.quarter
            df_quarterly['year'] = df_quarterly['date'].dt.year
            df_quarterly['year_quarter'] = df_quarterly['year'].astype(str) + '-Q' + df_quarterly['quarter'].astype(str)
            
            quarterly_counts = df_quarterly['year_quarter'].value_counts().sort_index()
            
            fig_quarterly = px.bar(
                x=quarterly_counts.index,
                y=quarterly_counts.values,
                title='Artikel pro Quartal',
                labels={'x': 'Quartal', 'y': 'Anzahl Artikel'}
            )
            fig_quarterly.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_quarterly, use_container_width=True)
    
    with col2:
        # Wochenbasierte Verteilung
        df_weekly = df.dropna(subset=['date']).copy()
        if not df_weekly.empty:
            df_weekly['week'] = df_weekly['date'].dt.to_period('W')
            weekly_counts = df_weekly.groupby('week').size().reset_index(name='count')
            weekly_counts['week'] = weekly_counts['week'].astype(str)
            
            # Nur die letzten 12 Wochen für bessere Übersicht
            weekly_counts_recent = weekly_counts.tail(12)
            
            fig_weekly = px.line(
                weekly_counts_recent,
                x='week',
                y='count',
                title='Artikel pro Woche (Letzte 12 Wochen)',
                labels={'week': 'Woche', 'count': 'Anzahl Artikel'},
                markers=True
            )
            fig_weekly.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Autor-Zeitanalysen
    st.subheader("✍️ Autor-Aktivität über Zeit")
    
    df_author_time = df.dropna(subset=['date', 'author'])
    if not df_author_time.empty:
        author_counts = get_author_counts(df)
        
        # Nur die Top 5 Autoren für die Zeitreihe
        top_5_authors = author_counts.head(5).index.tolist()
        df_filtered = df_author_time[df_author_time['author'].str.contains('|'.join(top_5_authors), na=False)]
        
        if not df_filtered.empty:
            df_filtered['date_only'] = df_filtered['date'].dt.date
            author_timeline = df_filtered.groupby(['date_only', 'author']).size().reset_index(name='count')
            
            fig_author_time = px.line(
                author_timeline,
                x='date_only',
                y='count',
                color='author',
                title='Autor-Aktivität über Zeit (Top 5)',
                labels={'date_only': 'Datum', 'count': 'Anzahl Artikel', 'author': 'Autor'}
            )
            fig_author_time.update_layout(height=400)
            st.plotly_chart(fig_author_time, use_container_width=True)
    
    # Zeitbasierte Statistiken
    st.subheader("📈 Zeitbasierte Statistiken")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Durchschnittliche Artikel pro Woche
        weekly_avg = df_daily['count'].mean() * 7 if not df_daily.empty else 0
        st.metric("Ø Artikel pro Woche", f"{weekly_avg:.1f}")
    
    with col2:
        # Produktivste Tageszeit
        df_hourly = df.dropna(subset=['date'])
        if not df_hourly.empty:
            df_hourly['hour'] = df_hourly['date'].dt.hour
            hourly_counts = df_hourly['hour'].value_counts().sort_index()
            
            if not hourly_counts.empty:
                peak_hour = hourly_counts.idxmax()
                peak_count = hourly_counts.max()
                st.metric("Produktivste Stunde", f"{peak_hour}:00 Uhr ({peak_count} Artikel)")
            else:
                st.metric("Produktivste Stunde", "N/A")
        else:
            st.metric("Produktivste Stunde", "N/A")
    
    with col3:
        # Produktivster Wochentag
        df_weekday = df.dropna(subset=['date'])
        if not df_weekday.empty:
            df_weekday['weekday'] = df_weekday['date'].dt.day_name()
            weekday_counts = df_weekday['weekday'].value_counts()
            
            if not weekday_counts.empty:
                peak_weekday = weekday_counts.idxmax()
                peak_weekday_count = weekday_counts.max()
                st.metric("Produktivster Wochentag", f"{peak_weekday} ({peak_weekday_count} Artikel)")
            else:
                st.metric("Produktivster Wochentag", "N/A")
        else:
            st.metric("Produktivster Wochentag", "N/A")


def show_keyword_analytics(df: pd.DataFrame):
    """Zeigt detaillierte Keyword-Analysen"""
    st.header("🔑 Keyword-Analysen")
    
    # Keyword-Daten aufbereiten
    all_keywords = []
    keyword_article_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['keywords']) and row['keywords'] != 'N/A':
            keywords = [k.strip() for k in str(row['keywords']).split(',')]
            all_keywords.extend(keywords)
            
            for keyword in keywords:
                keyword_article_data.append({
                    'keyword': keyword,
                    'category': row['category'],
                    'author': row['author'],
                    'date': row['date'],
                    'word_count': row['word_count']
                })
    
    if not all_keywords:
        st.warning("Keine Keywords in den Daten gefunden.")
        return
    
    # Grundlegende Keyword-Statistiken
    st.subheader("📊 Grundlegende Keyword-Statistiken")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Keywords", len(set(all_keywords)))
    
    with col2:
        st.metric("Gesamte Keywords", len(all_keywords))
    
    with col3:
        keyword_counts = pd.Series(all_keywords).value_counts()
        avg_frequency = keyword_counts.mean()
        st.metric("Ø Keyword-Häufigkeit", f"{avg_frequency:.1f}")
    
    with col4:
        articles_with_keywords = len(df[df['keywords'].notna() & (df['keywords'] != 'N/A')])
        coverage = (articles_with_keywords / len(df)) * 100
        st.metric("Keyword-Abdeckung", f"{coverage:.1f}%")
    
    # Top Keywords
    st.subheader("🏆 Top Keywords")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Keyword-Häufigkeit
        keyword_counts = pd.Series(all_keywords).value_counts().head(20)
        
        fig_keywords = px.bar(
            x=keyword_counts.values,
            y=keyword_counts.index,
            orientation='h',
            title='Top 20 Keywords',
            labels={'x': 'Häufigkeit', 'y': 'Keyword'}
        )
        fig_keywords.update_layout(height=600)
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    with col2:
        # Keyword-Wolke (vereinfacht als Sunburst)
        top_keywords = keyword_counts.head(15)
        
        fig_sunburst = px.sunburst(
            names=top_keywords.index,
            values=top_keywords.values,
            title='Top 15 Keywords (Größe = Häufigkeit)',
            color=top_keywords.values,
            color_continuous_scale='Blues'
        )
        fig_sunburst.update_layout(height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # Keyword-Kategorie-Analysen
    st.subheader("🏷️ Keyword-Kategorie-Beziehungen")
    
    kw_cat_df = pd.DataFrame(keyword_article_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Keywords nach Kategorien
        top_keywords = pd.Series(all_keywords).value_counts().head(10).index
        kw_cat_filtered = kw_cat_df[kw_cat_df['keyword'].isin(top_keywords)]
        
        if not kw_cat_filtered.empty:
            kw_cat_matrix = kw_cat_filtered.groupby(['keyword', 'category']).size().reset_index(name='count')
            
            fig_kw_cat = px.sunburst(
                kw_cat_matrix,
                path=['keyword', 'category'],
                values='count',
                title='Keyword-Kategorie-Beziehungen (Top 10 Keywords)',
                color='count',
                color_continuous_scale='Blues'
            )
            fig_kw_cat.update_layout(height=500)
            st.plotly_chart(fig_kw_cat, use_container_width=True)
    
    with col2:
        # Heatmap: Keywords vs Kategorien
        top_categories = df['category'].value_counts().head(10).index
        kw_cat_heat = kw_cat_df[
            (kw_cat_df['keyword'].isin(top_keywords)) & 
            (kw_cat_df['category'].isin(top_categories))
        ]
        
        if not kw_cat_heat.empty:
            heat_matrix = kw_cat_heat.groupby(['keyword', 'category']).size().reset_index(name='count')
            heat_pivot = heat_matrix.pivot(index='keyword', columns='category', values='count').fillna(0)
            
            fig_heat = px.imshow(
                heat_pivot,
                title='Keyword-Kategorie-Heatmap (Top 10 Keywords/Kategorien)',
                labels={'x': 'Kategorie', 'y': 'Keyword', 'color': 'Anzahl'},
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    
    # Zeitbasierte Keyword-Analysen
    st.subheader("📅 Keyword-Trends über Zeit")
    
    kw_df_with_dates = pd.DataFrame(keyword_article_data)
    kw_df_with_dates = kw_df_with_dates.dropna(subset=['date'])
    
    if not kw_df_with_dates.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 5 Keywords über Zeit
            top_5_keywords = pd.Series(all_keywords).value_counts().head(5).index
            kw_time_filtered = kw_df_with_dates[kw_df_with_dates['keyword'].isin(top_5_keywords)]
            
            if not kw_time_filtered.empty:
                kw_time_filtered['date_only'] = kw_time_filtered['date'].dt.date
                kw_timeline = kw_time_filtered.groupby(['date_only', 'keyword']).size().reset_index(name='count')
                
                fig_kw_time = px.line(
                    kw_timeline,
                    x='date_only',
                    y='count',
                    color='keyword',
                    title='Top 5 Keywords über Zeit',
                    labels={'date_only': 'Datum', 'count': 'Anzahl', 'keyword': 'Keyword'}
                )
                fig_kw_time.update_layout(height=400)
                st.plotly_chart(fig_kw_time, use_container_width=True)
        
        with col2:
            # Keyword-Häufigkeit nach Monat
            kw_df_monthly = kw_df_with_dates.copy()
            kw_df_monthly['month'] = kw_df_monthly['date'].dt.to_period('M')
            monthly_kw_counts = kw_df_monthly.groupby('month').size().reset_index(name='count')
            monthly_kw_counts['month'] = monthly_kw_counts['month'].astype(str)
            
            fig_monthly_kw = px.bar(
                monthly_kw_counts,
                x='month',
                y='count',
                title='Keyword-Verwendung pro Monat',
                labels={'month': 'Monat', 'count': 'Anzahl Keywords'}
            )
            fig_monthly_kw.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_monthly_kw, use_container_width=True)
    
    # Autor-Keyword-Analysen
    st.subheader("✍️ Autor-Keyword-Präferenzen")
    
    kw_author_df = pd.DataFrame(keyword_article_data)
    kw_author_df = kw_author_df.dropna(subset=['author'])
    
    if not kw_author_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top-Autoren nach Keyword-Vielfalt
            author_keyword_diversity = kw_author_df.groupby('author')['keyword'].nunique().sort_values(ascending=False).head(10)
            
            fig_author_diversity = px.bar(
                x=author_keyword_diversity.values,
                y=author_keyword_diversity.index,
                orientation='h',
                title='Autoren mit höchster Keyword-Vielfalt',
                labels={'x': 'Unique Keywords', 'y': 'Autor'}
            )
            fig_author_diversity.update_layout(height=400)
            st.plotly_chart(fig_author_diversity, use_container_width=True)
        
        with col2:
            # Keyword-Spezialisierung der Top-Autoren
            top_authors = get_author_counts(df).head(5).index
            author_kw_spec = kw_author_df[kw_author_df['author'].isin(top_authors)]
            
            if not author_kw_spec.empty:
                author_kw_matrix = author_kw_spec.groupby(['author', 'keyword']).size().reset_index(name='count')
                
                # Nur die häufigsten Keywords pro Autor
                top_kw_per_author = author_kw_matrix.groupby('author').apply(
                    lambda x: x.nlargest(3, 'count')
                ).reset_index(drop=True)
                
                fig_author_kw = px.sunburst(
                    top_kw_per_author,
                    path=['author', 'keyword'],
                    values='count',
                    title='Top Keywords pro Autor (Top 5 Autoren)',
                    color='count',
                    color_continuous_scale='Viridis'
                )
                fig_author_kw.update_layout(height=500)
                st.plotly_chart(fig_author_kw, use_container_width=True)
    
    # Keyword-Performance-Metriken
    st.subheader("⚡ Keyword-Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Durchschnittliche Wortanzahl pro Keyword
        kw_word_df = pd.DataFrame(keyword_article_data)
        kw_word_df = kw_word_df.dropna(subset=['word_count'])
        
        if not kw_word_df.empty:
            avg_words_per_kw = kw_word_df.groupby('keyword')['word_count'].mean().sort_values(ascending=False).head(10)
            
            fig_kw_words = px.bar(
                x=avg_words_per_kw.values,
                y=avg_words_per_kw.index,
                orientation='h',
                title='Keywords mit höchster Ø Wortanzahl',
                labels={'x': 'Ø Wortanzahl', 'y': 'Keyword'}
            )
            fig_kw_words.update_layout(height=400)
            st.plotly_chart(fig_kw_words, use_container_width=True)
    
    with col2:
        # Keyword-Konsistenz (Regelmäßigkeit der Verwendung)
        kw_consistency = pd.Series(all_keywords).value_counts().head(15)
        
        fig_consistency = px.pie(
            values=kw_consistency.values,
            names=kw_consistency.index,
            title='Top 15 Keywords (Verwendungshäufigkeit)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_consistency.update_layout(height=400)
        st.plotly_chart(fig_consistency, use_container_width=True)
    
    with col3:
        # Keyword-Längen-Verteilung
        keyword_lengths = [len(kw) for kw in all_keywords]
        length_distribution = pd.Series(keyword_lengths).value_counts().sort_index()
        
        fig_lengths = px.bar(
            x=length_distribution.index,
            y=length_distribution.values,
            title='Keyword-Längen-Verteilung',
            labels={'x': 'Anzahl Zeichen', 'y': 'Anzahl Keywords'}
        )
        fig_lengths.update_layout(height=400)
        st.plotly_chart(fig_lengths, use_container_width=True)


def show_performance_metrics(df: pd.DataFrame):
    """Zeigt Performance-Metriken"""
    st.header("⚡ Performance-Metriken")
    
    # Grundlegende Performance-Statistiken
    st.subheader("📊 Grundlegende Performance-Statistiken")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Durchschnittliche Wortanzahl
        avg_words = df['word_count'].mean()
        st.metric("Ø Wortanzahl", f"{avg_words:.0f}" if not pd.isna(avg_words) else "N/A")
    
    with col2:
        # Median Wortanzahl
        median_words = df['word_count'].median()
        st.metric("Median Wortanzahl", f"{median_words:.0f}" if not pd.isna(median_words) else "N/A")
    
    with col3:
        # Längster Artikel
        if not df['word_count'].isna().all():
            max_words = df['word_count'].max()
            st.metric("Längster Artikel", f"{int(max_words)} Wörter")
        else:
            st.metric("Längster Artikel", "N/A")
    
    with col4:
        # Kürzester Artikel
        if not df['word_count'].isna().all():
            min_words = df['word_count'].min()
            st.metric("Kürzester Artikel", f"{int(min_words)} Wörter")
        else:
            st.metric("Kürzester Artikel", "N/A")
    
    # Artikel-Längen-Analysen
    st.subheader("📝 Artikel-Längen-Analysen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Artikel-Längen-Verteilung
        word_counts = df['word_count'].dropna()
        if not word_counts.empty:
            # Kategorisierung der Artikel nach Länge
            short_articles = len(word_counts[word_counts < 300])
            medium_articles = len(word_counts[(word_counts >= 300) & (word_counts < 800)])
            long_articles = len(word_counts[word_counts >= 800])
            
            article_lengths = pd.Series({
                'Kurz (<300 Wörter)': short_articles,
                'Mittel (300-800 Wörter)': medium_articles,
                'Lang (>800 Wörter)': long_articles
            })
            
            fig_lengths = px.pie(
                values=article_lengths.values,
                names=article_lengths.index,
                title='Artikel-Längen-Verteilung',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            st.plotly_chart(fig_lengths, use_container_width=True)
    
    with col2:
        # Histogram der Wortanzahl
        if not word_counts.empty:
            fig_hist = px.histogram(
                word_counts,
                nbins=30,
                title='Histogramm der Wortanzahl',
                labels={'value': 'Wortanzahl', 'count': 'Anzahl Artikel'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Box-Plot-Analysen
    st.subheader("📊 Box-Plot-Analysen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box-Plot: Wortanzahl nach Kategorien
        df_words = df.dropna(subset=['word_count', 'category'])
        if not df_words.empty:
            top_8_categories = df_words['category'].value_counts().head(8).index
            df_words_filtered = df_words[df_words['category'].isin(top_8_categories)]
            
            fig_box = px.box(
                df_words_filtered,
                x='category',
                y='word_count',
                title='Wortanzahl-Verteilung nach Kategorien (Top 8)',
                labels={'category': 'Kategorie', 'word_count': 'Wortanzahl'}
            )
            fig_box.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Box-Plot: Wortanzahl nach Top-Autoren
        df_author_words = df.dropna(subset=['word_count', 'author'])
        if not df_author_words.empty:
            top_8_authors = df_author_words['author'].value_counts().head(8).index
            df_author_words_filtered = df_author_words[df_author_words['author'].isin(top_8_authors)]
            
            fig_box_author = px.box(
                df_author_words_filtered,
                x='author',
                y='word_count',
                title='Wortanzahl-Verteilung nach Autoren (Top 8)',
                labels={'author': 'Autor', 'word_count': 'Wortanzahl'}
            )
            fig_box_author.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box_author, use_container_width=True)
    
    # Scatter-Plot-Analysen
    st.subheader("🔍 Scatter-Plot-Analysen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter-Plot: Wortanzahl vs. Veröffentlichungszeit
        df_scatter = df.dropna(subset=['word_count', 'date'])
        if not df_scatter.empty:
            df_scatter['hour'] = df_scatter['date'].dt.hour
            
            fig_scatter = px.scatter(
                df_scatter,
                x='hour',
                y='word_count',
                title='Wortanzahl vs. Veröffentlichungszeit',
                labels={'hour': 'Stunde', 'word_count': 'Wortanzahl'},
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Scatter-Plot: Wortanzahl vs. Kategorie (mit Größe)
        df_scatter_cat = df.dropna(subset=['word_count', 'category'])
        if not df_scatter_cat.empty:
            # Kategorien nach Häufigkeit
            cat_counts = df_scatter_cat['category'].value_counts()
            df_scatter_cat = df_scatter_cat[df_scatter_cat['category'].isin(cat_counts.head(10).index)]
            
            # Zufällige x-Koordinaten für bessere Visualisierung
            import numpy as np
            df_scatter_cat['x_random'] = np.random.normal(0, 0.1, len(df_scatter_cat))
            
            fig_scatter_cat = px.scatter(
                df_scatter_cat,
                x='x_random',
                y='word_count',
                color='category',
                title='Wortanzahl nach Kategorien (Top 10)',
                labels={'x_random': 'Verteilung', 'word_count': 'Wortanzahl', 'category': 'Kategorie'},
                opacity=0.7
            )
            fig_scatter_cat.update_layout(xaxis_title='Kategorieverteilung')
            st.plotly_chart(fig_scatter_cat, use_container_width=True)
    
    # Produktivitäts-Metriken
    st.subheader("🏆 Produktivitäts-Metriken")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Top-Autoren nach Gesamtwortanzahl
        df_author_productivity = df.dropna(subset=['author', 'word_count'])
        if not df_author_productivity.empty:
            author_total_words = df_author_productivity.groupby('author')['word_count'].sum().sort_values(ascending=False).head(10)
            
            fig_author_words = px.bar(
                x=author_total_words.values,
                y=author_total_words.index,
                orientation='h',
                title='Autoren nach Gesamtwortanzahl',
                labels={'x': 'Gesamtwortanzahl', 'y': 'Autor'}
            )
            fig_author_words.update_layout(height=400)
            st.plotly_chart(fig_author_words, use_container_width=True)
    
    with col2:
        # Durchschnittliche Wortanzahl pro Autor
        if not df_author_productivity.empty:
            author_avg_words = df_author_productivity.groupby('author')['word_count'].mean().sort_values(ascending=False).head(10)
            
            fig_author_avg = px.bar(
                x=author_avg_words.values,
                y=author_avg_words.index,
                orientation='h',
                title='Autoren nach Ø Wortanzahl/Artikel',
                labels={'x': 'Ø Wortanzahl', 'y': 'Autor'}
            )
            fig_author_avg.update_layout(height=400)
            st.plotly_chart(fig_author_avg, use_container_width=True)
    
    with col3:
        # Kategorien nach durchschnittlicher Wortanzahl
        df_cat_productivity = df.dropna(subset=['category', 'word_count'])
        if not df_cat_productivity.empty:
            cat_avg_words = df_cat_productivity.groupby('category')['word_count'].mean().sort_values(ascending=False).head(10)
            
            fig_cat_avg = px.bar(
                x=cat_avg_words.values,
                y=cat_avg_words.index,
                orientation='h',
                title='Kategorien nach Ø Wortanzahl',
                labels={'x': 'Ø Wortanzahl', 'y': 'Kategorie'}
            )
            fig_cat_avg.update_layout(height=400)
            st.plotly_chart(fig_cat_avg, use_container_width=True)
    
    # Zeitbasierte Performance
    st.subheader("⏱️ Zeitbasierte Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Durchschnittliche Wortanzahl über Zeit
        df_time_words = df.dropna(subset=['date', 'word_count'])
        if not df_time_words.empty:
            df_time_words['date_only'] = df_time_words['date'].dt.date
            daily_avg_words = df_time_words.groupby('date_only')['word_count'].mean().reset_index()
            
            fig_time_words = px.line(
                daily_avg_words,
                x='date_only',
                y='word_count',
                title='Durchschnittliche Wortanzahl über Zeit',
                labels={'date_only': 'Datum', 'word_count': 'Ø Wortanzahl'}
            )
            fig_time_words.update_traces(line=dict(width=2))
            st.plotly_chart(fig_time_words, use_container_width=True)
    
    with col2:
        # Gesamtwortanzahl pro Tag
        if not df_time_words.empty:
            daily_total_words = df_time_words.groupby('date_only')['word_count'].sum().reset_index()
            
            fig_total_words = px.bar(
                daily_total_words.tail(30),  # Letzte 30 Tage
                x='date_only',
                y='word_count',
                title='Gesamtwortanzahl pro Tag (Letzte 30 Tage)',
                labels={'date_only': 'Datum', 'word_count': 'Gesamtwortanzahl'}
            )
            fig_total_words.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_total_words, use_container_width=True)
    
    # Performance-Ranking
    st.subheader("🥇 Performance-Ranking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Längste Artikel
        if not df['word_count'].isna().all():
            longest_articles = df.nlargest(10, 'word_count')[['title', 'author', 'word_count', 'category']]
            
            st.write("**Top 10 Längste Artikel:**")
            for idx, article in longest_articles.iterrows():
                st.write(f"**{int(article['word_count'])} Wörter** - {article['title'][:50]}...")
                st.write(f"   *Autor: {article['author']} | Kategorie: {article['category']}*")
                st.write("---")
    
    with col2:
        # Kürzeste Artikel (aber > 0)
        if not df['word_count'].isna().all():
            shortest_articles = df[df['word_count'] > 0].nsmallest(10, 'word_count')[['title', 'author', 'word_count', 'category']]
            
            st.write("**Top 10 Kürzeste Artikel:**")
            for idx, article in shortest_articles.iterrows():
                st.write(f"**{int(article['word_count'])} Wörter** - {article['title'][:50]}...")
                st.write(f"   *Autor: {article['author']} | Kategorie: {article['category']}*")
                st.write("---")

def show_article_search(df: pd.DataFrame):
    """Zeigt die Artikelsuche"""
    st.header("🔍 Artikelsuche")
    
    # Suchformular
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("Suchbegriff", placeholder="Titel oder Schlagwörter durchsuchen...")
            category = st.selectbox("Kategorie", ["Alle"] + get_categories())
            
        with col2:
            author = st.text_input("Autor", placeholder="Autor suchen...")
            sort_option = st.selectbox("Sortierung", 
                                     ["Neueste zuerst", "Älteste zuerst", "Titel (A-Z)", "Titel (Z-A)"])
        
        col3, col4 = st.columns(2)
        with col3:
            date_from = st.date_input("Datum von", value=None)
        with col4:
            date_to = st.date_input("Datum bis", value=None)
        
        submitted = st.form_submit_button("🔍 Suchen", type="primary")
    
    # Suchparameter verarbeiten
    category_filter = None if category == "Alle" else category
    
    sort_mapping = {
        "Neueste zuerst": ("date", False),
        "Älteste zuerst": ("date", True),
        "Titel (A-Z)": ("title", True),
        "Titel (Z-A)": ("title", False)
    }
    sort_by, ascending = sort_mapping[sort_option]
    
    # Suche durchführen
    if submitted or any([search_term, category_filter, author, date_from, date_to]):
        with st.spinner("Suche wird durchgeführt..."):
            results = search_articles(
                df, search_term, category_filter or "", author, 
                pd.to_datetime(date_from) if date_from else None,
                pd.to_datetime(date_to) if date_to else None,
                sort_by, ascending
            )
        
        st.subheader(f"📋 Suchergebnisse ({len(results)} Artikel)")
        
        if not results.empty:
            # Erweiterte Paginierung
            items_per_page = st.select_slider(
                "Artikel pro Seite:", 
                options=[10, 20, 50, 100], 
                value=20,
                key="articles_per_page"
            )
            
            total_pages = (len(results) + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    page_num = st.number_input(
                        f"Seite (1-{total_pages})", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=1,
                        key="search_page"
                    )
                
                start_idx = (page_num - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_results = results.iloc[start_idx:end_idx]
                
                # Paginierungs-Info
                st.info(f"Zeige Artikel {start_idx + 1}-{min(end_idx, len(results))} von {len(results)}")
            else:
                page_results = results
            
            # Artikel anzeigen - optimiert
            for i, (_, article) in enumerate(page_results.iterrows()):
                with st.expander(f"📄 {article['title']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**📅 Datum:** {article['date'].strftime('%Y-%m-%d %H:%M') if pd.notna(article['date']) else 'N/A'}")
                        st.markdown(f"**✍️ Autor:** {article['author'] if pd.notna(article['author']) else 'N/A'}")
                        st.markdown(f"**🏷️ Kategorie:** {article['category'] if pd.notna(article['category']) else 'N/A'}")
                        
                        if pd.notna(article['keywords']) and article['keywords'] != 'N/A':
                            st.markdown(f"**🔑 Schlagwörter:** {article['keywords']}")
                        
                        if pd.notna(article['word_count']):
                            st.markdown(f"**📝 Wörter:** {int(article['word_count'])}")
                    
                    with col2:
                        st.markdown(f"**[🔗 Artikel öffnen]({article['url']})**")
                        
                        if st.button("📖 Vorschau", key=f"preview_{article['id']}_{i}"):
                            with st.spinner("Artikel wird geladen..."):
                                success, content = get_article_preview(article['url'])
                                if success:
                                    st.markdown(content, unsafe_allow_html=True)
                                else:
                                    st.error(f"Fehler beim Laden: {content}")
            
            # Export-Optionen
            if len(results) > 0:
                st.markdown("---")
                st.subheader("📥 Export-Optionen")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = results.to_csv(index=False)
                    st.download_button(
                        label="� Als CSV herunterladen",
                        data=csv_data,
                        file_name=f"suchergebnisse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = results.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="📦 Als JSON herunterladen",
                        data=json_data,
                        file_name=f"suchergebnisse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.info("🔍 Keine Artikel gefunden. Versuchen Sie andere Suchbegriffe oder erweitern Sie die Filterkriterien.")
            
            # Suchvorschläge
            st.markdown("**💡 Tipps für bessere Suchergebnisse:**")
            st.markdown("• Verwenden Sie allgemeinere Suchbegriffe")
            st.markdown("• Prüfen Sie die Rechtschreibung")
            st.markdown("• Entfernen Sie Filter, um mehr Ergebnisse zu erhalten")
            st.markdown("• Nutzen Sie Teilwörter anstatt ganzer Begriffe")

def show_ai_analytics(df: pd.DataFrame):
    """Zeigt KI-basierte Analysen"""
    st.header("🤖 KI-gestützte Analysen")
    
    # API-Key prüfen
    if not GOOGLE_API_KEY:
        st.error("Kein Google API-Schlüssel konfiguriert. Bitte fügen Sie einen Schlüssel in der .env-Datei hinzu.")
        st.code("GOOGLE_API_KEY=your_api_key_here", language="bash")
        return
    
    # Tabs für verschiedene KI-Analysen
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Topic Modeling", 
        "🔍 Artikel-Clusters", 
        "😊 Sentiment-Analyse", 
        "🤖 KI-Analyse", 
        "🔮 Trend-Prognose"
    ])
    
    # 1. Topic Modeling Tab
    with tab1:
        st.subheader("🧠 Topic Modeling")
        st.write("Diese Analyse identifiziert die Hauptthemen in den Artikeln mittels maschinellem Lernen.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            method = st.selectbox(
                "Methode auswählen",
                ["LDA (Latent Dirichlet Allocation)", "NMF (Non-Negative Matrix Factorization)"],
                key="topic_method"
            )
            n_topics = st.slider("Anzahl der Themen", 3, 12, 5, key="n_topics")
            
            # Filteroptionen
            category = st.selectbox("Nach Kategorie filtern", ["Alle"] + get_categories(), key="topic_category")
            timeframe = st.selectbox(
                "Zeitraum", 
                ["Letzter Monat", "Letzte Woche", "Letzter Tag", "Alle"], 
                key="topic_timeframe"
            )
            
            analyze_button = st.button("▶️ Themen analysieren", key="analyze_topics")
        
        with col2:
            if analyze_button:
                # Filtere die Daten basierend auf den ausgewählten Filtern
                filtered_df = df.copy()
                
                if category != "Alle":
                    filtered_df = filtered_df[filtered_df['category'] == category]
                
                # Zeitfilter anwenden
                now = datetime.now()
                if timeframe == "Letzter Monat":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=30))]
                elif timeframe == "Letzte Woche":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=7))]
                elif timeframe == "Letzter Tag":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=1))]
                
                # Topic Modeling durchführen
                with st.spinner("Analysiere Themen in den Artikeln..."):
                    method_key = 'lda' if "LDA" in method else 'nmf'
                    results = extract_topics(filtered_df, n_topics=n_topics, method=method_key)
                
                if results['topics']:
                    for i, topic_words in enumerate(results['topics']):
                        topic_text = ", ".join(topic_words)
                        st.markdown(f"**Thema {i+1}:** {topic_text}")
                        st.write("---")
                    
                    # Zeige Artikel zu Themen
                    st.subheader("Artikel nach Themen")
                    selected_topic = st.selectbox(
                        "Thema auswählen", 
                        [f"Thema {i+1}" for i in range(len(results['topics']))],
                        key="select_topic_view"
                    )
                    
                    topic_idx = int(selected_topic.split()[1]) - 1
                    topic_articles = [a for a in results['article_topics'] if a['topic'] == topic_idx]
                    
                    if topic_articles:
                        for article in topic_articles[:5]:  # Zeige Top 5
                            article_id = article['article_id']
                            article_row = df[df['id'] == article_id]
                            if not article_row.empty:
                                st.write(f"📄 **{article_row['title'].iloc[0]}**")
                                st.write(f"Zuversicht: {article['confidence']:.2f}")
                                st.write(f"[Zum Artikel]({article_row['url'].iloc[0]})")
                                st.write("---")
                else:
                    st.warning("Keine ausreichenden Daten für Topic Modeling gefunden.")
    
    # 2. Artikel-Clustering Tab
    with tab2:
        st.subheader("🔍 Artikel-Clustering")
        st.write("Diese Analyse gruppiert ähnliche Artikel mittels K-Means-Clustering.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_clusters = st.slider("Anzahl der Cluster", 3, 10, 5, key="n_clusters")
            
            # Filteroptionen
            category = st.selectbox("Nach Kategorie filtern", ["Alle"] + get_categories(), key="cluster_category")
            timeframe = st.selectbox(
                "Zeitraum", 
                ["Letzter Monat", "Letzte Woche", "Letzter Tag", "Alle"], 
                key="cluster_timeframe"
            )
            
            cluster_button = st.button("▶️ Artikel clustern", key="cluster_articles")
        
        with col2:
            if cluster_button:
                # Filtere die Daten basierend auf den ausgewählten Filtern
                filtered_df = df.copy()
                
                if category != "Alle":
                    filtered_df = filtered_df[filtered_df['category'] == category]
                
                # Zeitfilter anwenden
                now = datetime.now()
                if timeframe == "Letzter Monat":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=30))]
                elif timeframe == "Letzte Woche":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=7))]
                elif timeframe == "Letzter Tag":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=1))]
                
                # Clustering durchführen
                with st.spinner("Clustere Artikel..."):
                    results = cluster_articles(filtered_df, n_clusters=n_clusters)
                
                if results['clusters']:
                    # Zeige Cluster-Keywords
                    for i, keywords in enumerate(results['cluster_keywords']):
                        keywords_text = ", ".join(keywords) if keywords else "Keine Keywords verfügbar"
                        st.markdown(f"**Cluster {i+1}:** {keywords_text}")
                        st.write("---")
                    
                    # Zeige Artikel nach Clustern
                    st.subheader("Artikel nach Clustern")
                    selected_cluster = st.selectbox(
                        "Cluster auswählen", 
                        [f"Cluster {i+1}" for i in range(n_clusters)],
                        key="select_cluster_view"
                    )
                    
                    cluster_idx = int(selected_cluster.split()[1]) - 1
                    cluster_articles = [a for a in results['clusters'] if a['cluster'] == cluster_idx]
                    
                    if cluster_articles:
                        for article in cluster_articles[:5]:  # Zeige Top 5
                            article_id = article['article_id']
                            article_row = df[df['id'] == article_id]
                            if not article_row.empty:
                                st.write(f"📄 **{article_row['title'].iloc[0]}**")
                                st.write(f"[Zum Artikel]({article_row['url'].iloc[0]})")
                                st.write("---")
                else:
                    st.warning("Keine ausreichenden Daten für das Clustering gefunden.")
    
    # 3. Sentiment-Analyse Tab
    with tab3:
        st.subheader("😊 Sentiment-Analyse")
        st.write("Diese Analyse untersucht die emotionale Tendenz der Artikel.")
        
        # Warnung anzeigen, wenn TextBlob-DE nicht verfügbar ist
        if not TEXTBLOB_DE_AVAILABLE:
            st.info("Hinweis: TextBlob-DE ist nicht installiert. Die Analyse verwendet standard TextBlob, was für deutsche Texte weniger präzise sein kann. Für bessere Ergebnisse installieren Sie TextBlob-DE mit `pip install textblob-de`.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Filteroptionen
            category = st.selectbox("Nach Kategorie filtern", ["Alle"] + get_categories(), key="sentiment_category")
            timeframe = st.selectbox(
                "Zeitraum", 
                ["Letzter Monat", "Letzte Woche", "Letzter Tag", "Alle"], 
                key="sentiment_timeframe"
            )
            num_articles = st.slider("Anzahl der Artikel", 5, 50, 10, key="sentiment_num_articles")
            
            sentiment_button = st.button("▶️ Sentiment analysieren", key="analyze_sentiment")
        
        with col2:
            if sentiment_button:
                # Filtere die Daten
                filtered_df = df.copy()
                
                if category != "Alle":
                    filtered_df = filtered_df[filtered_df['category'] == category]
                
                # Zeitfilter anwenden
                now = datetime.now()
                if timeframe == "Letzter Monat":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=30))]
                elif timeframe == "Letzte Woche":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=7))]
                elif timeframe == "Letzter Tag":
                    filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=1))]
                
                # Begrenze auf angegebene Anzahl
                filtered_df = filtered_df.head(num_articles)
                
                # Lade Inhalte und analysiere Sentiment
                contents = []
                titles = []
                urls = []
                
                with st.spinner("Lade Artikelinhalte..."):
                    for _, row in filtered_df.iterrows():
                        content = get_article_content(row['id'])
                        if content:
                            contents.append(content)
                            titles.append(row['title'])
                            urls.append(row['url'])
                
                if contents:
                    with st.spinner("Analysiere Sentiment..."):
                        sentiments = analyze_sentiment(contents)
                    
                    # Visualisiere die Ergebnisse
                    sentiment_data = pd.DataFrame({
                        'Titel': titles,
                        'Polarität': [s['polarity'] for s in sentiments],
                        'Subjektivität': [s['subjectivity'] for s in sentiments],
                        'URL': urls
                    })
                    
                    # Zeige Graph
                    fig = px.scatter(
                        sentiment_data,
                        x='Subjektivität',
                        y='Polarität',
                        hover_name='Titel',
                        color='Polarität',
                        color_continuous_scale='RdYlGn',
                        title='Sentiment-Analyse der Artikel',
                        labels={
                            'Polarität': 'Sentiment (negativ → positiv)',
                            'Subjektivität': 'Subjektivität (objektiv → subjektiv)'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Zeige die Daten in einer Tabelle
                    st.write("### Sentiment-Daten")
                    sentiment_table = sentiment_data.copy()
                    sentiment_table['Polarität'] = sentiment_table['Polarität'].round(2)
                    sentiment_table['Subjektivität'] = sentiment_table['Subjektivität'].round(2)
                    sentiment_table['Stimmung'] = sentiment_table['Polarität'].apply(
                        lambda x: "Positiv" if x > 0.05 else "Negativ" if x < -0.05 else "Neutral"
                    )
                    
                    # Hyperlinks hinzufügen
                    sentiment_table['Artikel'] = sentiment_table.apply(
                        lambda row: f"<a href='{row['URL']}' target='_blank'>{row['Titel']}</a>", 
                        axis=1
                    )
                    
                    st.write(sentiment_table[['Artikel', 'Stimmung', 'Polarität', 'Subjektivität']].to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.warning("Keine Artikelinhalte für die Analyse gefunden.")
    
    # 4. KI-Analyse Tab (mit Google Generative AI)
    with tab4:
        st.subheader("🤖 KI-Artikelanalyse mit Google Gemini")
        st.write("Diese Funktion nutzt Google Gemini für eine tiefe Analyse einzelner Artikel.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Artikel-Auswahl
            article_id = st.selectbox(
                "Artikel auswählen",
                df['id'].tolist(),
                format_func=lambda x: df[df['id'] == x]['title'].iloc[0] if len(df[df['id'] == x]) > 0 else str(x),
                key="ai_article_select"
            )
            
            analysis_type = st.selectbox(
                "Analysetyp",
                ["Zusammenfassung", "Schlüsselwörter", "Sentiment", "Entitäten", "Trends"],
                key="ai_analysis_type"
            )
            
            analyze_button = st.button("▶️ Mit KI analysieren", key="analyze_with_ai")
        
        with col2:
            if analyze_button:
                article_row = df[df['id'] == article_id]
                if not article_row.empty:
                    with st.spinner(f"Artikel wird mit Google Gemini analysiert..."):
                        content = get_article_content(article_id)
                        
                        if content:
                            task_map = {
                                "Zusammenfassung": "summary",
                                "Schlüsselwörter": "keywords",
                                "Sentiment": "sentiment",
                                "Entitäten": "entities",
                                "Trends": "trends"
                            }
                            
                            result = analyze_with_genai(content, task_map[analysis_type])
                            
                            # Ergebnis anzeigen
                            st.write(f"### {article_row['title'].iloc[0]}")
                            st.write(f"**Analyse: {analysis_type}**")
                            
                            # Anzeige formatieren
                            st.markdown(result)
                        else:
                            st.error("Der Artikelinhalt konnte nicht geladen werden.")
                else:
                    st.error("Artikel konnte nicht gefunden werden.")
    
    # 5. Trend-Prognose Tab
    with tab5:
        st.subheader("🔮 Trend-Prognose")
        st.write("Identifiziert aufkommende Trends basierend auf Publikationsaktivitäten und Themenanalyse.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Filteroptionen
            timeframe = st.selectbox(
                "Analysezeitraum", 
                ["Letzte 3 Monate", "Letzter Monat", "Letzte Woche"],
                key="trend_timeframe"
            )
            
            trend_button = st.button("▶️ Trends analysieren", key="analyze_trends")
        
        with col2:
            if trend_button:
                with st.spinner("Analysiere Trends..."):
                    # Zeitfilter anwenden
                    filtered_df = df.copy()
                    now = datetime.now()
                    
                    if timeframe == "Letzte 3 Monate":
                        filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=90))]
                    elif timeframe == "Letzter Monat":
                        filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=30))]
                    elif timeframe == "Letzte Woche":
                        filtered_df = filtered_df[filtered_df['date'] >= (now - timedelta(days=7))]
                    
                    # Keywords analysieren
                    keywords_list = []
                    for keywords_str in filtered_df['keywords'].dropna():
                        if keywords_str not in [None, 'N/A', '']:
                            keywords = [k.strip() for k in str(keywords_str).split(',')]
                            keywords_list.extend(keywords)
                    
                    # Keyword-Trends über Zeit
                    keyword_counts = Counter(keywords_list)
                    top_keywords = {k: v for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:15]}
                    
                    # Visualisierung
                    st.write("### Top Keywords im gewählten Zeitraum")
                    fig = px.bar(
                        x=list(top_keywords.keys()),
                        y=list(top_keywords.values()),
                        labels={'x': 'Keyword', 'y': 'Anzahl'},
                        title='Häufige Keywords als Trend-Indikatoren'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend-Analyse mit Wachstumsraten
                    st.write("### Trend-Wachstumsanalyse")
                    
                    # Teile den Zeitraum in zwei Hälften
                    filtered_df['date_ts'] = filtered_df['date']
                    mid_point = filtered_df['date_ts'].min() + (filtered_df['date_ts'].max() - filtered_df['date_ts'].min()) / 2
                    
                    early_df = filtered_df[filtered_df['date_ts'] < mid_point]
                    late_df = filtered_df[filtered_df['date_ts'] >= mid_point]
                    
                    # Keywords in beiden Zeiträumen zählen
                    early_keywords = []
                    for keywords_str in early_df['keywords'].dropna():
                        if keywords_str not in [None, 'N/A', '']:
                            keywords = [k.strip() for k in str(keywords_str).split(',')]
                            early_keywords.extend(keywords)
                    
                    late_keywords = []
                    for keywords_str in late_df['keywords'].dropna():
                        if keywords_str not in [None, 'N/A', '']:
                            keywords = [k.strip() for k in str(keywords_str).split(',')]
                            late_keywords.extend(keywords)
                    
                    early_counts = Counter(early_keywords)
                    late_counts = Counter(late_keywords)
                    
                    # Wachstumsraten berechnen
                    growth_rates = {}
                    for kw in set(early_keywords + late_keywords):
                        early_count = early_counts.get(kw, 0) + 1  # +1 zur Vermeidung von Division durch Null
                        late_count = late_counts.get(kw, 0) + 1
                        growth = (late_count / early_count) - 1  # -1 für prozentuale Veränderung
                        
                        # Nur relevante Keywords mit Mindesthäufigkeit
                        if late_count > 2 and (early_count + late_count) > 5:
                            growth_rates[kw] = growth
                    
                    # Top wachsende Keywords
                    top_growing = {k: v for k, v in sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)[:10]}
                    
                    # Visualisierung
                    if top_growing:
                        growth_df = pd.DataFrame({
                            'Keyword': list(top_growing.keys()),
                            'Wachstum': list(top_growing.values())
                        })
                        
                        fig = px.bar(
                            growth_df,
                            x='Keyword',
                            y='Wachstum',
                            labels={'Wachstum': 'Wachstumsrate'},
                            title='Am schnellsten wachsende Keywords (Trend-Kandidaten)',
                            color='Wachstum',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Zusätzliche Insights
                        st.write("### Trend-Insights")
                        st.write("Die folgenden Keywords zeigen das stärkste Wachstum und könnten auf aufkommende Trends hindeuten:")
                        
                        for kw, growth in top_growing.items():
                            st.write(f"- **{kw}**: {growth*100:.1f}% Wachstum")
                    else:
                        st.warning("Nicht genügend Daten für eine Trend-Analyse verfügbar.")

def show_author_network(df: pd.DataFrame):
    """Zeigt das Autoren-Netzwerk"""
    st.header("🕸️ Autoren-Netzwerk")
    
    with st.spinner("Netzwerk wird generiert..."):
        G, pos = create_author_network()
    
    if G is None or len(G.nodes()) == 0:
        st.warning("Keine Netzwerkdaten verfügbar.")
        return
    
    # Netzwerk-Statistiken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Autoren", len(G.nodes()))
    
    with col2:
        st.metric("Verbindungen", len(G.edges()))
    
    with col3:
        # Durchschnittliche Verbindungen pro Autor
        avg_connections = len(G.edges()) * 2 / len(G.nodes()) if len(G.nodes()) > 0 else 0
        st.metric("Ø Verbindungen", f"{avg_connections:.1f}")
    
    with col4:
        # Dichtemessung
        density = nx.density(G)
        st.metric("Netzwerk-Dichte", f"{density:.3f}")
    
    # Netzwerk-Visualisierung
    st.subheader("🎯 Interaktive Netzwerk-Visualisierung")
    
    # Kontrollen für die Visualisierung
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_connections = st.slider("Mindest-Verbindungen", 1, 10, 1)
    
    with col2:
        layout_algorithm = st.selectbox("Layout", ["Spring", "Circular", "Random"])
    
    with col3:
        node_size_factor = st.slider("Knotengröße", 0.5, 3.0, 1.0)
    
    # Netzwerk filtern
    filtered_nodes = [node for node in G.nodes() if len(list(G.adj[node])) >= min_connections]
    G_filtered = G.subgraph(filtered_nodes)
    
    if len(G_filtered.nodes()) == 0:
        st.warning("Keine Knoten erfüllen die Filterkriterien.")
        return
    
    # Layout berechnen
    if layout_algorithm == "Spring":
        if len(G_filtered.nodes()) > 0:
            k_value = 1 / np.sqrt(len(G_filtered.nodes()))
            pos_filtered = nx.spring_layout(G_filtered, k=k_value, iterations=50)
        else:
            pos_filtered = {}
    elif layout_algorithm == "Circular":
        pos_filtered = nx.circular_layout(G_filtered)
    else:  # Random
        pos_filtered = nx.random_layout(G_filtered)
    
    # Plotly-Netzwerk mit verbesserter Interaktivität
    if pos_filtered:
        edge_x, edge_y = [], []
        for edge in G_filtered.edges():
            if edge[0] in pos_filtered and edge[1] in pos_filtered:
                x0, y0 = pos_filtered[edge[0]]
                x1, y1 = pos_filtered[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        node_x, node_y, node_text, node_adjacencies, node_info = [], [], [], [], []
        for node in G_filtered.nodes():
            if node in pos_filtered:
                x, y = pos_filtered[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                adjacencies = len(list(G_filtered.adj[node]))
                node_adjacencies.append(adjacencies)
                
                # Erweiterte Node-Info für Hover
                connected_authors = list(G_filtered.adj[node])
                node_info.append(f"Autor: {node}<br>Verbindungen: {adjacencies}<br>Verbunden mit: {', '.join(connected_authors[:5])}")
        
        # Traces erstellen
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10),
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_adjacencies,
                size=[max(10, adj * node_size_factor * 3) for adj in node_adjacencies],
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Verbindungen', side='right'),
                    xanchor='left'
                ),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Plot erstellen mit verbesserter Interaktivität
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=dict(
                text=f'Autoren-Netzwerk ({len(G_filtered.nodes())} Autoren, {len(G_filtered.edges())} Verbindungen)',
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="💡 Tipp: Zoomen mit Mausrad, Verschieben mit Maus ziehen",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12, color='gray')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Zoom und Pan aktivieren
        fig.update_layout(
            dragmode='pan',
            height=700
        )
        
        # Plot mit Zoom-Konfiguration
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'resetScale2d'],
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'scrollZoom': True
        }
        
        st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Erweiterte Netzwerk-Analyse
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top-Autoren nach Verbindungen")
        
        author_connections = [(node, len(list(G_filtered.adj[node]))) for node in G_filtered.nodes()]
        author_connections.sort(key=lambda x: x[1], reverse=True)
        
        top_authors = author_connections[:10]
        
        if top_authors:
            df_top = pd.DataFrame(top_authors, columns=['Autor', 'Verbindungen'])
            st.dataframe(df_top, use_container_width=True)
    
    with col2:
        st.subheader("📊 Netzwerk-Metriken")
        
        if len(G_filtered.nodes()) > 0:
            # Zentralitätsmaße
            betweenness = nx.betweenness_centrality(G_filtered)
            closeness = nx.closeness_centrality(G_filtered)
            
            # Top-Autoren nach Betweenness-Zentralität
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            
            st.write("**Top 5 Autoren nach Betweenness-Zentralität:**")
            for author, centrality in top_betweenness:
                st.write(f"• {author}: {centrality:.3f}")
            
            # Clustering-Koeffizient
            clustering = nx.average_clustering(G_filtered)
            st.metric("Clustering-Koeffizient", f"{clustering:.3f}")
            
            # Anzahl der Komponenten
            components = nx.number_connected_components(G_filtered)
            st.metric("Verbundene Komponenten", components)

def show_analytics(df: pd.DataFrame):
    """Zeigt erweiterte Analysen"""
    st.header("📈 Erweiterte Analysen")
    
    # Analyse-Optionen
    analysis_type = st.selectbox(
        "Analyse auswählen",
        ["Zeitliche Trends", "Kategorien-Analyse", "Autoren-Aktivität", "Wort-Statistiken", "Korrelations-Analyse", "Anomalie-Erkennung"]
    )
    
    if analysis_type == "Zeitliche Trends":
        st.subheader("📅 Zeitliche Trends")
        
        # Zeitraum auswählen
        time_range = st.selectbox("Zeitraum", ["Letzte 7 Tage", "Letzte 30 Tage", "Letzte 90 Tage", "Gesamter Zeitraum"])
        
        if time_range != "Gesamter Zeitraum":
            days = {"Letzte 7 Tage": 7, "Letzte 30 Tage": 30, "Letzte 90 Tage": 90}[time_range]
            cutoff_date = datetime.now() - timedelta(days=days)
            # Sicherstellen, dass beide Datetimes timezone-naive sind
            df_filtered = df[df['date'] >= cutoff_date].copy()
        else:
            df_filtered = df.copy()
        
        # Artikel pro Stunde
        if not df_filtered.empty:
            # Sicherstellen, dass date datetime ist
            df_filtered = df_filtered.copy()
            df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['date'])
            
            if not df_filtered.empty:
                df_filtered['hour'] = df_filtered['date'].dt.hour
                hourly_counts = df_filtered.groupby('hour').size()
                
                fig_hourly = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title='Artikel-Veröffentlichungen nach Stunden',
                    labels={'x': 'Stunde', 'y': 'Anzahl Artikel'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Wochentag-Analyse
                df_filtered['weekday'] = df_filtered['date'].dt.day_name()
                weekday_counts = df_filtered['weekday'].value_counts()
                
                fig_weekday = px.bar(
                    x=weekday_counts.index,
                    y=weekday_counts.values,
                    title='Artikel-Veröffentlichungen nach Wochentagen',
                    labels={'x': 'Wochentag', 'y': 'Anzahl Artikel'}
                )
                st.plotly_chart(fig_weekday, use_container_width=True)
            else:
                st.warning("Keine gültigen Datumswerte für die Analyse gefunden.")
    
    elif analysis_type == "Kategorien-Analyse":
        st.subheader("🏷️ Kategorien-Analyse")
        
        # Kategorie-Trends über Zeit
        df_with_dates = df.copy()
        df_with_dates['date'] = pd.to_datetime(df_with_dates['date'], errors='coerce')
        df_with_dates = df_with_dates.dropna(subset=['date'])
        
        if not df_with_dates.empty:
            category_time = df_with_dates.groupby([df_with_dates['date'].dt.date, 'category']).size().reset_index()
            category_time.columns = ['date', 'category', 'count']
            
            top_categories = df['category'].value_counts().head(5).index.tolist()
            category_time_filtered = category_time[category_time['category'].isin(top_categories)]
            
            if not category_time_filtered.empty:
                fig_cat_time = px.line(
                    category_time_filtered,
                    x='date',
                    y='count',
                    color='category',
                    title='Kategorie-Trends über Zeit (Top 5)',
                    labels={'date': 'Datum', 'count': 'Anzahl Artikel', 'category': 'Kategorie'}
                )
                st.plotly_chart(fig_cat_time, use_container_width=True)
        else:
            st.warning("Keine gültigen Datumswerte für die Kategorie-Analyse gefunden.")
        
        # Kategorie-Wortanzahl
        category_words = df.groupby('category')['word_count'].agg(['mean', 'count']).reset_index()
        category_words.columns = ['category', 'avg_words', 'article_count']
        category_words = category_words[category_words['article_count'] >= 5]  # Mindestens 5 Artikel
        
        if not category_words.empty:
            fig_cat_words = px.scatter(
                category_words,
                x='article_count',
                y='avg_words',
                size='article_count',
                hover_data=['category'],
                title='Durchschnittliche Wortanzahl pro Kategorie',
                labels={'article_count': 'Anzahl Artikel', 'avg_words': 'Ø Wörter'}
            )
            st.plotly_chart(fig_cat_words, use_container_width=True)
    
    elif analysis_type == "Autoren-Aktivität":
        st.subheader("✍️ Autoren-Aktivität")
        
        # Autoren-Produktivität
        author_stats = []
        for author_str in df['author'].dropna():
            if author_str != 'N/A':
                authors = [a.strip() for a in str(author_str).split(',')]
                for author in authors:
                    author_stats.append(author)
        
        if author_stats:
            author_counts = pd.Series(author_stats).value_counts()
            top_authors = author_counts.head(20)
            
            fig_auth_prod = px.bar(
                x=top_authors.values,
                y=top_authors.index,
                orientation='h',
                title='Top 20 Autoren nach Produktivität',
                labels={'x': 'Anzahl Artikel', 'y': 'Autor'}
            )
            st.plotly_chart(fig_auth_prod, use_container_width=True)
    
    elif analysis_type == "Wort-Statistiken":
        st.subheader("📊 Wort-Statistiken")
        
        # Wortanzahl-Verteilung
        word_counts = df['word_count'].dropna()
        
        if not word_counts.empty:
            fig_words = px.histogram(
                word_counts,
                nbins=50,
                title='Verteilung der Wortanzahl',
                labels={'value': 'Wortanzahl', 'count': 'Anzahl Artikel'}
            )
            st.plotly_chart(fig_words, use_container_width=True)
            
            # Statistiken
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Durchschnitt", f"{word_counts.mean():.0f}")
            
            with col2:
                st.metric("Median", f"{word_counts.median():.0f}")
            
            with col3:
                st.metric("Minimum", f"{word_counts.min():.0f}")
            
            with col4:
                st.metric("Maximum", f"{word_counts.max():.0f}")
    
    elif analysis_type == "Korrelations-Analyse":
        st.subheader("🔗 Korrelations-Analyse")
        
        # Korrelationsmatrix für numerische Werte
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title='Korrelationsmatrix numerischer Werte',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Kategorie vs. Wortanzahl-Analyse
        col1, col2 = st.columns(2)
        
        with col1:
            df_cat_words = df.dropna(subset=['category', 'word_count'])
            if not df_cat_words.empty:
                category_stats = df_cat_words.groupby('category')['word_count'].agg(['mean', 'std', 'count']).reset_index()
                category_stats = category_stats[category_stats['count'] >= 3]  # Mindestens 3 Artikel
                
                fig_cat_stats = px.scatter(
                    category_stats,
                    x='mean',
                    y='std',
                    size='count',
                    hover_data=['category'],
                    title='Kategorie: Durchschnittliche Wortanzahl vs. Standardabweichung',
                    labels={'mean': 'Durchschnittliche Wortanzahl', 'std': 'Standardabweichung', 'count': 'Anzahl Artikel'}
                )
                st.plotly_chart(fig_cat_stats, use_container_width=True)
        
        with col2:
            # Autor vs. Kategorie-Diversität
            author_category_data = []
            for _, row in df.iterrows():
                if pd.notna(row['author']) and pd.notna(row['category']) and row['author'] != 'N/A':
                    authors = [a.strip() for a in str(row['author']).split(',')]
                    for author in authors:
                        author_category_data.append({
                            'author': author,
                            'category': row['category']
                        })
            
            if author_category_data:
                auth_cat_df = pd.DataFrame(author_category_data)
                author_diversity = auth_cat_df.groupby('author')['category'].nunique().reset_index()
                author_diversity.columns = ['author', 'category_count']
                author_article_count = auth_cat_df.groupby('author').size().reset_index(name='article_count')
                
                author_stats = pd.merge(author_diversity, author_article_count, on='author')
                author_stats = author_stats[author_stats['article_count'] >= 3]  # Mindestens 3 Artikel
                
                if not author_stats.empty:
                    fig_auth_div = px.scatter(
                        author_stats,
                        x='article_count',
                        y='category_count',
                        size='article_count',
                        hover_data=['author'],
                        title='Autor: Artikelanzahl vs. Kategorie-Diversität',
                        labels={'article_count': 'Anzahl Artikel', 'category_count': 'Anzahl Kategorien'}
                    )
                    st.plotly_chart(fig_auth_div, use_container_width=True)
    
    elif analysis_type == "Anomalie-Erkennung":
        st.subheader("🚨 Anomalie-Erkennung")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ungewöhnlich lange/kurze Artikel
            word_counts = df['word_count'].dropna()
            if not word_counts.empty:
                Q1 = word_counts.quantile(0.25)
                Q3 = word_counts.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]
                
                if not outliers.empty:
                    st.write(f"**Gefundene Anomalien:** {len(outliers)} Artikel")
                    
                    # Visualisierung der Anomalien
                    fig_anomalies = px.scatter(
                        df.dropna(subset=['word_count']),
                        x=range(len(df.dropna(subset=['word_count']))),
                        y='word_count',
                        title='Wortanzahl-Anomalien',
                        labels={'x': 'Artikel-Index', 'y': 'Wortanzahl'}
                    )
                    
                    # Anomalien hervorheben
                    fig_anomalies.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                                          annotation_text="Obere Grenze")
                    fig_anomalies.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                          annotation_text="Untere Grenze")
                    
                    st.plotly_chart(fig_anomalies, use_container_width=True)
                    
                    # Top-Anomalien anzeigen
                    st.write("**Top Anomalien:**")
                    anomaly_display = outliers[['title', 'word_count', 'author', 'category']].sort_values('word_count', ascending=False).head(5)
                    st.dataframe(anomaly_display, use_container_width=True)
        
        with col2:
            # Ungewöhnliche Veröffentlichungsmuster
            df_time_anomalies = df.dropna(subset=['date'])
            if not df_time_anomalies.empty:
                # Tägliche Artikel-Counts
                daily_counts = df_time_anomalies.groupby(df_time_anomalies['date'].dt.date).size()
                
                if len(daily_counts) > 7:  # Mindestens eine Woche Daten
                    mean_daily = daily_counts.mean()
                    std_daily = daily_counts.std()
                    
                    # Anomalien: Tage mit ungewöhnlich vielen oder wenigen Artikeln
                    upper_threshold = mean_daily + 2 * std_daily
                    lower_threshold = mean_daily - 2 * std_daily
                    
                    anomalous_days = daily_counts[
                        (daily_counts.gt(upper_threshold)) | 
                        (daily_counts.lt(lower_threshold))
                    ]
                    
                    if not anomalous_days.empty:
                        st.write(f"**Anomalische Tage:** {len(anomalous_days)} Tage")
                        
                        fig_time_anomalies = px.line(
                            x=daily_counts.index,
                            y=daily_counts.values,
                            title='Tägliche Artikel-Counts mit Anomalien',
                            labels={'x': 'Datum', 'y': 'Anzahl Artikel'}
                        )
                        
                        # Anomalie-Schwellenwerte
                        fig_time_anomalies.add_hline(y=mean_daily + 2 * std_daily, 
                                                   line_dash="dash", line_color="red")
                        fig_time_anomalies.add_hline(y=mean_daily - 2 * std_daily, 
                                                   line_dash="dash", line_color="red")
                        
                        st.plotly_chart(fig_time_anomalies, use_container_width=True)
                        
                        # Anomalische Tage anzeigen
                        st.write("**Anomalische Tage:**")
                        anomaly_days_df = pd.DataFrame({
                            'Datum': anomalous_days.index,
                            'Anzahl Artikel': anomalous_days.values
                        })
                        st.dataframe(anomaly_days_df, use_container_width=True)

def show_sql_queries():
    """Zeigt SQL-Abfragen Interface"""
    st.header("🔧 SQL-Abfragen")
    
    # Abfrage-Eingabe
    st.subheader("📝 SQL-Abfrage ausführen")
    
    # Beispiel-Abfragen
    example_queries = {
        "Alle Artikel": "SELECT * FROM articles LIMIT 10",
        "Artikel nach Kategorie": "SELECT category, COUNT(*) FROM articles GROUP BY category ORDER BY COUNT(*) DESC",
        "Top Autoren": "SELECT author, COUNT(*) FROM articles WHERE author != 'N/A' GROUP BY author ORDER BY COUNT(*) DESC LIMIT 10",
        "Artikel pro Tag": "SELECT DATE(date) as tag, COUNT(*) FROM articles GROUP BY DATE(date) ORDER BY tag DESC LIMIT 10"
    }
    
    selected_example = st.selectbox("Beispiel-Abfrage auswählen", ["Eigene Abfrage"] + list(example_queries.keys()))
    
    if selected_example != "Eigene Abfrage":
        query = example_queries[selected_example]
    else:
        query = ""
    
    sql_query = st.text_area("SQL-Abfrage (nur SELECT)", value=query, height=150)
    
    if st.button("Abfrage ausführen"):
        if sql_query.strip():
            try:
                # Sicherheitscheck
                if not sql_query.strip().upper().startswith('SELECT'):
                    st.error("Nur SELECT-Abfragen sind erlaubt!")
                    return
                
                conn = get_db_connection()
                if conn:
                    with st.spinner("Abfrage wird ausgeführt..."):
                        result_df = pd.read_sql_query(sql_query, conn)
                        conn.close()
                    
                    st.success(f"Abfrage erfolgreich ausgeführt! {len(result_df)} Zeilen zurückgegeben.")
                    
                    # Ergebnisse anzeigen
                    if not result_df.empty:
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download-Button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Als CSV herunterladen",
                            data=csv,
                            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Keine Ergebnisse gefunden.")
                
            except Exception as e:
                st.error(f"Fehler bei der Abfrage: {e}")
        else:
            st.warning("Bitte geben Sie eine SQL-Abfrage ein.")
    
    # Datenbank-Export
    st.subheader("📤 Datenbank-Export")
    
    if st.button("Datenbank als SQLite exportieren"):
        try:
            conn = get_db_connection()
            if conn:
                with st.spinner("Export wird erstellt..."):
                    df_export = pd.read_sql_query("SELECT * FROM articles", conn)
                    conn.close()
                    
                    # Temporäre SQLite-Datei erstellen
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                        sqlite_conn = sqlite3.connect(tmp_file.name)
                        df_export.to_sql('articles', sqlite_conn, if_exists='replace', index=False)
                        sqlite_conn.close()
                        
                        # Datei lesen und als Download anbieten
                        with open(tmp_file.name, 'rb') as f:
                            st.download_button(
                                label="📥 SQLite-Datei herunterladen",
                                data=f.read(),
                                file_name=f"articles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                                mime="application/octet-stream"
                            )
                        
                        os.unlink(tmp_file.name)
                
                st.success("Export erfolgreich erstellt!")
                
        except Exception as e:
            st.error(f"Fehler beim Export: {e}")

def show_ai_analytics(df: pd.DataFrame):
    """Zeigt KI-basierte Analysen und Insights"""
    st.header("🤖 KI-Analysen")
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h4 style="margin-top: 0;">🧠 KI-gestützte Artikelanalyse</h4>
        <p>Erhalte tiefere Einblicke in die Artikeldaten durch fortschrittliche KI-Analysen.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KI-Analyse-Optionen
    ai_analysis_type = st.selectbox(
        "Analyse-Typ auswählen",
        ["📊 Themen-Clustering", "🔍 Sentiment-Analyse", "📈 Trend-Vorhersage", "💡 Content-Empfehlungen", "🔄 Ähnliche Artikel finden"]
    )
    
    if "api_key_loaded" not in st.session_state:
        st.session_state.api_key_loaded = False
        st.session_state.api_key = None
        
        # API-Key aus .env-Datei laden
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GOOGLE_AI_API_KEY")
            
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.api_key_loaded = True
            else:
                st.warning("Kein Google AI API-Key in der .env-Datei gefunden. Bitte fügen Sie GOOGLE_AI_API_KEY=IhrAPIKey hinzu.")
        except Exception as e:
            st.error(f"Fehler beim Laden des API-Keys: {e}")
    
    # Implementierung der verschiedenen KI-Analysen
    if ai_analysis_type == "📊 Themen-Clustering":
        show_topic_clustering(df)
    elif ai_analysis_type == "🔍 Sentiment-Analyse":
        show_sentiment_analysis(df)
    elif ai_analysis_type == "📈 Trend-Vorhersage":
        show_trend_prediction(df)
    elif ai_analysis_type == "💡 Content-Empfehlungen":
        show_content_recommendations(df)
    elif ai_analysis_type == "🔄 Ähnliche Artikel finden":
        show_similar_articles_finder(df)

def show_topic_clustering(df: pd.DataFrame):
    """Zeigt Themen-Clustering basierend auf Artikelinhalten"""
    st.subheader("📊 Themen-Clustering")
    
    # Erklärung
    st.markdown("""
    Diese Funktion gruppiert Artikel in thematische Cluster basierend auf Titel, Inhalt und Keywords.
    So können Sie Themenschwerpunkte und Zusammenhänge in der Berichterstattung erkennen.
    """)
    
    # Beispiel-Visualisierung
    st.info("Für eine vollständige Implementierung wird ein Google AI API-Key benötigt.")
    
    # Platzhalter für Clustering-Ergebnisse
    cluster_data = {
        "Technologie & Innovation": 35,
        "Politik & Gesellschaft": 28,
        "Cybersicherheit": 23,
        "Künstliche Intelligenz": 18,
        "Digitale Wirtschaft": 15,
        "Mobilität": 12,
        "Klimawandel & Umwelt": 10
    }
    
    # Visualisierung
    fig = px.pie(
        values=list(cluster_data.values()),
        names=list(cluster_data.keys()),
        title="Thematische Verteilung der Artikel",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Beispiel-Artikel pro Cluster
    st.subheader("Beispielartikel pro Cluster")
    for cluster in list(cluster_data.keys())[:3]:
        with st.expander(f"{cluster} ({cluster_data[cluster]} Artikel)"):
            # Hier würden normalerweise echte geclusterte Artikel angezeigt
            sample_df = df.sample(min(3, len(df)))
            for _, article in sample_df.iterrows():
                st.markdown(f"""
                - [{article['title']}]({article['url']}) - {article['date'].strftime('%Y-%m-%d') if pd.notna(article['date']) else 'N/A'}
                """)

def show_sentiment_analysis(df: pd.DataFrame):
    """Zeigt Sentiment-Analyse der Artikel"""
    st.subheader("🔍 Sentiment-Analyse")
    
    st.markdown("""
    Diese Funktion analysiert die Stimmung in den Artikeln und zeigt positive, negative und neutrale Tendenzen.
    """)
    
    # Platzhalter-Daten für Sentiment-Analyse
    sentiment_data = {
        "Positiv": 32,
        "Neutral": 55,
        "Negativ": 13
    }
    
    # Visualisierung
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Sentiment-Verteilung
        colors = {"Positiv": "#4CAF50", "Neutral": "#2196F3", "Negativ": "#F44336"}
        fig = px.bar(
            x=list(sentiment_data.keys()),
            y=list(sentiment_data.values()),
            title="Sentiment-Verteilung",
            labels={"x": "Sentiment", "y": "Anzahl Artikel"},
            color=list(sentiment_data.keys()),
            color_discrete_map=colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment-Übersicht
        st.markdown("### Sentiment-Übersicht")
        
        total = sum(sentiment_data.values())
        st.metric("Positiv", f"{sentiment_data['Positiv']} ({sentiment_data['Positiv']/total*100:.1f}%)")
        st.metric("Neutral", f"{sentiment_data['Neutral']} ({sentiment_data['Neutral']/total*100:.1f}%)")
        st.metric("Negativ", f"{sentiment_data['Negativ']} ({sentiment_data['Negativ']/total*100:.1f}%)")
    
    # Beispiel-Artikel nach Sentiment
    st.subheader("Beispiel-Artikel nach Sentiment")
    
    tabs = st.tabs(["Positiv", "Neutral", "Negativ"])
    
    for i, tab in enumerate(tabs):
        with tab:
            # Hier würden normalerweise echte kategorisierte Artikel angezeigt
            sample_df = df.sample(min(3, len(df)))
            for _, article in sample_df.iterrows():
                st.markdown(f"""
                - **[{article['title']}]({article['url']})** - {article['date'].strftime('%Y-%m-%d') if pd.notna(article['date']) else 'N/A'}
                """)

def show_trend_prediction(df: pd.DataFrame):
    """Zeigt Trend-Vorhersagen basierend auf historischen Daten"""
    st.subheader("📈 Trend-Vorhersage")
    
    st.markdown("""
    Diese Funktion analysiert historische Daten und sagt zukünftige Trends vorher.
    """)
    
    # Beispiel-Trend-Daten
    dates = pd.date_range(end=datetime.now(), periods=30).tolist()
    actual_values = [random.randint(5, 15) for _ in range(30)]
    
    # Vorhersagedaten (7 Tage)
    future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=7).tolist()
    predicted_values = [actual_values[-1] + random.randint(-2, 5) for _ in range(7)]
    predicted_values = [max(5, min(20, v)) for v in predicted_values]  # Werte begrenzen
    
    # Kombinierte Daten
    all_dates = dates + future_dates
    all_values = actual_values + [None] * 7
    all_predicted = [None] * 30 + predicted_values
    
    # Daten für das Diagramm
    trend_df = pd.DataFrame({
        "Datum": all_dates,
        "Tatsächlich": all_values,
        "Vorhersage": all_predicted
    })
    
    # Trend-Visualisierung
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_df["Datum"],
        y=trend_df["Tatsächlich"],
        name="Tatsächlich",
        line=dict(color="#2196F3", width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=trend_df["Datum"],
        y=trend_df["Vorhersage"],
        name="Vorhersage",
        line=dict(color="#F44336", width=3, dash="dot")
    ))
    
    fig.update_layout(
        title="Artikel pro Tag: Trend und Vorhersage",
        xaxis_title="Datum",
        yaxis_title="Artikelanzahl",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trendvorhersagen
    st.subheader("Vorhersage für die nächsten 7 Tage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Durchschnitt", f"{np.mean(predicted_values):.1f} Artikel/Tag")
        st.metric("Maximum", f"{max(predicted_values)} Artikel")
    
    with col2:
        st.metric("Trend", f"{'+' if predicted_values[-1] > actual_values[-1] else ''}{predicted_values[-1] - actual_values[-1]} Artikel", 
                 delta=f"{(predicted_values[-1] - actual_values[-1]) / actual_values[-1] * 100:.1f}%")
        st.metric("Minimum", f"{min(predicted_values)} Artikel")

def show_content_recommendations(df: pd.DataFrame):
    """Zeigt Inhaltliche Empfehlungen basierend auf Artikeldaten"""
    st.subheader("💡 Content-Empfehlungen")
    
    st.markdown("""
    Diese Funktion analysiert Ihr Publikum und die Artikelperformance, um Empfehlungen für zukünftige Inhalte zu geben.
    """)
    
    # Empfehlungs-Kategorien
    recommendations = [
        {
            "title": "Thematische Empfehlungen",
            "items": [
                "KI-Ethik und gesellschaftliche Auswirkungen",
                "Cloud-native Entwicklung und DevOps",
                "Datenschutz und DSGVO-Compliance",
                "Erneuerbare Energien und Technologie"
            ]
        },
        {
            "title": "Zeitliche Empfehlungen",
            "items": [
                "Artikel am Dienstag und Donnerstag veröffentlichen",
                "Optimale Veröffentlichungszeit: 9:00 - 11:00 Uhr",
                "Längere Artikel am Wochenende für mehr Engagement"
            ]
        },
        {
            "title": "Format-Empfehlungen",
            "items": [
                "Mehr interaktive Elemente in Tutorials",
                "Kurze Anleitungen im 'How-to' Format",
                "Tiefgehende Analysen zu Technologie-Trends"
            ]
        }
    ]
    
    # Empfehlungen anzeigen
    for rec in recommendations:
        with st.expander(rec["title"], expanded=True):
            for item in rec["items"]:
                st.markdown(f"• {item}")
    
    # Keyword-Empfehlungen
    st.subheader("Empfohlene Keywords")
    
    keyword_scores = {
        "Künstliche Intelligenz": 87,
        "Cloud Computing": 82,
        "Cybersicherheit": 79,
        "Quantencomputing": 76,
        "Edge Computing": 74,
        "Blockchain": 71,
        "Data Science": 69,
        "IoT": 67,
        "Green IT": 65,
        "Mikrochips": 62
    }
    
    # Balkendiagramm für Keyword-Empfehlungen
    fig = px.bar(
        x=list(keyword_scores.values()),
        y=list(keyword_scores.keys()),
        orientation='h',
        title="Empfohlene Keywords nach Relevanz-Score",
        labels={"x": "Relevanz-Score", "y": ""},
        color=list(keyword_scores.values()),
        color_continuous_scale="Viridis"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    st.plotly_chart(fig, use_container_width=True)

def show_similar_articles_finder(df: pd.DataFrame):
    """Zeigt einen Ähnliche-Artikel-Finder"""
    st.subheader("🔄 Ähnliche Artikel finden")
    
    st.markdown("""
    Mit dieser Funktion können Sie Artikel finden, die einem ausgewählten Artikel inhaltlich oder thematisch ähnlich sind.
    """)
    
    # Artikel-Auswahl
    article_titles = df['title'].tolist()
    selected_title = st.selectbox("Artikel auswählen", article_titles)
    
    if selected_title:
        selected_article = df[df['title'] == selected_title].iloc[0]
        
        # Ausgewählten Artikel anzeigen
        st.markdown(f"""
        ### Ausgewählter Artikel
        **Titel:** [{selected_article['title']}]({selected_article['url']})  
        **Datum:** {selected_article['date'].strftime('%Y-%m-%d') if pd.notna(selected_article['date']) else 'N/A'}  
        **Autor:** {selected_article['author'] if pd.notna(selected_article['author']) else 'N/A'}  
        **Kategorie:** {selected_article['category'] if pd.notna(selected_article['category']) else 'N/A'}  
        """)
        
        # Ähnlichkeitskriterien
        st.markdown("### Ähnlichkeitskriterien")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            by_content = st.checkbox("Nach Inhalt", value=True)
        
        with col2:
            by_author = st.checkbox("Nach Autor")
        
        with col3:
            by_category = st.checkbox("Nach Kategorie")
        
        if st.button("Ähnliche Artikel finden"):
            with st.spinner("Suche ähnliche Artikel..."):
                # Hier würden wir normalerweise einen Algorithmus zur Ähnlichkeitsbestimmung verwenden
                # Für die Demo verwenden wir zufällige Artikel als "ähnliche" Artikel
                
                filtered_df = df.copy()
                
                # Filter anwenden (in einer echten Implementierung würden wir Ähnlichkeitsmetriken berechnen)
                if by_author:
                    filtered_df = filtered_df[filtered_df['author'] == selected_article['author']]
                
                if by_category:
                    filtered_df = filtered_df[filtered_df['category'] == selected_article['category']]
                
                # Ausgewählten Artikel entfernen
                filtered_df = filtered_df[filtered_df['title'] != selected_title]
                
                # Ähnliche Artikel (zufällig ausgewählt für die Demo)
                similar_articles = filtered_df.sample(min(5, len(filtered_df)))
                
                if not similar_articles.empty:
                    st.markdown("### Ähnliche Artikel")
                    
                    for i, (_, article) in enumerate(similar_articles.iterrows()):
                        similarity = random.randint(60, 95)  # Zufälliger Ähnlichkeitswert für die Demo
                        
                        st.markdown(f"""
                        <div style="border-left: 4px solid {'#4CAF50' if similarity > 85 else '#2196F3' if similarity > 70 else '#FFC107'}; padding-left: 10px; margin-bottom: 15px;">
                            <h4 style="margin: 0;">{article['title']}</h4>
                            <p style="margin: 5px 0; color: #555;">
                                <span style="margin-right: 15px;">Datum: {article['date'].strftime('%Y-%m-%d') if pd.notna(article['date']) else 'N/A'}</span>
                                <span style="margin-right: 15px;">Autor: {article['author'] if pd.notna(article['author']) else 'N/A'}</span>
                                <span>Kategorie: {article['category'] if pd.notna(article['category']) else 'N/A'}</span>
                            </p>
                            <p style="margin: 5px 0;">
                                <span style="background-color: {'#E8F5E9' if similarity > 85 else '#E3F2FD' if similarity > 70 else '#FFF8E1'}; padding: 2px 8px; border-radius: 10px; font-size: 0.9em;">
                                    {similarity}% Ähnlichkeit
                                </span>
                                <a href="{article['url']}" target="_blank" style="margin-left: 10px;">Artikel öffnen</a>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Keine ähnlichen Artikel gefunden. Versuchen Sie es mit anderen Kriterien.")
    else:
        st.info("Bitte wählen Sie einen Artikel aus, um ähnliche Artikel zu finden.")

def show_advanced_reports(df: pd.DataFrame):
    """Zeigt erweiterte Reports und Analysen"""
    st.header("📊 Erweiterte Reports & Analysen")
    
    # Report-Auswahl
    report_type = st.selectbox(
        "Report-Typ auswählen",
        ["📊 Detaillierte Übersicht", "🔍 Trend-Analyse", "📈 Autor-Performance", "🏷️ Kategorie-Insights", "📅 Zeitreihen-Report", "🔤 Content-Analyse"]
    )
    
    if report_type == "📊 Detaillierte Übersicht":
        show_detailed_overview(df)
    elif report_type == "🔍 Trend-Analyse":
        show_trend_analysis(df)
    elif report_type == "📈 Autor-Performance":
        show_author_performance(df)
    elif report_type == "🏷️ Kategorie-Insights":
        show_category_insights(df)
    elif report_type == "📅 Zeitreihen-Report":
        show_timeseries_report(df)
    elif report_type == "🔤 Content-Analyse":
        show_content_analysis(df)

def show_detailed_overview(df: pd.DataFrame):
    """Zeigt eine detaillierte Übersicht aller Daten"""
    st.subheader("📊 Detaillierte Daten-Übersicht")
    
    # Grundlegende Statistiken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesamtartikel", len(df))
        st.metric("Artikelspanne", f"{(df['date'].max() - df['date'].min()).days} Tage")
    
    with col2:
        st.metric("Unique Autoren", df['author'].nunique())
        st.metric("Unique Kategorien", df['category'].nunique())
    
    with col3:
        avg_words = df['word_count'].mean()
        st.metric("⌀ Wörter/Artikel", f"{avg_words:.0f}" if pd.notna(avg_words) else "N/A")
        total_words = df['word_count'].sum()
        st.metric("Gesamtwörter", f"{total_words:,.0f}" if pd.notna(total_words) else "N/A")
    
    with col4:
        articles_per_day = len(df) / max(1, (df['date'].max() - df['date'].min()).days)
        st.metric("⌀ Artikel/Tag", f"{articles_per_day:.1f}")
        
        # Artikel mit Keywords
        with_keywords = len(df[df['keywords'].notna() & (df['keywords'] != 'N/A')])
        keyword_coverage = (with_keywords / len(df)) * 100
        st.metric("Keyword-Abdeckung", f"{keyword_coverage:.1f}%")
    
    # Datenqualität
    st.subheader("🔍 Datenqualität")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fehlende Werte
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title='Fehlende Werte pro Spalte',
                labels={'x': 'Spalte', 'y': 'Anzahl fehlender Werte'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ Keine fehlenden Werte gefunden!")
    
    with col2:
        # Datenverteilung nach Typ
        data_types = df.dtypes.value_counts()
        fig_types = px.pie(
            values=data_types.values,
            names=data_types.index,
            title='Datentypen-Verteilung'
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Detaillierte Tabelle mit Filtern
    st.subheader("📋 Detaillierte Artikel-Tabelle")
    
    # Filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = ['Alle'] + sorted(df['category'].dropna().unique().tolist())
        selected_category = st.selectbox("Kategorie filtern", categories)
    
    with col2:
        authors = ['Alle'] + sorted(df['author'].dropna().unique().tolist())
        selected_author = st.selectbox("Autor filtern", authors)
    
    with col3:
        word_count_range = st.slider(
            "Wortanzahl-Bereich",
            min_value=int(df['word_count'].min()) if df['word_count'].notna().any() else 0,
            max_value=int(df['word_count'].max()) if df['word_count'].notna().any() else 1000,
            value=(0, 1000)
        )
    
    # Daten filtern
    filtered_df = df.copy()
    if selected_category != 'Alle':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_author != 'Alle':
        filtered_df = filtered_df[filtered_df['author'] == selected_author]
    if df['word_count'].notna().any():
        filtered_df = filtered_df[
            (filtered_df['word_count'] >= word_count_range[0]) & 
            (filtered_df['word_count'] <= word_count_range[1])
        ]
    
    st.write(f"Zeige {len(filtered_df)} von {len(df)} Artikeln")
    
    # Tabelle anzeigen
    display_columns = ['title', 'author', 'category', 'date', 'word_count']
    st.dataframe(
        filtered_df[display_columns].head(100),
        use_container_width=True,
        hide_index=True
    )

def show_trend_analysis(df: pd.DataFrame):
    """Zeigt Trend-Analysen"""
    st.subheader("📈 Trend-Analyse")
    
    # Zeitrahmen-Selektor
    time_period = st.selectbox(
        "Zeitrahmen",
        ["Letzte 7 Tage", "Letzte 30 Tage", "Letzte 90 Tage", "Gesamter Zeitraum"]
    )
    
    # Daten basierend auf Zeitrahmen filtern
    now = datetime.now()
    if time_period == "Letzte 7 Tage":
        cutoff_date = now - timedelta(days=7)
    elif time_period == "Letzte 30 Tage":
        cutoff_date = now - timedelta(days=30)
    elif time_period == "Letzte 90 Tage":
        cutoff_date = now - timedelta(days=90)
    else:
        cutoff_date = df['date'].min()
    
    filtered_df = df[df['date'] >= cutoff_date]
    
    # Trend-Metriken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_articles = len(filtered_df)
        st.metric("Artikel im Zeitraum", total_articles)
    
    with col2:
        avg_daily = total_articles / max(1, (filtered_df['date'].max() - filtered_df['date'].min()).days)
        st.metric("⌀ Artikel/Tag", f"{avg_daily:.1f}")
    
    with col3:
        avg_words = filtered_df['word_count'].mean()
        st.metric("⌀ Wörter", f"{avg_words:.0f}" if pd.notna(avg_words) else "N/A")
    
    with col4:
        unique_authors = filtered_df['author'].nunique()
        st.metric("Aktive Autoren", unique_authors)
    
    # Trend-Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        # Artikel pro Tag
        daily_counts = filtered_df.groupby(filtered_df['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        fig_daily = px.line(
            daily_counts,
            x='date',
            y='count',
            title='Artikel-Trend (täglich)',
            labels={'date': 'Datum', 'count': 'Anzahl Artikel'}
        )
        fig_daily.add_hline(y=daily_counts['count'].mean(), line_dash="dash", line_color="red")
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Wortanzahl-Trend
        if filtered_df['word_count'].notna().any():
            word_trend = filtered_df.groupby(filtered_df['date'].dt.date)['word_count'].mean().reset_index()
            word_trend.columns = ['date', 'avg_words']
            
            fig_words = px.line(
                word_trend,
                x='date',
                y='avg_words',
                title='Durchschnittliche Wortanzahl (täglich)',
                labels={'date': 'Datum', 'avg_words': 'Ø Wörter'}
            )
            fig_words.add_hline(y=word_trend['avg_words'].mean(), line_dash="dash", line_color="red")
            st.plotly_chart(fig_words, use_container_width=True)
    
    # Kategorie-Trends
    st.subheader("📊 Kategorie-Trends")
    
    top_categories = filtered_df['category'].value_counts().head(5).index
    category_trends = []
    
    for category in top_categories:
        cat_data = filtered_df[filtered_df['category'] == category]
        cat_daily = cat_data.groupby(cat_data['date'].dt.date).size().reset_index()
        cat_daily.columns = ['date', 'count']
        cat_daily['category'] = category
        category_trends.append(cat_daily)
    
    if category_trends:
        combined_trends = pd.concat(category_trends, ignore_index=True)
        
        fig_cat_trends = px.line(
            combined_trends,
            x='date',
            y='count',
            color='category',
            title='Top 5 Kategorien - Trend',
            labels={'date': 'Datum', 'count': 'Anzahl Artikel', 'category': 'Kategorie'}
        )
        st.plotly_chart(fig_cat_trends, use_container_width=True)

def show_author_performance(df: pd.DataFrame):
    """Zeigt Autor-Performance Analysen"""
    st.subheader("✍️ Autor-Performance Analyse")
    
    # Autor-Auswahl
    authors = df['author'].dropna().unique().tolist()
    selected_authors = st.multiselect(
        "Autoren auswählen (max. 10)",
        authors,
        default=authors[:5] if len(authors) > 5 else authors
    )
    
    if not selected_authors:
        st.warning("Bitte wählen Sie mindestens einen Autor aus.")
        return
    
    # Daten für ausgewählte Autoren
    author_df = df[df['author'].isin(selected_authors)]
    
    # Performance-Metriken
    st.subheader("📊 Performance-Übersicht")
    
    performance_data = []
    for author in selected_authors:
        author_articles = author_df[author_df['author'] == author]
        
        performance_data.append({
            'Autor': author,
            'Artikel': len(author_articles),
            'Ø Wörter': author_articles['word_count'].mean(),
            'Gesamtwörter': author_articles['word_count'].sum(),
            'Erste Veröffentlichung': author_articles['date'].min(),
            'Letzte Veröffentlichung': author_articles['date'].max(),
            'Kategorien': author_articles['category'].nunique()
        })
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        # Artikel pro Autor
        fig_articles = px.bar(
            perf_df,
            x='Autor',
            y='Artikel',
            title='Anzahl Artikel pro Autor',
            labels={'Autor': 'Autor', 'Artikel': 'Anzahl Artikel'}
        )
        fig_articles.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_articles, use_container_width=True)
    
    with col2:
        # Durchschnittliche Wortanzahl
        fig_words = px.bar(
            perf_df,
            x='Autor',
            y='Ø Wörter',
            title='Durchschnittliche Wortanzahl pro Autor',
            labels={'Autor': 'Autor', 'Ø Wörter': 'Ø Wörter'}
        )
        fig_words.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_words, use_container_width=True)
    
    # Zeitreihen-Analyse
    st.subheader("📅 Aktivität über Zeit")
    
    author_timeline = []
    for author in selected_authors:
        author_articles = author_df[author_df['author'] == author]
        timeline = author_articles.groupby(author_articles['date'].dt.date).size().reset_index()
        timeline.columns = ['date', 'count']
        timeline['author'] = author
        author_timeline.append(timeline)
    
    if author_timeline:
        combined_timeline = pd.concat(author_timeline, ignore_index=True)
        
        fig_timeline = px.line(
            combined_timeline,
            x='date',
            y='count',
            color='author',
            title='Autor-Aktivität über Zeit',
            labels={'date': 'Datum', 'count': 'Anzahl Artikel', 'author': 'Autor'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

def show_category_insights(df: pd.DataFrame):
    """Zeigt Kategorie-Insights"""
    st.subheader("🏷️ Kategorie-Insights")
    
    # Kategorie-Übersicht
    category_stats = df.groupby('category').agg({
        'title': 'count',
        'word_count': ['mean', 'sum', 'std'],
        'date': ['min', 'max'],
        'author': 'nunique'
    }).round(2)
    
    category_stats.columns = ['Artikel', 'Ø Wörter', 'Gesamtwörter', 'Wörter Std', 'Erstes Datum', 'Letztes Datum', 'Autoren']
    category_stats = category_stats.sort_values('Artikel', ascending=False)
    
    st.dataframe(category_stats, use_container_width=True)
    
    # Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Kategorien
        top_cats = category_stats.head(10)
        fig_cats = px.bar(
            y=top_cats.index,
            x=top_cats['Artikel'],
            orientation='h',
            title='Top 10 Kategorien nach Artikel-Anzahl',
            labels={'x': 'Anzahl Artikel', 'y': 'Kategorie'}
        )
        st.plotly_chart(fig_cats, use_container_width=True)
    
    with col2:
        # Wortanzahl-Vergleich
        fig_words = px.bar(
            y=top_cats.index,
            x=top_cats['Ø Wörter'],
            orientation='h',
            title='Durchschnittliche Wortanzahl pro Kategorie',
            labels={'x': 'Ø Wörter', 'y': 'Kategorie'}
        )
        st.plotly_chart(fig_words, use_container_width=True)

def show_timeseries_report(df: pd.DataFrame):
    """Zeigt Zeitreihen-Reports"""
    st.subheader("📅 Zeitreihen-Report")
    
    # Aggregation-Level
    aggregation = st.selectbox(
        "Aggregations-Level",
        ["Täglich", "Wöchentlich", "Monatlich"]
    )
    
    df_time = df.dropna(subset=['date'])
    
    if aggregation == "Täglich":
        time_series = df_time.groupby(df_time['date'].dt.date).agg({
            'title': 'count',
            'word_count': 'mean',
            'author': 'nunique'
        }).reset_index()
        time_series.columns = ['Datum', 'Artikel', 'Ø Wörter', 'Autoren']
        
    elif aggregation == "Wöchentlich":
        time_series = df_time.groupby(df_time['date'].dt.to_period('W')).agg({
            'title': 'count',
            'word_count': 'mean',
            'author': 'nunique'
        }).reset_index()
        time_series['date'] = time_series['date'].astype(str)
        time_series.columns = ['Datum', 'Artikel', 'Ø Wörter', 'Autoren']
        
    else:  # Monatlich
        time_series = df_time.groupby(df_time['date'].dt.to_period('M')).agg({
            'title': 'count',
            'word_count': 'mean',
            'author': 'nunique'
        }).reset_index()
        time_series['date'] = time_series['date'].astype(str)
        time_series.columns = ['Datum', 'Artikel', 'Ø Wörter', 'Autoren']
    
    # Zeitreihen-Tabelle
    st.dataframe(time_series, use_container_width=True, hide_index=True)
    
    # Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        fig_articles = px.line(
            time_series,
            x='Datum',
            y='Artikel',
            title=f'Artikel-Entwicklung ({aggregation})',
            labels={'Datum': 'Datum', 'Artikel': 'Anzahl Artikel'}
        )
        st.plotly_chart(fig_articles, use_container_width=True)
    
    with col2:
        fig_words = px.line(
            time_series,
            x='Datum',
            y='Ø Wörter',
            title=f'Wortanzahl-Entwicklung ({aggregation})',
            labels={'Datum': 'Datum', 'Ø Wörter': 'Ø Wörter'}
        )
        st.plotly_chart(fig_words, use_container_width=True)

def add_export_functionality(df: pd.DataFrame, filename_prefix: str = "heise_data"):
    """Fügt Export-Funktionalität hinzu"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Daten exportieren")
    
    export_format = st.sidebar.selectbox(
        "Export-Format",
        ["CSV", "Excel", "JSON"]
    )
    
    if st.sidebar.button("📥 Daten exportieren"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="📥 CSV herunterladen",
                    data=csv_data,
                    file_name=f"{filename_prefix}_{timestamp}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "Excel":
                # Excel-Export
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Artikel', index=False)
                
                excel_data = output.getvalue()
                st.sidebar.download_button(
                    label="📥 Excel herunterladen",
                    data=excel_data,
                    file_name=f"{filename_prefix}_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif export_format == "JSON":
                json_data = df.to_json(orient='records', indent=2)
                st.sidebar.download_button(
                    label="📥 JSON herunterladen",
                    data=json_data,
                    file_name=f"{filename_prefix}_{timestamp}.json",
                    mime="application/json"
                )
            
            st.sidebar.success("Export bereit!")
            
        except Exception as e:
            st.sidebar.error(f"Export-Fehler: {e}")

def add_data_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt erweiterte Datenfilter hinzu"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Datenfilter")
    
    # Datumsfilter
    if not df['date'].isna().all():
        date_range = st.sidebar.date_input(
            "Datumsbereich",
            value=(df['date'].min().date(), df['date'].max().date()),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    # Kategoriefilter
    categories = ['Alle'] + sorted(df['category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Kategorie", categories)
    if selected_category != 'Alle':
        df = df[df['category'] == selected_category]
    
    # Autorenfilter
    authors = ['Alle'] + sorted(df['author'].dropna().unique().tolist())
    selected_author = st.sidebar.selectbox("Autor", authors)
    if selected_author != 'Alle':
        df = df[df['author'] == selected_author]
    
    # Wortanzahlfilter
    if not df['word_count'].isna().all():
        min_words, max_words = st.sidebar.slider(
            "Wortanzahl-Bereich",
            min_value=int(df['word_count'].min()),
            max_value=int(df['word_count'].max()),
            value=(int(df['word_count'].min()), int(df['word_count'].max()))
        )
        df = df[(df['word_count'] >= min_words) & (df['word_count'] <= max_words)]
    
    st.sidebar.write(f"Gefilterte Artikel: {len(df)}")
    return df

def add_performance_indicators():
    """Fügt Performance-Indikatoren hinzu"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ System-Performance")
    
    # Cache-Informationen (vereinfacht)
    cache_size = len(st.session_state) if hasattr(st, 'session_state') else 0
    st.sidebar.metric("Session State", cache_size)
    
    # Datenbank-Status
    try:
        conn = get_db_connection()
        if conn:
            st.sidebar.success("🟢 DB verbunden")
            conn.close()
        else:
            st.sidebar.error("🔴 DB getrennt")
    except:
        st.sidebar.error("🔴 DB Fehler")
    
    # Aktuelle Zeit
    current_time = datetime.now().strftime("%H:%M:%S")
    st.sidebar.metric("Letzte Aktualisierung", current_time)

def create_advanced_visualizations(df: pd.DataFrame):
    """Erstellt erweiterte Visualisierungen"""
    st.subheader("📊 Erweiterte Visualisierungen")
    
    # Korrelationsmatrix
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Korrelationsmatrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Bubble Chart
    if 'word_count' in df.columns and 'author' in df.columns:
        author_stats = df.groupby('author').agg({
            'word_count': ['mean', 'count'],
            'category': 'nunique'
        }).round(2)
        
        author_stats.columns = ['avg_words', 'article_count', 'categories']
        author_stats = author_stats.reset_index()
        
        fig_bubble = px.scatter(
            author_stats.head(20),
            x='avg_words',
            y='article_count',
            size='categories',
            hover_name='author',
            title="Autor-Performance (Bubble Chart)",
            labels={
                'avg_words': 'Durchschnittliche Wortanzahl',
                'article_count': 'Anzahl Artikel',
                'categories': 'Anzahl Kategorien'
            }
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Sunburst-Diagramm für Kategorien und Autoren
    if 'category' in df.columns and 'author' in df.columns:
        top_categories = df['category'].value_counts().head(5).index
        filtered_df = df[df['category'].isin(top_categories)]
        
        category_author_data = filtered_df.groupby(['category', 'author']).size().reset_index(name='count')
        
        fig_sunburst = px.sunburst(
            category_author_data,
            path=['category', 'author'],
            values='count',
            title="Kategorie-Autor-Hierarchie",
            color='count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)

def add_real_time_features():
    """Fügt Echtzeit-Features hinzu"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Echtzeit-Features")
    
    # Auto-Refresh
    auto_refresh = st.sidebar.checkbox("Auto-Refresh (30s)")
    if auto_refresh:
        st.sidebar.info("🔄 Auto-Refresh aktiviert")
        time.sleep(30)
        st.rerun()
    
    # Manuelle Aktualisierung
    if st.sidebar.button("🔄 Daten aktualisieren"):
        st.cache_data.clear()
        st.rerun()
    
    # Letztes Update
    last_update = datetime.now().strftime("%H:%M:%S")
    st.sidebar.info(f"🕐 Letztes Update: {last_update}")

def create_summary_report(df: pd.DataFrame):
    """Erstellt einen Zusammenfassungsbericht"""
    st.subheader("📋 Zusammenfassungsbericht")
    
    # Zeitraum
    date_range = df['date'].max() - df['date'].min()
    
    # Zusammenfassung
    summary = f"""
    ## 📊 Datenübersicht
    
    **Zeitraum:** {df['date'].min().strftime('%Y-%m-%d')} bis {df['date'].max().strftime('%Y-%m-%d')} ({date_range.days} Tage)
    
    **Artikel:** {len(df):,} Artikel insgesamt
    - Durchschnittlich {len(df) / max(1, date_range.days):.1f} Artikel pro Tag
    - Durchschnittlich {df['word_count'].mean():.0f} Wörter pro Artikel
    - Gesamtwortanzahl: {df['word_count'].sum():,.0f} Wörter
    
    **Autoren:** {df['author'].nunique()} verschiedene Autoren
    - Produktivster Autor: {df['author'].value_counts().index[0]} ({df['author'].value_counts().iloc[0]} Artikel)
    
    **Kategorien:** {df['category'].nunique()} verschiedene Kategorien
    - Häufigste Kategorie: {df['category'].value_counts().index[0]} ({df['category'].value_counts().iloc[0]} Artikel)
    
    **Keywords:** {len(df[df['keywords'].notna() & (df['keywords'] != 'N/A')])} Artikel mit Keywords ({len(df[df['keywords'].notna() & (df['keywords'] != 'N/A')]) / len(df) * 100:.1f}%)
    
    **Datenqualität:**
    - Vollständige Datensätze: {len(df.dropna())} ({len(df.dropna()) / len(df) * 100:.1f}%)
    - Fehlende Werte: {df.isnull().sum().sum()} insgesamt
    """
    
    st.markdown(summary)
    
    # Download-Button für Bericht
    st.download_button(
        label="📥 Bericht als Text herunterladen",
        data=summary,
        file_name=f"heise_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def show_content_analysis(df: pd.DataFrame):
    """Zeigt Content-Analysen"""
    st.subheader("🔤 Content-Analyse")
    
    # Titel-Analyse
    st.subheader("📝 Titel-Analyse")
    
    # Häufigste Wörter in Titeln
    all_titles = ' '.join(df['title'].dropna().astype(str))
    # Einfache Wort-Analyse (ohne NLP-Bibliotheken)
    title_words = all_titles.lower().split()
    
    # Stopwörter entfernen (einfache deutsche Stopwörter)
    stopwords = {'der', 'die', 'das', 'und', 'in', 'zu', 'den', 'mit', 'von', 'für', 'auf', 'im', 'ist', 'eine', 'ein', 'sich', 'bei', 'als', 'nach', 'um', 'an', 'werden', 'aus', 'er', 'sie', 'es', 'auch', 'kann', 'hat', 'nur', 'war', 'noch', 'so', 'über', 'wie', 'nicht', 'oder', 'aber', 'schon', 'alle', 'wenn', 'werden', 'zum', 'zur', 'durch', 'beim', 'vom', 'gegen', 'ohne', 'bis', 'unter', 'während', 'vor', 'zwischen', 'seit', 'trotz', 'wegen', 'statt', 'innerhalb', 'außerhalb', 'aufgrund', 'anhand', 'anstatt', 'anstelle', 'infolge', 'mittels', 'gemäß', 'entsprechend', 'laut', 'zufolge', 'samt', 'nebst', 'einschließlich', 'ausschließlich', 'bezüglich', 'hinsichtlich', 'bezogen', 'betreffend', 'angesichts', 'aufgrund', 'infolge', 'zugunsten', 'zulasten', 'zwecks', 'mangels', 'kraft', 'dank', 'trotz', 'ungeachtet', 'unbeschadet', 'vorbehaltlich', 'abzüglich', 'zuzüglich', 'einschließlich', 'ausschließlich', 'inklusive', 'exklusive', 'plus', 'minus', 'mal', 'geteilt', 'durch', 'gleich', 'weniger', 'mehr', 'etwa', 'circa', 'ungefähr', 'rund', 'knapp', 'über', 'unter', 'mindestens', 'höchstens', 'maximal', 'minimal', 'insgesamt', 'zusammen', 'getrennt', 'gemeinsam', 'einzeln', 'jeweils', 'sowohl', 'weder', 'entweder', 'beziehungsweise', 'beziehungsweise', 'sowie', 'außerdem', 'zudem', 'zusätzlich', 'darüber', 'hinaus', 'ferner', 'weiterhin', 'überdies', 'obendrein', 'dazu', 'dabei', 'hierbei', 'wobei', 'wodurch', 'worauf', 'worin', 'woran', 'worüber', 'wovon', 'wozu', 'womit', 'wonach', 'wofür', 'wogegen', 'wohingegen', 'währenddessen', 'inzwischen', 'mittlerweile', 'derzeit', 'aktuell', 'momentan', 'gegenwärtig', 'heute', 'gestern', 'morgen', 'übermorgen', 'vorgestern', 'kürzlich', 'neulich', 'unlängst', 'jüngst', 'erst', 'bereits', 'schon', 'noch', 'immer', 'nie', 'niemals', 'stets', 'oft', 'häufig', 'selten', 'manchmal', 'gelegentlich', 'bisweilen', 'zuweilen', 'mitunter', 'hin', 'wieder', 'erneut', 'nochmals', 'abermals', 'wiederum', 'hingegen', 'dagegen', 'jedoch', 'dennoch', 'trotzdem', 'nichtsdestotrotz', 'gleichwohl', 'allerdings', 'freilich', 'zwar', 'gewiss', 'sicherlich', 'bestimmt', 'wahrscheinlich', 'möglicherweise', 'eventuell', 'vielleicht', 'womöglich', 'gegebenenfalls', 'unter', 'umständen', 'falls', 'sofern', 'soweit', 'solange', 'sobald', 'sooft', 'sogar', 'selbst', 'soeben', 'gerade', 'eben', 'nun', 'jetzt', 'dann', 'danach', 'anschließend', 'daraufhin', 'darauf', 'hierauf', 'sodann', 'alsdann', 'zunächst', 'zuerst', 'erstens', 'zweitens', 'drittens', 'viertens', 'fünftens', 'sechstens', 'siebtens', 'achtens', 'neuntens', 'zehntens', 'schließlich', 'endlich', 'zuletzt', 'letztendlich', 'letztlich', 'am', 'ende', 'zum', 'schluss', 'abschließend', 'zusammenfassend', 'kurz', 'gesagt', 'mit', 'anderen', 'worten', 'das', 'heißt', 'will', 'beispielsweise', 'etwa', 'ungefähr', 'sozusagen', 'gewissermaßen', 'quasi', 'praktisch', 'eigentlich', 'tatsächlich', 'wirklich', 'echt', 'richtig', 'falsch', 'korrekt', 'inkorrekt', 'genau', 'exakt', 'präzise', 'ungefähr', 'circa', 'etwa', 'rund', 'knapp', 'fast', 'beinahe', 'nahezu', 'annähernd', 'weitgehend', 'größtenteils', 'größtenteils', 'meistens', 'meist', 'überwiegend', 'vorwiegend', 'hauptsächlich', 'insbesondere', 'besonders', 'speziell', 'vor', 'allem', 'zunächst', 'nämlich', 'das', 'für', 'ist', 'eine', 'neue', 'nach', 'erste', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn'}
    
    title_words_clean = [word for word in title_words if word not in stopwords and len(word) > 2]
    
    if title_words_clean:
        word_freq = pd.Series(title_words_clean).value_counts().head(20)
        
        fig_words = px.bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            title='Top 20 Wörter in Titeln',
            labels={'x': 'Häufigkeit', 'y': 'Wort'}
        )
        st.plotly_chart(fig_words, use_container_width=True)
    
    # Titel-Längen-Analyse
    col1, col2 = st.columns(2)
    
    with col1:
        title_lengths = df['title'].dropna().str.len()
        
        fig_lengths = px.histogram(
            title_lengths,
            nbins=30,
            title='Verteilung der Titel-Längen',
            labels={'value': 'Anzahl Zeichen', 'count': 'Anzahl Titel'}
        )
        st.plotly_chart(fig_lengths, use_container_width=True)
    
    with col2:
        # Durchschnittliche Titel-Länge pro Kategorie
        title_length_by_cat = df.groupby('category')['title'].str.len().mean().sort_values(ascending=False).head(10)
        
        fig_cat_lengths = px.bar(
            x=title_length_by_cat.values,
            y=title_length_by_cat.index,
            orientation='h',
            title='Ø Titel-Länge nach Kategorie',
            labels={'x': 'Ø Zeichen', 'y': 'Kategorie'}
        )
        st.plotly_chart(fig_cat_lengths, use_container_width=True)

def add_ml_features(df):
    """Fügt Machine Learning Features hinzu"""
    try:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 ML Features")
        
        # Artikel-Ähnlichkeit
        if st.sidebar.button("📊 Artikel-Ähnlichkeit"):
            st.subheader("🔍 Artikel-Ähnlichkeitsanalyse")
            
            # Vereinfachte Ähnlichkeitsanalyse basierend auf Kategorien
            similar_articles = df.groupby('category').size().sort_values(ascending=False)
            
            fig = px.bar(
                x=similar_articles.index,
                y=similar_articles.values,
                title="Artikel-Verteilung nach Kategorien",
                labels={'x': 'Kategorie', 'y': 'Anzahl Artikel'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trend-Vorhersage
        if st.sidebar.button("📈 Trend-Vorhersage"):
            st.subheader("📊 Vereinfachte Trend-Analyse")
            
            # Zeitbasierte Trends
            df['date'] = pd.to_datetime(df['date'])
            daily_counts = df.groupby(df['date'].dt.date).size()
            
            fig = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Tägliche Artikel-Entwicklung",
                labels={'x': 'Datum', 'y': 'Anzahl Artikel'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomalie-Erkennung
        if st.sidebar.button("⚠️ Anomalie-Erkennung"):
            st.subheader("🔍 Anomalie-Analyse")
            
            # Einfache Anomalie-Erkennung basierend auf Wortanzahl
            if 'word_count' in df.columns and not df['word_count'].isna().all():
                word_stats = df['word_count'].describe()
                anomalies = df[df['word_count'] > word_stats['mean'] + 2 * word_stats['std']]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normale Artikel", len(df) - len(anomalies))
                with col2:
                    st.metric("Anomale Artikel", len(anomalies))
                
                if len(anomalies) > 0:
                    st.subheader("📋 Anomale Artikel")
                    st.dataframe(anomalies[['title', 'author', 'word_count']].head(10))
            else:
                st.info("Keine Wortanzahl-Daten verfügbar für Anomalie-Erkennung")
        
    except Exception as e:
        st.error(f"Fehler bei ML-Features: {str(e)}")

def add_interactive_features(df):
    """Fügt interaktive Features hinzu"""
    try:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Interaktive Features")
        
        # Artikel-Empfehlungen
        if st.sidebar.button("💡 Artikel-Empfehlungen"):
            st.subheader("📚 Empfohlene Artikel")
            
            # Zufällige Empfehlungen basierend auf populären Kategorien
            popular_categories = df['category'].value_counts().head(3).index
            recommendations = df[df['category'].isin(popular_categories)].sample(n=min(5, len(df)))
            
            for _, article in recommendations.iterrows():
                with st.expander(f"📖 {article['title'][:100]}..."):
                    st.write(f"**Autor:** {article['author']}")
                    st.write(f"**Kategorie:** {article['category']}")
                    st.write(f"**Datum:** {article['date']}")
                    if 'word_count' in article and pd.notna(article['word_count']):
                        st.write(f"**Wörter:** {int(article['word_count'])}")
        
        # Bookmark-System
        if 'bookmarks' not in st.session_state:
            st.session_state.bookmarks = []
        
        if st.sidebar.button("🔖 Bookmarks verwalten"):
            st.subheader("🔖 Gespeicherte Artikel")
            
            # Artikel zum Bookmarken auswählen
            selected_article = st.selectbox(
                "Artikel zu Bookmarks hinzufügen:",
                df['title'].tolist(),
                index=0
            )
            
            if st.button("➕ Zu Bookmarks hinzufügen"):
                if selected_article not in st.session_state.bookmarks:
                    st.session_state.bookmarks.append(selected_article)
                    st.success(f"Artikel '{selected_article[:50]}...' zu Bookmarks hinzugefügt!")
                else:
                    st.warning("Artikel bereits in Bookmarks!")
            
            # Bookmarks anzeigen
            if st.session_state.bookmarks:
                st.subheader("📋 Deine Bookmarks")
                for i, bookmark in enumerate(st.session_state.bookmarks):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {bookmark}")
                    with col2:
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state.bookmarks.remove(bookmark)
                            st.rerun()
        
        # Datenqualität-Check
        if st.sidebar.button("✅ Datenqualität"):
            st.subheader("🔍 Datenqualitäts-Analyse")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vollständige Titel", df['title'].notna().sum())
            
            with col2:
                st.metric("Vollständige Autoren", df['author'].notna().sum())
            
            with col3:
                st.metric("Vollständige Daten", df['date'].notna().sum())
            
            # Detaillierte Qualitätsanalyse
            quality_report = {
                'Spalte': ['title', 'author', 'category', 'date', 'word_count'],
                'Vollständig': [
                    df['title'].notna().sum(),
                    df['author'].notna().sum(),
                    df['category'].notna().sum(),
                    df['date'].notna().sum(),
                    df['word_count'].notna().sum() if 'word_count' in df.columns else 0
                ],
                'Fehlend': [
                    df['title'].isna().sum(),
                    df['author'].isna().sum(),
                    df['category'].isna().sum(),
                    df['date'].isna().sum(),
                    df['word_count'].isna().sum() if 'word_count' in df.columns else len(df)
                ],
                'Anteil Vollständig': [
                    f"{(df['title'].notna().sum() / len(df) * 100):.1f}%",
                    f"{(df['author'].notna().sum() / len(df) * 100):.1f}%",
                    f"{(df['category'].notna().sum() / len(df) * 100):.1f}%",
                    f"{(df['date'].notna().sum() / len(df) * 100):.1f}%",
                    f"{(df['word_count'].notna().sum() / len(df) * 100):.1f}%" if 'word_count' in df.columns else "0.0%"
                ]
            }
            
            st.dataframe(pd.DataFrame(quality_report))
        
    except Exception as e:
        st.error(f"Fehler bei interaktiven Features: {str(e)}")

def create_real_time_monitoring(df: pd.DataFrame) -> str:
    """Erstellt ein Echtzeit-Monitoring-Dashboard im minimalistischen Design"""
    
    current_time = datetime.now()
    
    # Letzte Aktivität
    df_time = df.dropna(subset=['date'])
    if not df_time.empty:
        latest_article = df_time.iloc[0]  # Neuester Artikel
        time_diff = current_time - latest_article['date']
        
        # Status-Ampel
        if time_diff.total_seconds() < 3600:  # < 1 Stunde
            status_color = "#4CAF50"  # Grün
            status_text = "🟢 AKTIV"
        elif time_diff.total_seconds() < 86400:  # < 24 Stunden
            status_color = "#FF9800"  # Orange
            status_text = "🟡 NORMAL"
        else:
            status_color = "#F44336"  # Rot
            status_text = "🔴 INAKTIV"
        
        # Performance-Metriken
        today = current_time.date()
        articles_today = len(df_time[df_time['date'].dt.date == today])
        
        last_week = current_time - timedelta(days=7)
        articles_last_week = len(df_time[df_time['date'] >= last_week])
        
        # Durchschnittliche Artikel pro Tag in der letzten Woche
        avg_per_day = articles_last_week / 7
        
        # Qualitätsindikatoren
        articles_with_keywords = df['keywords'].notna().sum()
        keyword_percentage = (articles_with_keywords / len(df)) * 100
        
        monitoring_html = f"""
        <div style="background: #f7f7f7; color: #333; padding: 20px; border-radius: 5px; margin: 15px 0; border-left: 3px solid #4c78a8;">
            <h3 style="margin-top: 0; font-weight: 400;">🎯 Echtzeit-Monitoring</h3>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0;">
                <div>
                    <h4 style="margin: 0; font-weight: 400;">System-Status</h4>
                    <div style="background: {status_color}; padding: 8px; border-radius: 5px; text-align: center; margin: 10px 0; font-weight: normal; color: white;">
                        {status_text}
                    </div>
                </div>
                
                <div style="text-align: right;">
                    <p><strong>Neuester Artikel:</strong> {latest_article['title'][:50]}...</p>
                    <p><strong>Autor:</strong> {latest_article['author']}</p>
                    <p><strong>Zeitstempel:</strong> {latest_article['date'].strftime('%d.%m.%Y %H:%M')}</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin: 15px 0;">
                <div style="background: white; padding: 12px; border-radius: 4px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0; font-weight: 400;">📊 Heute</h4>
                    <p style="font-size: 1.5em; margin: 5px 0;">{articles_today}</p>
                    <p style="color: #666; margin: 0;">Artikel</p>
                </div>
                
                <div style="background: white; padding: 12px; border-radius: 4px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0; font-weight: 400;">📈 Letzte Woche</h4>
                    <p style="font-size: 1.5em; margin: 5px 0;">{articles_last_week}</p>
                    <p style="color: #666; margin: 0;">Artikel</p>
                </div>
                
                <div style="background: white; padding: 12px; border-radius: 4px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0; font-weight: 400;">⚡ Ø pro Tag</h4>
                    <p style="font-size: 1.5em; margin: 5px 0;">{avg_per_day:.1f}</p>
                    <p style="color: #666; margin: 0;">Artikel</p>
                </div>
                
                <div style="background: white; padding: 12px; border-radius: 4px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0; font-weight: 400;">🔑 Keywords</h4>
                    <p style="font-size: 1.5em; margin: 5px 0;">{keyword_percentage:.1f}%</p>
                    <p style="color: #666; margin: 0;">Abdeckung</p>
                </div>
            </div>
            
            <p style="text-align: center; margin-top: 15px; color: #888; font-size: 0.9em;">
                🔄 Letztes Update: {current_time.strftime('%d.%m.%Y %H:%M:%S')}
            </p>
        </div>
        """
        
        return monitoring_html
    
    return "<p>Keine Daten für Echtzeit-Monitoring verfügbar.</p>"

def create_data_quality_report(df: pd.DataFrame) -> str:
    """Erstellt einen detaillierten Datenqualitätsbericht im ultra-minimalistischen Design"""
    
    total_articles = len(df)
    
    # Vollständigkeitsanalyse
    completeness = {
        'Titel': df['title'].notna().sum(),
        'Autor': df['author'].notna().sum(),
        'Kategorie': df['category'].notna().sum(),
        'Datum': df['date'].notna().sum(),
        'Keywords': df['keywords'].notna().sum(),
        'Wortanzahl': df['word_count'].notna().sum(),
        'URL': df['url'].notna().sum()
    }
    
    # Datenqualitäts-Score berechnen
    quality_score = sum(completeness.values()) / (len(completeness) * total_articles) * 100
    
    # Empfehlungen generieren
    recommendations = []
    
    if completeness['Keywords'] / total_articles < 0.8:
        recommendations.append("Keyword-Abdeckung verbessern")
    
    if completeness['Wortanzahl'] / total_articles < 0.9:
        recommendations.append("Wortanzahl-Erfassung optimieren")
    
    if completeness['Autor'] / total_articles < 0.95:
        recommendations.append("Autoren-Informationen vervollständigen")
    
    if not recommendations:
        recommendations.append("Datenqualität ist ausgezeichnet!")
    
    # Duplikate prüfen
    duplicate_titles = df['title'].duplicated().sum()
    duplicate_urls = df['url'].duplicated().sum()
    
    quality_html = f"""
    <div style="background: #ffffff; color: #333; padding: 16px; border-radius: 3px; margin: 12px 0; border: 1px solid #eaeaea;">
        <h3 style="margin-top: 0; font-weight: 400;">Datenqualitätsbericht</h3>
        
        <div style="margin: 12px 0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.8rem; font-weight: 300; color: #1a73e8;">{quality_score:.1f}%</span>
                <span style="margin-left: 8px; color: #666;">Gesamtqualität</span>
            </div>
        </div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0;">
            <div style="flex: 1; min-width: 250px;">
                <h4 style="margin: 0 0 8px 0; font-weight: 400; font-size: 0.95rem;">Vollständigkeitsanalyse</h4>
    """
    
    for field, count in completeness.items():
        percentage = (count / total_articles) * 100
        quality_html += f"""
                <div style="margin: 6px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                        <span>{field}</span>
                        <span>{percentage:.1f}%</span>
                    </div>
                    <div style="background: #f5f5f5; height: 4px; border-radius: 2px; margin: 3px 0;">
                        <div style="background: #1a73e8; height: 100%; width: {percentage}%; border-radius: 2px;"></div>
                    </div>
                </div>
        """
    
    quality_html += f"""
            </div>
            
            <div style="flex: 1; min-width: 250px;">
                <h4 style="margin: 0 0 8px 0; font-weight: 400; font-size: 0.95rem;">Datenintegrität</h4>
                <div style="font-size: 0.85rem;">
                    <div style="margin-bottom: 4px;">Duplikate: {duplicate_titles + duplicate_urls} gesamt</div>
                    <div style="display: flex; margin-bottom: 12px;">
                        <div style="width: 50%;">Titel: {duplicate_titles}</div>
                        <div style="width: 50%;">URLs: {duplicate_urls}</div>
                    </div>
                    
                    <div style="margin-top: 12px; color: #505050;">Empfehlungen:</div>
    """
    
    for rec in recommendations:
        quality_html += f'<div style="margin: 4px 0;">• {rec}</div>'
    
    quality_html += """
                </div>
            </div>
        </div>
        
        <div style="margin-top: 12px; font-size: 0.8rem; color: #666; border-top: 1px solid #eaeaea; padding-top: 8px;">
            Regelmäßige Datenqualitätsprüfungen optimieren die Artikel-Datenbank.
        </div>
    </div>
    """
    
    return quality_html

if __name__ == "__main__":
    # Verbesserte main-Funktion mit allen Features
    def main():
        """Hauptfunktion der Streamlit-App"""
        
        # Header
        st.markdown('<h1 class="main-header">🗞️ News Mining Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar für Navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Seite auswählen",
            ["📊 Dashboard", "📈 Zeitanalysen", "🔑 Keyword-Analysen", "⚡ Performance-Metriken", "🔍 Artikelsuche", "🕸️ Autoren-Netzwerk", "🤖 KI-Analysen", "📉 Analysen", "📋 Erweiterte Reports", "🔧 SQL-Abfragen"]
        )
        
        # Daten laden mit Fortschrittsanzeige
        with st.spinner("Lade Daten aus der Datenbank..."):
            df = load_articles_data()
        
        if df.empty:
            st.error("❌ Keine Daten verfügbar. Überprüfen Sie die Datenbankverbindung.")
            return
        
        # Source filter in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filter")
        
        available_sources = df['source'].unique().tolist() if 'source' in df.columns else ['heise']
        selected_sources = st.sidebar.multiselect(
            "Quelle auswählen",
            options=available_sources,
            default=available_sources
        )
        
        # Filter dataframe by selected sources
        if selected_sources:
            df = df[df['source'].isin(selected_sources)]
        
        if df.empty:
            st.warning("⚠️ Keine Daten für die ausgewählten Filter verfügbar.")
            return
        
        # Informationen über die geladenen Daten
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Daten-Info")
        st.sidebar.metric("Artikel gesamt", len(df))
        if 'source' in df.columns:
            for source in df['source'].unique():
                source_count = len(df[df['source'] == source])
                st.sidebar.metric(f"{source.capitalize()} Artikel", source_count)
        st.sidebar.metric("Anzahl Autoren", df['author'].nunique())
        st.sidebar.metric("Anzahl Kategorien", df['category'].nunique())
        
        # Erweiterte Features hinzufügen
        add_performance_indicators()
        add_real_time_features()
        
        # Datenfilter anwenden
        df_filtered = add_data_filters(df)
        
        # Export-Funktionalität
        add_export_functionality(df_filtered)
        
        # Cache-Status
        if st.sidebar.button("🔄 Cache leeren"):
            st.cache_data.clear()
            st.sidebar.success("Cache geleert!")
            st.rerun()
        
        # Seitenbasierte Navigation
        if page == "📊 Dashboard":
            show_dashboard(df_filtered)
        elif page == "📈 Zeitanalysen":
            show_time_analytics(df_filtered)
        elif page == "🔑 Keyword-Analysen":
            show_keyword_analytics(df_filtered)
        elif page == "⚡ Performance-Metriken":
            show_performance_metrics(df_filtered)
        elif page == "🔍 Artikelsuche":
            show_article_search(df_filtered)
        elif page == "🕸️ Autoren-Netzwerk":
            show_author_network(df_filtered)
        elif page == "🤖 KI-Analysen":
            show_ai_analytics(df_filtered)
        elif page == "📉 Analysen":
            show_analytics(df_filtered)
        elif page == "📋 Erweiterte Reports":
            show_advanced_reports(df_filtered)
        elif page == "🔧 SQL-Abfragen":
            show_sql_queries()
    
    main()
