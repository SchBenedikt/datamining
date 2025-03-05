import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objs as go

# Funktion zum Parsen der Datumsangaben
def parse_date(s):
    try:
        return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S').tz_localize('UTC')

# Daten laden
df = pd.read_excel("data/articles_export.xlsx")

# Konvertiere die 'date' Spalte in datetime
df['date'] = df['date'].apply(parse_date)

# Force conversion to datetimelike type
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')

# Extrahiere Datum und Uhrzeit
df['date_only'] = df['date'].dt.date
df['hour'] = df['date'].dt.hour

# Zähle die Anzahl der Artikel pro Stunde und pro Tag
hourly_counts = df.groupby(['date_only', 'hour']).size().reset_index(name='count')

# Berechne die durchschnittliche Artikelzahl pro Stunde
avg_hourly = hourly_counts.groupby('hour')['count'].mean().reset_index()

# Bestimme die Stunde mit den meisten Artikeln
peak_hour = avg_hourly.loc[avg_hourly['count'].idxmax()]

# Erstelle die Visualisierung
trace = go.Bar(
    x=avg_hourly['hour'],
    y=avg_hourly['count'],
    marker=dict(color='skyblue')
)

# Füge die Linie für die Peak-Stunde hinzu
peak_line = go.layout.Shape(
    type='line',
    x0=peak_hour['hour'],
    y0=0,
    x1=peak_hour['hour'],
    y1=max(avg_hourly['count']),
    line=dict(color='red', width=2, dash='dash')
)

layout = go.Layout(
    title="Durchschnittliche Artikelzahl pro Stunde im Jahr",
    xaxis=dict(title="Uhrzeit", tickvals=list(range(0, 24))),
    yaxis=dict(title="Durchschnittliche Artikelzahl pro Tag"),
    shapes=[peak_line]
)

# Starte die Dash-App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Artikelanalyse", style={'textAlign': 'center'}),
    dcc.Graph(
        id='hourly-article-graph',
        figure={
            'data': [trace],
            'layout': layout
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
