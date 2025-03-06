import pandas as pd

def parse_date(s):
    try:
        return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S%z').tz_convert(None)  # Zeitzone entfernen
    except ValueError:
        return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S').tz_localize(None)  # Sicherstellen, dass kein UTC-Offset bleibt

# Excel-Datei einlesen – Pfad anpassen
df = pd.read_excel('data/articles_export.xlsx')

# Datum konvertieren & Zeitzone entfernen
df['date'] = df['date'].apply(parse_date)

# Nur das Datum (ohne Uhrzeit) extrahieren
df['day'] = df['date'].dt.date

# Autoren aufsplitten (auch wenn mehrere in einer Zelle sind)
df['author'] = df['author'].fillna("Unbekannt").astype(str)
df = df.assign(author=df['author'].str.split(', ')).explode('author')

# Alle Kombinationen aus Autor und Tag speichern
df_result = df[['author', 'day']]

# Neue Excel-Datei speichern
df_result.to_excel('author_days_full.xlsx', index=False)

print("Datei erfolgreich gespeichert als 'author_days_full.xlsx'")
