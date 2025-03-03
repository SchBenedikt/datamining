import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv  # hinzugefügt

load_dotenv()  # hinzugefügt

def export_articles(engine, export_format):
    query = "SELECT * FROM articles"
    df = pd.read_sql_query(query, engine)

    if export_format == "excel":
        output_file = "articles_export.xlsx"
        df.to_excel(output_file, index=False)
    elif export_format == "csv":
        output_file = "articles_export.csv"
        df.to_csv(output_file, index=False)
    elif export_format == "json":
        output_file = "articles_export.json"
        df.to_json(output_file, orient="records")
    else:
        print(f"Unsupported export format: {export_format}")
        sys.exit(1)
    print(f"Exported articles table to {output_file}")

if __name__ == '__main__':
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)

    print("Export der Artikeltabelle")
    print("Wählen Sie das Exportformat für die Artikeltabelle:")
    print("1: Excel")
    print("2: CSV")
    print("3: JSON")
    format_choice = input("Bitte geben Sie 1, 2 oder 3 ein: ").strip()
    if format_choice == "1":
        export_format = "excel"
    elif format_choice == "2":
        export_format = "csv"
    elif format_choice == "3":
        export_format = "json"
    else:
        print("Ungültige Auswahl beim Format. Es wird Excel verwendet.")
        export_format = "excel"
    export_articles(engine, export_format)
