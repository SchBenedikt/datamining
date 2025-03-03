import os
import sys
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

load_dotenv()

def export_articles(engine, export_format):
    if export_format == "excel":
        query = "SELECT * FROM articles"
        df = pd.read_sql_query(query, engine)
        output_file = "articles_export.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Exported articles table to {output_file}")
    elif export_format == "csv":
        query = "SELECT * FROM articles"
        df = pd.read_sql_query(query, engine)
        output_file = "articles_export.csv"
        df.to_csv(output_file, index=False)
        print(f"Exported articles table to {output_file}")
    elif export_format == "json":
        query = "SELECT * FROM articles"
        df = pd.read_sql_query(query, engine)
        output_file = "articles_export.json"
        df.to_json(output_file, orient="records")
        print(f"Exported articles table to {output_file}")
    elif export_format == "sql":
        # Exportiere alle Tabellen der Datenbank
        output_file = "database_export.sql"
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        with open(output_file, "w", encoding="utf-8") as f:
            for table in table_names:
                df = pd.read_sql_query(f"SELECT * FROM {table}", engine)
                columns = df.columns.tolist()
                f.write(f"-- Table: {table} --\n")
                for _, row in df.iterrows():
                    values = []
                    for value in row:
                        if pd.isnull(value):
                            values.append("NULL")
                        else:
                            val = str(value).replace("'", "''")
                            values.append(f"'{val}'")
                    columns_str = ", ".join(columns)
                    values_str = ", ".join(values)
                    statement = f"INSERT INTO {table} ({columns_str}) VALUES ({values_str});\n"
                    f.write(statement)
        print(f"Exported entire database to {output_file}")
    else:
        print(f"Unsupported export format: {export_format}")
        sys.exit(1)

if __name__ == '__main__':
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)

    print("Export der Artikeltabelle")
    print("Wählen Sie das Exportformat:")
    print("1: Excel")
    print("2: CSV")
    print("3: JSON")
    print("4: SQL (gesamte Datenbank)")
    format_choice = input("Bitte geben Sie 1, 2, 3 oder 4 ein: ").strip()
    if format_choice == "1":
        export_format = "excel"
    elif format_choice == "2":
        export_format = "csv"
    elif format_choice == "3":
        export_format = "json"
    elif format_choice == "4":
        export_format = "sql"
    else:
        print("Ungültige Auswahl. Es wird Excel verwendet.")
        export_format = "excel"

    export_articles(engine, export_format)
