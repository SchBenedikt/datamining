import os
import pandas as pd
from sqlalchemy import create_engine

# Erstelle einen connection string über SQLAlchemy
db_name = os.getenv('DB_NAME', 'web_crawler')
db_user = os.getenv('DB_USER', 'schaechner')
db_password = os.getenv('DB_PASSWORD', 'SchaechnerServer')
db_host = os.getenv('DB_HOST', '192.168.188.36')
db_port = os.getenv('DB_PORT', '6543')

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

# Lese die Tabelle "articles" in ein DataFrame ein
query = "SELECT * FROM articles"
df = pd.read_sql_query(query, engine)

# Exportiere das DataFrame als Excel-Datei
output_file = "articles_export.xlsx"
df.to_excel(output_file, index=False)
print(f"Exported articles table to {output_file}")
