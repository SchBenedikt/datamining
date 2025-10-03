"""
Database Initialization Module

This module ensures that all required database tables and columns exist
before the application starts. It is automatically called when main.py
is executed for the first time or when database schema is missing.
"""

import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from root .env file
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME', 'datamining'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

def print_status(message, level="INFO"):
    """Outputs a colored status message with current date and time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": "\033[92m",
        "SUCCESS": "\033[94m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{now} {colors.get(level, colors['INFO'])}[{level}] {message}{colors['RESET']}")

def connect_db():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        print_status(f"Database connection error: {e}", "ERROR")
        raise

def init_heise_table(conn):
    """Creates or updates the heise table with all required columns."""
    try:
        with conn.cursor() as cur:
            # Create table if it doesn't exist
            cur.execute('''
                CREATE TABLE IF NOT EXISTS heise (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    url TEXT UNIQUE,
                    date TEXT,
                    author TEXT,
                    category TEXT,
                    keywords TEXT,
                    word_count INTEGER,
                    editor_abbr TEXT,
                    site_name TEXT
                )
            ''')
            conn.commit()
            print_status("Table 'heise' initialized successfully", "SUCCESS")
            
            # Check for existing columns
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'heise'
            """)
            existing_columns = {row[0] for row in cur.fetchall()}
            
            # Add missing columns if needed
            required_columns = {
                'title': 'TEXT',
                'url': 'TEXT UNIQUE',
                'date': 'TEXT',
                'author': 'TEXT',
                'category': 'TEXT',
                'keywords': 'TEXT',
                'word_count': 'INTEGER',
                'editor_abbr': 'TEXT',
                'site_name': 'TEXT'
            }
            
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    # Skip UNIQUE constraint for ALTER TABLE
                    col_type_clean = col_type.replace(' UNIQUE', '')
                    cur.execute(f'ALTER TABLE heise ADD COLUMN "{col_name}" {col_type_clean}')
                    conn.commit()
                    print_status(f"Added column '{col_name}' to heise table", "INFO")
                    
    except Exception as e:
        print_status(f"Error initializing heise table: {e}", "ERROR")
        conn.rollback()
        raise

def init_chip_table(conn):
    """Creates or updates the chip table with all required columns."""
    try:
        with conn.cursor() as cur:
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chip (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    author TEXT,
                    date TEXT,
                    keywords TEXT,
                    description TEXT,
                    type TEXT,
                    page_level1 TEXT,
                    page_level2 TEXT,
                    page_level3 TEXT,
                    page_template TEXT
                )
            """)
            conn.commit()
            print_status("Table 'chip' initialized successfully", "SUCCESS")
            
            # Check for existing columns
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'chip'
            """)
            existing_columns = {row[0] for row in cur.fetchall()}
            
            # Add missing columns if needed
            required_columns = {
                'url': 'TEXT UNIQUE',
                'title': 'TEXT',
                'author': 'TEXT',
                'date': 'TEXT',
                'keywords': 'TEXT',
                'description': 'TEXT',
                'type': 'TEXT',
                'page_level1': 'TEXT',
                'page_level2': 'TEXT',
                'page_level3': 'TEXT',
                'page_template': 'TEXT'
            }
            
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    # Skip UNIQUE constraint for ALTER TABLE
                    col_type_clean = col_type.replace(' UNIQUE', '')
                    cur.execute(f'ALTER TABLE chip ADD COLUMN "{col_name}" {col_type_clean}')
                    conn.commit()
                    print_status(f"Added column '{col_name}' to chip table", "INFO")
                    
    except Exception as e:
        print_status(f"Error initializing chip table: {e}", "ERROR")
        conn.rollback()
        raise

def init_crawl_state_tables(conn):
    """Creates crawl state tables for tracking crawler progress."""
    try:
        with conn.cursor() as cur:
            # Heise crawl state
            cur.execute("""
                CREATE TABLE IF NOT EXISTS heise_crawl_state (
                    id SERIAL PRIMARY KEY,
                    year INTEGER,
                    month INTEGER,
                    article_index INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chip crawl state
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chip_crawl_state (
                    id SERIAL PRIMARY KEY,
                    sitemap_index INTEGER,
                    article_index INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            print_status("Crawl state tables initialized successfully", "SUCCESS")
            
    except Exception as e:
        print_status(f"Error initializing crawl state tables: {e}", "ERROR")
        conn.rollback()
        raise

def ensure_language_columns(conn, table_name, languages):
    """
    Ensures that for each language in 'languages' list,
    a corresponding column exists in the specified table.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """)
            existing_columns = {row[0] for row in cur.fetchall()}
        
        for lang in languages:
            if lang not in existing_columns:
                with conn.cursor() as cur:
                    cur.execute(f'ALTER TABLE {table_name} ADD COLUMN "{lang}" TEXT')
                conn.commit()
                print_status(f"Added language column '{lang}' to {table_name} table", "INFO")
                
    except Exception as e:
        print_status(f"Error adding language columns to {table_name}: {e}", "ERROR")
        conn.rollback()
        raise

def initialize_database(languages=None):
    """
    Main function to initialize all database tables and columns.
    This function is called automatically when main.py starts.
    
    Args:
        languages: Optional list of language codes for translation columns
    """
    try:
        print_status("Starting database initialization...", "INFO")
        
        conn = connect_db()
        
        # Initialize main tables
        init_heise_table(conn)
        init_chip_table(conn)
        
        # Initialize crawl state tables
        init_crawl_state_tables(conn)
        
        # Add language columns if specified
        if languages:
            ensure_language_columns(conn, 'heise', languages)
            ensure_language_columns(conn, 'chip', languages)
        
        conn.close()
        print_status("Database initialization completed successfully!", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Database initialization failed: {e}", "ERROR")
        return False

if __name__ == "__main__":
    # Run initialization when executed directly
    initialize_database()
