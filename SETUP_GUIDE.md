# 🗞️ News Mining System - Setup Guide

This guide explains the new centralized configuration and database initialization system.

## 🎯 Key Improvements

### 1. Centralized Configuration
- **Single `.env` file** in the root directory
- All modules (heise, chip, visualization) load from the same configuration
- No more duplicate `.env` files in subdirectories

### 2. Automatic Database Initialization
- **`init_database.py`** automatically creates all required tables and columns
- Runs automatically when you start `heise/main.py` or `chip/main.py`
- Handles schema updates without manual intervention

### 3. Simplified Streamlit Dashboard
- **Cleaner sidebar** with 6 main pages instead of 10
- **Metrics moved to main area** for better visibility
- **Auto-refresh functionality** with configurable intervals (30s, 60s, 120s, 300s)
- **Faster data updates** with 60-second cache TTL

### 4. English Language
- All UI text, comments, and function names are now in English
- Consistent terminology across the application

## 📋 Quick Start

### 1. Configure Environment

Create or edit `.env` file in the **root directory**:

```bash
# Database Configuration (default values)
DB_NAME=datamining
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Email Notifications (optional)
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_email_password
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
ALERT_EMAIL=alert_recipient@example.com

# Google AI (optional, for Streamlit AI features)
GOOGLE_API_KEY=your_google_api_key
```

**Note:** A `.env.example` file is provided as a template.

### 2. Start Crawlers

The database will be automatically initialized on first run:

```bash
# Start Heise crawler
cd heise
python3 main.py

# Or start Chip crawler
cd chip
python3 main.py
```

On first run, you'll see:
```
[INFO] Initializing database...
[SUCCESS] Table 'heise' initialized successfully
[SUCCESS] Table 'chip' initialized successfully
[SUCCESS] Crawl state tables initialized successfully
[SUCCESS] Database initialization completed successfully!
```

### 3. Launch Streamlit Dashboard

```bash
cd visualization
streamlit run streamlit_app.py
```

## 🔄 Auto-Refresh Feature

The Streamlit dashboard now includes automatic data refresh:

1. In the sidebar, check **"Enable Auto-Refresh"**
2. Select your preferred interval (30s, 60s, 120s, or 300s)
3. The dashboard will automatically reload data at the specified interval

You can also manually refresh data using the **"🔄 Refresh Data"** button.

## 📊 Simplified Navigation

The sidebar now has 6 main pages:

1. **📊 Dashboard** - Overview with key metrics and charts
2. **📈 Time Analysis** - Temporal trends and patterns
3. **🔑 Keywords** - Keyword frequency and analysis
4. **🔍 Search** - Search and filter articles
5. **🕸️ Author Network** - Visual relationship graphs
6. **🔧 SQL Queries** - Custom database queries

Key metrics (total articles, authors, categories, database status) are now displayed in the main content area instead of cluttering the sidebar.

## 🗄️ Database Schema

### Heise Table
```sql
CREATE TABLE heise (
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
);
```

### Chip Table
```sql
CREATE TABLE chip (
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
);
```

### Crawl State Tables
```sql
CREATE TABLE heise_crawl_state (
    id SERIAL PRIMARY KEY,
    year INTEGER,
    month INTEGER,
    article_index INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chip_crawl_state (
    id SERIAL PRIMARY KEY,
    sitemap_index INTEGER,
    article_index INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 Manual Database Initialization

If you need to manually initialize the database:

```bash
python3 init_database.py
```

This is useful for:
- Setting up the database before running crawlers
- Adding missing columns to existing tables
- Verifying database configuration

## 📁 Project Structure

```
datamining/
├── .env                      # ⭐ Centralized configuration
├── .env.example              # Template for configuration
├── init_database.py          # ⭐ Database initialization module
├── heise/
│   ├── main.py              # Heise crawler (auto-init DB)
│   ├── notification.py      # Loads .env from root
│   └── ...
├── chip/
│   ├── main.py              # Chip crawler (auto-init DB)
│   ├── notification.py      # Loads .env from root
│   └── ...
└── visualization/
    └── streamlit_app.py     # Dashboard (loads .env from root)
```

## ⚡ Performance Tips

1. **Cache Duration**: Data is cached for 60 seconds by default. For more frequent updates, the auto-refresh feature will bypass cache.

2. **Database Connection**: The dashboard checks database connectivity and displays status in real-time.

3. **Filter Data**: Use the "Data Source" filter in sidebar to focus on specific sources (Heise or Chip).

## 🔍 Troubleshooting

### Database Connection Errors

If you see "No data available. Check database connection":

1. Verify PostgreSQL is running:
   ```bash
   sudo systemctl status postgresql
   ```

2. Check `.env` configuration:
   ```bash
   cat .env
   ```

3. Test database connection:
   ```bash
   python3 init_database.py
   ```

### Import Errors

If you get `ModuleNotFoundError`:

```bash
pip install -r requirements.txt
```

For Streamlit-specific dependencies:
```bash
pip install -r visualization/requirements_streamlit.txt
```

### Auto-Refresh Not Working

1. Make sure "Enable Auto-Refresh" is checked in the sidebar
2. The page will reload after the selected interval
3. If using a very short interval (30s), be aware of database load

## 🆕 Migration from Old Setup

If you have existing `.env` files in subdirectories:

1. **Merge all `.env` files** into a single `.env` in the root directory
2. **Remove old `.env` files** from `heise/`, `chip/`, and other subdirectories
3. **Restart all services** to use the new configuration

The system will automatically use the root `.env` file.

## 📝 Next Steps

1. ✅ Configure `.env` with your database credentials
2. ✅ Run database initialization (automatic on first crawler start)
3. ✅ Start crawlers to collect data
4. ✅ Launch Streamlit dashboard to visualize data
5. ✅ Enable auto-refresh for real-time monitoring

## 🤝 Contributing

When adding new features:
- Use English for all code, comments, and documentation
- Load configuration from root `.env` file
- Update `init_database.py` if adding new tables/columns
- Keep the sidebar minimal and focused

## 📖 Additional Resources

- [Quick Start Guide](QUICKSTART.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Docker Setup](DOCKER_SETUP.md)
- [Security Best Practices](SECURITY.md)
