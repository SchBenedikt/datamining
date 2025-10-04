
# 🏗️ System Architecture

## Overview

The Unified News Mining System is a fully integrated crawler system with separate database tables, a unified dashboard, and centralized management via Docker.

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED NEWS MINING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐              ┌──────────────────────┐
│   Heise Crawlers     │              │   Chip Crawlers      │
├──────────────────────┤              ├──────────────────────┤
│ Archive Crawler      │              │ Archive Crawler      │
│ (backwards)          │              │ (forwards)           │
│ - Start: 2025/10     │              │ - Start: Page 1      │
│ - Target: 2000/01    │              │ - Target: Last Page  │
│                      │              │                      │
│ Live Crawler         │              │ Live Crawler         │
│ (every 5 minutes)    │              │ (every 10 minutes)   │
│ - Checks: Current    │              │ - Checks: Page 1     │
│   Month              │              │   (newest)           │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           │ INSERT INTO heise                   │ INSERT INTO chip
           │ (title, url, date,                  │ (title, url, date,
           │  author, category,                  │  author, keywords,
           │  keywords, ...)                     │  description, ...)
           │                                     │
           └─────────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   PostgreSQL DB      │
              ├──────────────────────┤
              │                      │
              │  ┌────────────────┐  │
              │  │  heise table   │  │
              │  │  (10 columns)  │  │
              │  └────────────────┘  │
              │                      │
              │  ┌────────────────┐  │
              │  │  chip table    │  │
              │  │  (12 columns)  │  │
              │  └────────────────┘  │
              │                      │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
         ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Streamlit   │  │ Discord Bot │  │ Export      │
│ Dashboard   │  │             │  │ Tools       │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ • Merge     │  │ • Heise     │  │ heise/      │
│   both      │  │   stats     │  │ export_     │
│   tables    │  │             │  │ articles.py │
│             │  │ • Chip      │  │             │
│ • Filter:   │  │   stats     │  │ chip/       │
│   - Source  │  │             │  │ export_     │
│   - Date    │  │ • Today     │  │ articles.py │
│   - Author  │  │   & Total   │  │             │
│   - Cat.    │  │             │  │ Formats:    │
│             │  │ • Updates   │  │ • CSV       │
│ • 20+       │  │   every 10  │  │ • XLSX      │
│   Viz.      │  │   minutes   │  │ • JSON      │
│             │  │             │  │ • SQL       │
│ • Export    │  │             │  │             │
│ • AI        │  │             │  │             │
│   Analytics │  │             │  │             │
└─────────────┘  └─────────────┘  └─────────────┘┌─────────────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE STACK                         │
├─────────────────────────────────────────────────────────────────┤
│  heise-archive-crawler   │  heise-live-crawler                  │
│  chip-archive-crawler    │  chip-live-crawler                   │
│  streamlit-dashboard     │  discord-bot                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  DOCKER MANAGEMENT TOOLS (Optional)              │
├─────────────────────────────────────────────────────────────────┤
│  Portainer (Port 9000)   │  Dockge (Port 5001)                  │
│  - Container starten/stoppen/pausieren                          │
│  - Logs in Echtzeit ansehen                                     │
│  - Ressourcen-Monitoring                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### 1. Crawling Phase

```
Heise Archive Crawler:
  https://www.heise.de/newsticker/archiv/2025/10
  └─> Extracts article metadata
      └─> Saves to heise table
          └─> Goes to 2025/09, 2025/08, ...

Heise Live Crawler:
  Every 5 minutes:
  └─> Checks current month
      └─> Finds new articles
          └─> Saves only new ones (duplicate check via URL)

Chip Archive Crawler:
  https://www.chip.de/news/?p=1
  └─> Extracts article metadata
      └─> Saves to chip table
          └─> Goes to page 2, 3, 4, ...

Chip Live Crawler:
  Every 10 minutes:
  └─> Checks page 1 (newest articles)
      └─> Finds new articles
          └─> Saves only new ones (duplicate check via URL)
```

### 2. Database Phase

```
PostgreSQL Database:
  ├─> heise table
  │   ├─> id, title, url, date, author
  │   ├─> category, keywords, word_count
  │   └─> editor_abbr, site_name
  │
  └─> chip table
      ├─> id, url, title, author, date
      ├─> keywords, description, type
      └─> page_level1, page_level2, page_level3, page_template
```

### 3. Visualization Phase

```
Streamlit Dashboard:
  └─> SELECT * FROM heise
  └─> SELECT * FROM chip
      └─> pd.concat([df_heise, df_chip])
          └─> Filter by source
              └─> Visualizations:
                  ├─> Author networks
                  ├─> Keyword analyses
                  ├─> Time analyses
                  ├─> AI analyses
                  └─> Export functions
```

### 4. Notification Phase

```
Discord Bot:
  └─> Every 10 minutes:
      ├─> SELECT COUNT(*) FROM heise
      ├─> SELECT COUNT(*) FROM chip
      └─> Posts statistics in Discord channel

Email Notifications:
  └─> On errors:
      └─> Sends alert to ALERT_EMAIL
```

---

## 📊 Database Schema

### Heise Table

```sql
CREATE TABLE IF NOT EXISTS heise (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    date TEXT,
    author TEXT,
    category TEXT,
    keywords TEXT,
    word_count INTEGER,
    editor_abbr TEXT,
    site_name TEXT
);

CREATE INDEX idx_heise_date ON heise(date);
CREATE INDEX idx_heise_author ON heise(author);
CREATE INDEX idx_heise_category ON heise(category);
```

**Example Data:**
```json
{
  "id": 1,
  "title": "New AI Technology revolutionizes...",
  "url": "https://www.heise.de/news/...",
  "date": "2025-10-02T10:30:00",
  "author": "Max Mustermann",
  "category": "Artificial Intelligence",
  "keywords": "AI, Machine Learning, Innovation",
  "word_count": 450,
  "editor_abbr": "mm",
  "site_name": "heise online"
}
```

### Chip Table

```sql
CREATE TABLE IF NOT EXISTS chip (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
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

CREATE INDEX idx_chip_date ON chip(date);
CREATE INDEX idx_chip_author ON chip(author);
CREATE INDEX idx_chip_type ON chip(type);
```

**Example Data:**
```json
{
  "id": 1,
  "url": "https://www.chip.de/news/...",
  "title": "Smartphone Test 2025: The best...",
  "author": "CHIP Editorial Team",
  "date": "2025-10-02",
  "keywords": "Smartphone, Test, Comparison",
  "description": "In the big comparison test...",
  "type": "Test",
  "page_level1": "News",
  "page_level2": "Mobile",
  "page_level3": "Smartphones",
  "page_template": "article"
}
```

---

## 🔧 Component Details

### Heise Crawler

**File:** `heise/main.py` (Archive), `heise/current_crawler.py` (Live)

**Functionality:**
1. Loads archive page: `https://www.heise.de/newsticker/archiv/YYYY/MM`
2. Parses HTML with BeautifulSoup
3. Extracts article links and metadata
4. Checks for duplicates via URL
5. Saves new articles to `heise` table
6. If < 10 articles/day: Email alert

**Features:**
- Backward crawling (newest to oldest)
- Live crawler only checks current month
- Recognizes editor abbreviations (e.g., "mm", "js")
- Captures word count

### Chip Crawler

**File:** `chip/main.py` (Archive), `chip/current_crawler.py` (Live)

**Functionality:**
1. Loads news page: `https://www.chip.de/news/?p=PAGE`
2. Parses HTML with BeautifulSoup
3. Extracts article links and metadata from `<script type="application/ld+json">`
4. Checks for duplicates via URL
5. Saves new articles to `chip` table

**Features:**
- Forward crawling (page 1 to page N)
- Live crawler only checks page 1
- Extracts structured data (JSON-LD)
- Captures page hierarchy (Level 1-3)

### Streamlit Dashboard

**File:** `visualization/streamlit_app.py`

**Functionality:**
1. Loads data from both tables
2. Adds `source` column ('heise' or 'chip')
3. Merges DataFrames: `pd.concat([df_heise, df_chip])`
4. Provides filter options in sidebar
5. Generates visualizations on-the-fly
6. Caches data for performance

**Features:**
- **Overview:** KPIs, Statistics, Trends
- **Time Analysis:** Articles per day/week/month
- **Author Networks:** NetworkX + Plotly
- **Keyword Analysis:** Top Keywords, Trends
- **Word Clouds:** Most frequent terms
- **AI Analysis:** Topic Modeling, Sentiment
- **Search Function:** Full-text search
- **Export:** CSV, Excel, JSON, SQL

### Discord Bot

**File:** `heise/bot.py`

**Functionality:**
1. Connects to Discord
2. Every 10 minutes:
   - Counts articles in both tables
   - Counts today's articles
   - Counts authors
3. Posts embed message with statistics

**Output:**
```
📊 News Mining Statistics

📰 Articles today: 45 (Heise: 25, Chip: 20)
📚 Articles total: 12,345 (Heise: 8,000, Chip: 4,345)
✍️ Authors total: 234

As of: 10/02/2025 10:30
```

### Export Tools

**Files:** `heise/export_articles.py`, `chip/export_articles.py`

**Functionality:**
1. Connects to database
2. Reads all articles from respective table
3. Converts to desired format
4. Saves to `data/` directory

**Formats:**
- **CSV:** `data/heise_articles_YYYYMMDD.csv`
- **Excel:** `data/heise_articles_YYYYMMDD.xlsx`
- **JSON:** `data/heise_articles_YYYYMMDD.json`
- **SQL:** `data/heise_articles_YYYYMMDD.sql`

---

## 🐳 Docker Architecture

### Docker Compose Services

```yaml
services:
  heise-archive-crawler:
    - Runs heise/main.py
    - Backward crawling
    - Restart: unless-stopped
    
  heise-live-crawler:
    - Runs heise/current_crawler.py
    - Checks every 5 minutes
    - Restart: unless-stopped
    
  chip-archive-crawler:
    - Runs chip/main.py
    - Forward crawling
    - Restart: unless-stopped
    
  chip-live-crawler:
    - Runs chip/current_crawler.py
    - Checks every 10 minutes
    - Restart: unless-stopped
    
  streamlit-dashboard:
    - Runs streamlit run
    - Port 8501 exposed
    - Volumes for code updates
    
  discord-bot:
    - Runs heise/bot.py
    - Posts every 10 minutes
    - Restart: unless-stopped
```

### Docker Network

```
crawler-network (bridge):
  ├─> heise-archive-crawler
  ├─> heise-live-crawler
  ├─> chip-archive-crawler
  ├─> chip-live-crawler
  ├─> streamlit-dashboard
  └─> discord-bot
```

All containers can reach each other via this network and share the same `.env` file.

---

## 🔐 Security

### Environment Variables

Sensitive data is managed via `.env` file:
- Never commit to Git (`.gitignore`)
- Read-only for containers
- Encrypted transmission (SMTP SSL/TLS)

### Database Security

- PostgreSQL access only via credentials
- Unique constraints prevent duplicates
- Prepared statements against SQL injection
- Index on frequently queried columns

### API Security

- No authentication (local access)
- For public deployment: OAuth/JWT recommended
- Rate limiting for API endpoints

---

## 📈 Scalability

### Horizontal Scaling

**Adding more sources:**
1. Create new folder (e.g., `golem/`)
2. Copy and adapt crawler scripts
3. Create new table in DB
4. Add service to `docker-compose.yml`
5. Streamlit automatically loads new table

**Example:**
```yaml
golem-live-crawler:
  build: .
  container_name: golem-live-crawler
  command: python3 golem/current_crawler.py
  ...
```

### Vertical Scaling

**Performance Optimizations:**
- Database indexes on frequently queried columns
- Streamlit caching for large datasets
- Batch inserts instead of individual INSERTs
- Connection pooling for database

### Load Balancing

**Under high load:**
- Multiple Streamlit instances behind Nginx
- PostgreSQL read replicas
- Redis for session management
- CDN for static assets

---

## 🔄 Extensibility

### Plugin Architecture

The system is modularly structured:

```
plugins/
├── crawlers/
│   ├── heise_crawler.py
│   ├── chip_crawler.py
│   └── custom_crawler.py  <- New crawler
│
├── exporters/
│   ├── csv_exporter.py
│   ├── json_exporter.py
│   └── pdf_exporter.py    <- New exporter
│
└── visualizations/
    ├── network_graph.py
    ├── time_series.py
    └── custom_viz.py      <- New visualization
```

### API Endpoints

**Existing:**
- `/stats` - Overall statistics
- `/articles` - All articles

**Extensible:**
- `/api/v1/heise/articles` - Heise only
- `/api/v1/chip/articles` - Chip only
- `/api/v1/search?q=keyword` - Search
- `/api/v1/authors` - Author list
- `/api/v1/keywords` - Keyword trends

---

## 🎯 Best Practices

### Crawler

1. **Rate Limiting:** Pause between requests (1-2 seconds)
2. **User-Agent:** Identifiable as bot
3. **Robots.txt:** Respect crawling rules
4. **Error Handling:** Graceful degradation on errors
5. **Logging:** Detailed logs for debugging

### Database

1. **Normalization:** Separate tables for better performance
2. **Indexes:** On frequently queried columns
3. **Backups:** Regular database backups
4. **Constraints:** UNIQUE on URL prevents duplicates
5. **Transactions:** Utilize ACID properties

### Streamlit

1. **Caching:** `@st.cache_data` for expensive operations
2. **Lazy Loading:** Load large datasets only when needed
3. **Pagination:** For very many articles
4. **Responsive:** Mobile-friendly layout
5. **Error Handling:** Try-Except for all DB queries

---

## 🛠️ Monitoring & Debugging

### Logs

```bash
# Docker Logs
docker-compose logs -f [service-name]

# Specific Crawler
docker-compose logs -f heise-live-crawler

# All Services
docker-compose logs -f
```

### Metrics

**Important KPIs:**
- Articles per day
- Crawler success rate
- Duplicate detection rate
- API response time
- Streamlit load time

### Alerts

**Email notifications for:**
- Less than 10 articles/day
- Database connection errors
- Crawler crashes
- Disk space < 10%

---

## 🚀 Deployment Options

### Option 1: Local Deployment

```bash
# Start crawlers manually
python3 heise/main.py
python3 chip/main.py

# Start Streamlit
streamlit run visualization/streamlit_app.py
```

### Option 2: Docker Deployment

```bash
# Start all services
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

**Recommended Architecture:**
- EC2/Compute Engine/VM for containers
- RDS/Cloud SQL/Azure DB for PostgreSQL
- CloudWatch/Logging for monitoring
- S3/Cloud Storage for exports
- Load Balancer for Streamlit

---

## 📚 Further Documentation

- **[README.md](README.md)** - Main documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker setup details
- **[SECURITY.md](SECURITY.md)** - Security guidelines

---

## 🤝 Contributions

Contributions are welcome! Please open an issue or pull request on GitHub.

**Contribution Guidelines:**
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

---

**As of:** October 2025  
**Version:** 2.0 (Separate Tables Architecture)  
**Status:** ✅ Production Ready

````
