<a href="https://discord.com/invite/Q6Nn2z3tUP">
  <img src="https://discord.com/api/guilds/1346160903304773703/widget.png?style=banner2" width="250"/>
</a>
<a href="https://deepnote.com/app/schachner/Web-Crawler-d5025a36-3829-4c12-ad2d-b81aa84bd217?utm_source=app-settings&utm_medium=product-embed&utm_campaign=data-app&utm_content=d5025a36-3829-4c12-ad2d-b81aa84bd217&__embedded=true">
  <img src="https://img.shields.io/badge/Open%20in-Deepnote-blue?style=for-the-badge&logo=deepnote" width="250"/>
</a>

# 🗞️ Unified News Mining System

A comprehensive, unified crawler system for collecting and analyzing news articles from **Heise.de** and **Chip.de**.

---

## 📑 Table of Contents

- [🎯 Quick Links](#-quick-links)
- [✨ Key Features](#-key-features)
- [🌍 Purpose & Functionality](#-purpose--functionality)
- [🚀 Installation & Setup](#-installation--setup)
  - [Prerequisites](#1️⃣-prerequisites)
  - [Clone Repository](#2️⃣-clone-repository)
  - [Install Dependencies](#3️⃣-install-dependencies)
  - [Configure Environment Variables](#4️⃣-configure-environment-variables)
  - [Database Setup](#5️⃣-database-setup)
- [🛠 Usage](#-usage)
  - [Start Crawlers](#start-crawlers)
  - [Streamlit Dashboard](#streamlit-dashboard)
  - [Discord Bot](#discord-bot)
  - [API Endpoints](#api-endpoints)
  - [Export Data](#export-data)
- [🐳 Docker Deployment](#-docker-deployment)
- [🏗 Database Schema](#-database-schema)
- [📊 Streamlit Features](#-streamlit-features)
- [📂 Project Structure](#-project-structure)
- [🔧 Management with Docker Tools](#-management-with-docker-tools)
- [❗ Troubleshooting](#-troubleshooting)
- [🗂️ Examples & Screenshots](#️-examples--screenshots)
- [📜 License](#-license)
- [🙋 About Us](#-about-us)

---

## 🎯 Quick Links

- 📖 **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- ⚙️ **[Setup Guide](SETUP_GUIDE.md)** - New centralized configuration and auto-refresh features
- 🏗️ **[Architecture](ARCHITECTURE.md)** - System architecture and data flow
- 🐳 **[Docker Setup](DOCKER_SETUP.md)** - Deployment with Docker


---

## 🌍 Purpose & Functionality

The **News Mining System** is designed to automatically extract and store news articles from multiple sources. The main objectives are:

- 📡 **Data Collection** - Capture historical news articles from Heise.de and Chip.de
- 🏛 **Structured Storage** - Articles from both sources in separate PostgreSQL tables
- 🔍 **Metadata Extraction** - Capture title, author, category, keywords, word count and more
- 🔄 **Incremental Crawling** - Duplicate detection and storage of only new articles
- 🔔 **Notifications** - Email notifications for errors during the crawling process
- 🎨 **Enhanced Terminal Output** - Use of PyFiglet for better readability
- 📤 **Data Export** - Export as CSV, JSON, XLSX with source filtering
- 🖥 **API** - Provision of statistics and complete datasets
- 📈 **Analytics** - Detailed analysis of authors, categories and time trends
- 🔍 **Article Search** - Search all articles with advanced filter options
- 🎯 **Unified Dashboard** - One Streamlit application for both sources
- 🤖 **Discord Bot** - Real-time statistics for both sources in Discord
- 📊 **Extensive Visualizations** - Over 20 different charts, graphs and representations
- 🕸️ **Author Networks** - Visualization of connections between authors
- 📈 **Trend Analysis** - Time-based analysis and predictions

An API endpoint is also provided that can display the crawled data and statistics.

---

## 🚀 Installation & Setup

### 1️⃣ Prerequisites

🔹 **Python 3.8+** (recommended: Python 3.11)

🔹 **PostgreSQL 13+** (local or remote)

🔹 **Git** (for cloning the repository)

🔹 **pip3** (Python Package Manager)

Optional:
- 🐳 **Docker & Docker Compose** (for containerized deployment)
- 🎮 **Discord Bot Token** (for Discord integration)
- 🤖 **Google API Key** (for AI analysis)

### 2️⃣ Clone Repository

```bash
git clone https://github.com/SchBenedikt/datamining.git
cd datamining
```

### 3️⃣ Install Dependencies

Install all required Python libraries:

```bash
pip3 install -r requirements.txt
```

For the Streamlit application (advanced visualizations):

```bash
cd visualization
pip3 install -r requirements_streamlit.txt
cd ..
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432

# Email Notifications (optional)
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ALERT_EMAIL=recipient@example.com

# Discord Bot (optional)
DISCORD_TOKEN=your_discord_bot_token
CHANNEL_ID=your_discord_channel_id

# Google AI (optional, for advanced analysis)
GOOGLE_API_KEY=your_google_api_key
```

**Notes:**
- For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833)
- Get Discord Token from the [Discord Developer Portal](https://discord.com/developers/applications)
- Create Google API Key in the [Google Cloud Console](https://console.cloud.google.com)

### 5️⃣ Database Setup

Create the PostgreSQL database:

```bash
# Open PostgreSQL console
psql -U postgres

# Create database
CREATE DATABASE your_database_name;

# Exit
\q
```

The required tables will be created automatically when the crawlers start for the first time.

**Manual table creation (optional):**

```sql
-- Heise-Tabelle
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
);

-- Chip-Tabelle
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
);
```

---

## 🛠 Usage

### Start Crawlers

#### Heise Archive Crawler (crawls backwards from newest to oldest)

```bash
cd heise
python3 main.py
```

**Example Terminal Output:**

```
[INFO] Crawling URL: https://www.heise.de/newsticker/archiv/2025/10
[INFO] Found articles (total): 55
2025-10-02 10:30:15 [INFO] Processing 16 articles for day 2025-10-02
2025-10-02 10:30:15 [INFO] 2025-10-02T20:00:00 - article-name
```

If fewer than 10 articles per day are found, an email will be sent.

#### Heise Live Crawler (checks every 5 minutes for new articles)

```bash
cd heise
python3 current_crawler.py
```

**Example Terminal Output:**

```
[INFO] Crawling URL: https://www.heise.de/newsticker/archiv/2025/10
[INFO] Found articles (total): 55
2025-10-02 10:35:00 [INFO] Current crawl cycle completed.
2025-10-02 10:35:00 [INFO] Waiting 300 seconds until next crawl.
```

#### Chip Archive Crawler (crawls from page 1 upwards)

```bash
cd chip
python3 main.py
```

#### Chip Live Crawler (checks every 10 minutes for new articles)

```bash
cd chip
python3 current_crawler.py
```

---

### Streamlit Dashboard

Start the interactive Streamlit dashboard with support for both sources:

```bash
cd visualization
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`.

---

### Discord Bot

Start the Discord bot for real-time statistics updates:

```bash
cd heise
python3 bot.py
```

**The bot provides:**
- Total article count for both sources
- Today's article count for both sources
- Author statistics
- Updates every 10 minutes

---

### API Endpoints

The API server starts automatically when running `heise/main.py`. Statistics can be retrieved here:

```
http://127.0.0.1:6600/stats
```

**Manual API start:**

```bash
cd heise
python3 api.py
```

---

### Export Data

You can export data for each source as CSV, JSON, or XLSX files.

**Export Heise articles:**

```bash
cd heise
python3 export_articles.py
```

**Export Chip articles:**

```bash
cd chip
python3 export_articles.py
```

Exported articles are saved in the `data/` directory.

---

## 🐳 Docker Deployment

### Start all services with one command

```bash
docker-compose up -d
```

### Manage individual services

```bash
# Start Heise Archive Crawler
docker-compose up -d heise-archive-crawler

# Start Chip Live Crawler
docker-compose up -d chip-live-crawler

# Start Streamlit Dashboard
docker-compose up -d streamlit-dashboard

# Start Discord Bot
docker-compose up -d discord-bot
```

### View logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f heise-live-crawler
```

### Stop services

```bash
# Stop all services
docker-compose down

# Specific service
docker-compose stop heise-archive-crawler
```

### Access Dashboard

After starting, the Streamlit dashboard is available at:

```
http://localhost:8501
```

---

## 🏗 Database Schema

The database uses **two separate tables** for better organization:

### Heise Table

| Column       | Type   | Description                 |
| ------------ | ------ | --------------------------- |
| id           | SERIAL | Unique ID                   |
| title        | TEXT   | Article title               |
| url          | TEXT   | Article URL (unique)        |
| date         | TEXT   | Publication date            |
| author       | TEXT   | Author(s)                   |
| category     | TEXT   | Category                    |
| keywords     | TEXT   | Keywords                    |
| word\_count  | INT    | Word count                  |
| editor\_abbr | TEXT   | Editor abbreviation         |
| site\_name   | TEXT   | Website name                |

### Chip Table

| Column         | Type   | Description                 |
| -------------- | ------ | --------------------------- |
| id             | SERIAL | Unique ID                   |
| url            | TEXT   | Article URL (unique)        |
| title          | TEXT   | Article title               |
| author         | TEXT   | Author(s)                   |
| date           | TEXT   | Publication date            |
| keywords       | TEXT   | Keywords                    |
| description    | TEXT   | Article description         |
| type           | TEXT   | Article type                |
| page\_level1   | TEXT   | Page level 1                |
| page\_level2   | TEXT   | Page level 2                |
| page\_level3   | TEXT   | Page level 3                |
| page\_template | TEXT   | Page template               |

**Note:** The Streamlit dashboard merges data from both tables for a unified view.

---

## 📊 Streamlit Features

The dashboard offers over **20 different features and visualizations**:

### 📈 Visualizations

- **Author Networks** (🕸️) - Interactive network graphs showing connections between authors
- **Keyword Analysis** (🔑) - Frequency distribution of key keywords
- **Word Clouds** - Visual representation of most common terms
- **Time Analysis** (📅) - Article publications over time
- **Trend Analysis** - Predictions and pattern recognition
- **AI Analysis** (🤖) - Topic Modeling, Sentiment Analysis
- **Sentiment Analysis** - Article sentiment analysis
- **Topic Clustering** - Automatic topic grouping
- **Content Recommendations** - Find similar articles
- **Performance Metrics** (⚡) - System statistics

### 🔧 Interactive Features

- **Source Filter** - Show Heise, Chip, or both
- **Search Function** (🔍) - Full-text search in articles
- **Date Range Filter** - Time-based filtering
- **Category Filter** - Filter by category
- **Author Filter** - Filter by author
- **Export Function** - CSV, Excel, JSON
- **SQL Queries** (🔧) - Execute custom queries
- **Cache Management** - Clear data cache

### 📥 Export Options

- CSV export with source info
- Excel export (.xlsx)
- JSON export
- SQL export
- Filtered exports possible

---

## 📂 Project Structure

```
📂 datamining/
├── 📂 heise/                          # Heise crawlers and related scripts
│   ├── 📄 main.py                     # Archive crawler (backwards)
│   ├── 📄 current_crawler.py          # Live crawler (every 5 minutes)
│   ├── 📄 bot.py                      # Discord bot
│   ├── 📄 api.py                      # API functionalities
│   ├── 📄 notification.py             # Email notifications
│   ├── 📄 export_articles.py          # Export functionality
│   ├── 📄 test_notification.py        # Notification test
│   └── 📂 templates/                  # HTML templates
│       ├── 📄 news_feed.html
│       └── 📄 query.html
├── 📂 chip/                           # Chip crawlers and related scripts
│   ├── 📄 main.py                     # Archive crawler (forwards)
│   ├── 📄 current_crawler.py          # Live crawler (every 10 minutes)
│   ├── 📄 notification.py             # Email notifications
│   └── 📄 export_articles.py          # Export functionality
├── 📂 visualization/                  # Unified Streamlit dashboard
│   ├── 📄 streamlit_app.py            # Main Streamlit application
│   └── 📄 requirements_streamlit.txt  # Streamlit dependencies
├── 📂 data/                           # Export directory
├── 📂 docker/                         # Docker configurations (if present)
├── 📄 docker-compose.yml              # Docker Compose configuration
├── 📄 Dockerfile                      # Docker image definition
├── 📄 requirements.txt                # Python dependencies
├── 📄 .env                            # Environment variables (create manually)
├── 📄 .gitignore                      # Git ignore file
├── 📄 README.md                       # This file
├── 📄 QUICKSTART.md                   # Quick start guide
├── 📄 ARCHITECTURE.md                 # System architecture
├── 📄 DOCKER_SETUP.md                 # Docker setup guide
├── 📄 SECURITY.md                     # Security guidelines
└── 📄 LICENSE                         # License (GNU GPL)
```

---

## 🔧 Management with Docker Tools

For centralized management of your Docker containers, we recommend the following 3rd-party solutions:

### 🏆 Portainer (Recommended)

**Installation:**

```bash
docker volume create portainer_data

docker run -d \
  -p 9000:9000 \
  --name portainer \
  --restart always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest
```

**Access:** `http://localhost:9000`

**Features:**
- Web-based GUI for container management
- View logs in real-time
- Start/stop/pause containers
- Resource monitoring
- Stack management (Docker Compose)
- User-friendly

### 🎨 Dockge (Alternative)

**Installation:**

```bash
docker run -d \
  -p 5001:5001 \
  --name dockge \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v dockge_data:/app/data \
  louislam/dockge:1
```

**Access:** `http://localhost:5001`

**Features:**
- Modern alternative to Portainer
- Docker Compose focused
- Simple user interface
- Live logs

### 🚢 Yacht

**Installation:**

```bash
docker volume create yacht

docker run -d \
  -p 8000:8000 \
  --name yacht \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v yacht:/config \
  selfhostedpro/yacht
```

**Access:** `http://localhost:8000`

**Features:**
- Self-hosted Docker management
- Template-based
- Clean UI

---

## ❗ Troubleshooting

### Problem: Database connection error

**Solution:**
1. Check `.env` file for correct database credentials
2. Make sure PostgreSQL is running:
   ```bash
   # macOS
   brew services list
   
   # Linux
   sudo systemctl status postgresql
   ```
3. Test the connection:
   ```bash
   psql -U $DB_USER -d $DB_NAME -h $DB_HOST
   ```

### Problem: No data in Streamlit dashboard

**Solution:**
1. Check if tables contain data:
   ```sql
   SELECT COUNT(*) FROM heise;
   SELECT COUNT(*) FROM chip;
   ```
2. Clear Streamlit cache with the "🔄 Clear Cache" button
3. Restart the Streamlit app

### Problem: Email notifications not working

**Solution:**
1. For Gmail: Use an [App Password](https://support.google.com/accounts/answer/185833)
2. Test the notification function:
   ```bash
   cd heise
   python3 test_notification.py
   ```
3. Check SMTP settings in `.env`

### Problem: Discord bot not responding

**Solution:**
1. Check `DISCORD_TOKEN` and `CHANNEL_ID` in `.env`
2. Make sure the bot has the right permissions
3. Check bot logs for errors

### Problem: Docker containers not starting

**Solution:**
1. Check Docker logs:
   ```bash
   docker-compose logs
   ```
2. Make sure all ports are available
3. Check the `.env` file

### Problem: "Table does not exist"

**Solution:**
Run a crawler to create the table:
```bash
cd heise
python3 main.py
```

---

## 🗂️ Examples & Screenshots

(with Tableau and DeepNote, as of March 2025)

![image](https://github.com/user-attachments/assets/ce6ceae0-bdf4-499c-9577-973017bb1eff)

![image](https://github.com/user-attachments/assets/3affd472-8475-4534-99e6-54500493418c)

![image](https://github.com/user-attachments/assets/984babc4-d264-44be-8534-17fdae1f8d5f)

![image](https://github.com/user-attachments/assets/0c1d7a13-0f28-497c-afb3-048ee0a309e7)

![image](https://github.com/user-attachments/assets/ba9a3180-4ae8-4ab3-b4ae-3e81f4621c23)

![image](https://github.com/user-attachments/assets/85ecd8a3-1f31-49d0-ae3a-efdfd98bef21)

![image](https://github.com/user-attachments/assets/1d5c57f7-72be-4aca-8f03-d4fba8bfba9d)

![image](https://github.com/user-attachments/assets/cde65d2c-2b22-481d-9ba4-1c4086eb3f23)

![image](https://github.com/user-attachments/assets/10c87c9c-d444-487c-992f-73d3d4b4a185)

### Deepnote:

We have also generated some graphs with [Deepnote](https://deepnote.com/app/schachner/Web-Crawler-d5025a36-3829-4c12-ad2d-b81aa84bd217?utm_source=app-settings&utm_medium=product-embed&utm_campaign=data-app&utm_content=d5025a36-3829-4c12-ad2d-b81aa84bd217&__embedded=true) (❗ only with random 10,000 rows ❗)

![image](https://github.com/user-attachments/assets/ea99ead8-0b48-47d0-8ddc-7c8ce3bd6b53)

Also check out the [data/Datamining_Heise web crawler-3.twb](https://github.com/SchBenedikt/datamining/blob/3f3fe413aeff25a1ae024215745ed6fa82fc2add/data/Datamining_Heise%20web%20crawler-3.twb) file with an excerpt of analyses.

---

## 📜 License

This program is licensed under **GNU GENERAL PUBLIC LICENSE**

See [LICENSE](LICENSE) for more details.

---

## 🙋 About Us

This project was programmed by both of us within a few days and is constantly being further developed:
- https://github.com/schBenedikt
- https://github.com/schVinzenz

### 📬 Contact

Don't hesitate to contact us if you have questions, feedback, or just want to say hello!

📧 Email: [server@schächner.de](mailto:server@schächner.de)

🌐 Website:
- https://technik.schächner.de
- https://benedikt.schächner.de
- https://vinzenz.schächner.de

### 💖 Special Thanks

The idea for our Heise News Crawler comes from David Kriesel and his presentation "Spiegel Mining" at 33c3.

---

**Happy Crawling! 🎉**

````
