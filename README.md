<a href="https://discord.com/invite/Q6Nn2z3tUP">
  <img src="https://discord.com/api/guilds/1346160903304773703/widget.png?style=banner2" width="250"/>
</a>
<a href="https://deepnote.com/app/schachner/Web-Crawler-d5025a36-3829-4c12-ad2d-b81aa84bd217?utm_source=app-settings&utm_medium=product-embed&utm_campaign=data-app&utm_content=d5025a36-3829-4c12-ad2d-b81aa84bd217&__embedded=true">
  <img src="https://img.shields.io/badge/Open%20in-Deepnote-blue?style=for-the-badge&logo=deepnote" width="250"/>
</a>

# 🗞️ Unified News Mining System

A comprehensive, unified crawler system for collecting and analyzing news articles from **Heise.de** and **Chip.de**.

## 🎯 Quick Links

- 📖 **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- 🏗️ **[Architecture](ARCHITECTURE.md)** - System design and data flow
- 🐳 **[Docker Setup](DOCKER_SETUP.md)** - Deployment with Docker
- 🔧 **[Integration Guide](INTEGRATION_GUIDE.md)** - Technical details
- 📊 **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - What was built

## ✨ Key Features

- ✅ **Separate Tables** - Two dedicated PostgreSQL tables (heise & chip)
- ✅ **Live Monitoring** - Automatic checks every 5-10 minutes
- ✅ **Single Dashboard** - Streamlit app for both sources
- ✅ **Source Filtering** - View Heise, Chip, or both
- ✅ **Discord Bot** - Real-time statistics updates
- ✅ **Export Data** - CSV, Excel, JSON for both sources
- ✅ **Docker Ready** - One-command deployment
- ✅ **AI Analytics** - Powered by Google Generative AI

---

# 🌍 Purpose & Functionality
The **News Mining System** is designed to automatically extract and store news articles from multiple sources. The primary goals are:

- 📡 **Data Collection:** Gather historical news articles from Heise.de and Chip.de.
- 🏛 **Structured Storage:** Store articles from both sources in a unified PostgreSQL database.
- 🔍 **Metadata Extraction:** Retrieve key information such as title, author, category, keywords, and word count.
- 🔄 **Incremental crawling:** Detect duplicate articles and save only new articles.
- 🔔 **Notifications:** Send an email if an error occurs during the crawling process.
- 🎨 **Enhanced Terminal Output:** Uses PyFiglet for improved readability.
- 📤 **Data export:** Export articles as .csv, .json, .xlsx-file with source filtering.
- 🖥 **API**: Provision of statistics and complete data sets.
- 🤖 **AI Analytics:** Advanced analysis with Google Generative AI for topic modeling, sentiment analysis, and trend detection.
- 🎯 **Unified Dashboard:** Single Streamlit application for both Heise and Chip data.
- 🤖 **Discord Bot:** Real-time statistics for both sources in Discord.
  
Also an API endpoint is provided that can display the crawled data and statistics.

---

## 🚀 Installation & Setup

### 1️⃣ Requirements

🔹 Python 3

🔹 PostgreSQL

🔹 Required Python Libraries (Dependencies in [requirements.txt](requirements.txt))

### 2️⃣ Install Dependencies

Install required Python libraries:

```sh
pip3 install -r requirements.txt
```

### 3️⃣ Create `.env` File

Set up your database, email credentials, and AI API keys by creating a `.env` file:

```env
EMAIL_USER=...
EMAIL_PASSWORD=...
SMTP_SERVER=...
SMTP_PORT=...
ALERT_EMAIL=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=...
DISCORD_TOKEN=...
CHANNEL_ID=...
GOOGLE_API_KEY=...  # Für KI-Analysen mit Google Generative AI
```


---

## 🛠 Usage

### 1️⃣ Start the Heise Archive Crawler (crawls backward from newest to oldest)

```sh
cd heise
python3 main.py
```

#### Example Terminal Output

```
[INFO] Crawle URL: https://www.heise.de/newsticker/archiv/xxxx/xx
[INFO] Gefundene Artikel (insgesamt): 55
xxxx-xx-xx xx:xx:xx [INFO] Verarbeite 16 Artikel für den Tag xxxx-xx-xx
xxxx-xx-xx xx:xx:xx [INFO] 2025-03-01T20:00:00 - article-name
(⬆️ date)
```
If fewer than 10 items are found per day, an e-mail will be sent


### 2️⃣ Start the Heise Live Crawler (checks for new articles every 10 minutes)

```sh
cd heise
python3 current_crawler.py
```

#### Example Terminal Output

```
[INFO] Crawle URL: https://www.heise.de/newsticker/archiv/xxxx/xx
[INFO] Gefundene Artikel (insgesamt): 55
xxxx-xx-xx xx:xx:xx [INFO] Aktueller Crawl-Durchlauf abgeschlossen.
xxxx-xx-xx xx:xx:xx [INFO] Warte 300 Sekunden bis zum nächsten Crawl.
(⬆️ date)
```

### 3️⃣ Start the Chip Archive Crawler (crawls from page 1 onward)

```sh
cd chip
python3 main.py
```

### 4️⃣ Start the Chip Live Crawler (checks for new articles every 10 minutes)

```sh
cd chip
python3 current_crawler.py
```

### 5️⃣ Use API

The API server starts automatically when running heise/main.py. You can call up the statistics here:
```
http://127.0.0.1:6600/stats
```

### 6️⃣ Start Unified Streamlit Dashboard

Start the interactive Streamlit dashboard with support for both Heise and Chip:

```sh
cd visualization
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

The dashboard includes:
- 📊 Interactive visualizations
- 📈 Time-based analysis
- 🔍 Keyword and content exploration
- 🤖 AI-powered analytics with Google Generative AI
- 🔮 Trend detection and topic modeling
- 🔀 Source filtering (Heise, Chip, or both)
- 📥 Export functionality (CSV, Excel, JSON)


### 7️⃣ Start Discord Bot

Start the Discord bot for real-time statistics updates:

```sh
cd heise
python3 bot.py
```

The bot provides:
- Total article counts for both sources
- Today's article counts for both sources
- Author statistics
- Updates every 10 minutes


### 8️⃣ Export articles

You can export the data for each source to a CSV, JSON or XLSX file.

**Export Heise articles:**
```sh
cd heise
python3 export_articles.py
```

**Export Chip articles:**
```sh
cd chip
python3 export_articles.py
```

Exported articles are saved in the data/ directory.

---




---

## 🏗 Database Schema

The database now uses **two separate tables** for better organization:

### Heise Table

| Column       | Type   | Description          |
| ------------ | ------ | -------------------- |
| id           | SERIAL | Unique ID            |
| title        | TEXT   | Article title        |
| url          | TEXT   | Article URL (unique) |
| date         | TEXT   | Publication date     |
| author       | TEXT   | Author(s)            |
| category     | TEXT   | Category             |
| keywords     | TEXT   | Keywords             |
| word\_count  | INT    | Word count           |
| editor\_abbr | TEXT   | Editor abbreviation  |
| site\_name   | TEXT   | Website name         |

### Chip Table

| Column       | Type   | Description          |
| ------------ | ------ | -------------------- |
| id           | SERIAL | Unique ID            |
| title        | TEXT   | Article title        |
| url          | TEXT   | Article URL (unique) |
| date         | TEXT   | Publication date     |
| author       | TEXT   | Author(s)            |
| keywords     | TEXT   | Keywords             |
| description  | TEXT   | Article description  |
| type         | TEXT   | Article type         |
| page\_level1 | TEXT   | Page level 1         |
| page\_level2 | TEXT   | Page level 2         |
| page\_level3 | TEXT   | Page level 3         |
| page\_template | TEXT | Page template        |

Note: The Streamlit dashboard automatically merges data from both tables for unified viewing.

---



## 📩 Error Notifications

If any errors occur, an email notification will be sent.

---

## 📂 Project Structure

```
📂 datamining
├── 📂 heise/                      # Heise crawler and related scripts
│   ├── 📄 main.py                 # Archive crawler (backward crawling)
│   ├── 📄 current_crawler.py      # Live crawler (every 10 minutes)
│   ├── 📄 bot.py                  # Discord bot
│   ├── 📄 api.py                  # API functionalities
│   ├── 📄 notification.py         # Email notification handler
│   ├── 📄 export_articles.py      # Export functionality
│   └── 📂 templates/              # HTML templates
├── 📂 chip/                       # Chip crawler and related scripts
│   ├── 📄 main.py                 # Archive crawler (forward crawling)
│   ├── 📄 current_crawler.py      # Live crawler (every 10 minutes)
│   ├── 📄 notification.py         # Email notification handler
│   └── 📄 export_articles.py      # Export functionality
├── 📂 visualization/              # Unified Streamlit dashboard
│   ├── 📄 streamlit_app.py        # Main Streamlit application
│   └── 📄 requirements_streamlit.txt
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env                        # Environment variables (create manually)
└── 📄 README.md
```

## ❗Troubleshooting

### 🌐 Start API manually

```sh
python3 api.py
```

### 📧 Testing Notifications

```sh
python3 test_notification.py
```

### ⚠️ Found an error?
Please create a pull request or contact us via server@schächner.de

---





## 🗂️ Examples

(with Tableu and DeepNote, status March 2025)
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
We have also generated some graphs with [Deepnote](https://deepnote.com/app/schachner/Web-Crawler-d5025a36-3829-4c12-ad2d-b81aa84bd217?utm_source=app-settings&utm_medium=product-embed&utm_campaign=data-app&utm_content=d5025a36-3829-4c12-ad2d-b81aa84bd217&__embedded=true) (❗ only with Random 10.000 rows ❗)

![image](https://github.com/user-attachments/assets/ea99ead8-0b48-47d0-8ddc-7c8ce3bd6b53)


Check out also the [data/Datamining_Heise web crawler-3.twb](https://github.com/SchBenedikt/datamining/blob/3f3fe413aeff25a1ae024215745ed6fa82fc2add/data/Datamining_Heise%20web%20crawler-3.twb)-file with an excerpt of analyses.

---

## 📜 License
This program is licensed under **GNU GENERAL PUBLIC LICENSE**



## 🙋 About us

This project was programmed by both of us within a few days and is constantly being further developed:
- https://github.com/schBenedikt
- https://github.com/schVinzenz

### 📬 Contact

Feel free to reach out if you have any questions, feedback, or just want to say hi!

📧 Email: [server@schächner.de](mailto:server@schächner.de)

🌐 Website:
- https://technik.schächner.de
- https://benedikt.schächner.de
- https://vinzenz.schächner.de


💖 Special Thanks

The idea for our Heise News Crawler comes from David Kriesel and his presentation “Spiegel Mining” at 33c3.


---

Happy Crawling! 🎉
