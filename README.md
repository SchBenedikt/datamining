<table>
  <tr>
    <!-- Linke Spalte: Bild -->
    <td style="vertical-align: top; width: 50%;">
      <img src="https://github.com/user-attachments/assets/98dc9437-12aa-4245-85c0-fac7db8973b1" alt="Web Crawler - visual selection-2" style="display: block; margin-top: 0; max-width:100%;">
    </td>
    <!-- Rechte Spalte: Text -->
    <td style="vertical-align: top; width: 50%;">

# 🌍 Purpose & Functionality

The **Heise News Crawler** is designed to automatically extract and store news articles from Heise's archive. The primary goals are:

- 📡 **Data Collection:** Gather historical news articles from Heise.de.
- 🏛 **Structured Storage:** Store articles in a PostgreSQL database for easy querying and analysis.
- 🔍 **Metadata Extraction:** Retrieve key information such as title, author, category, keywords, and word count.
- 🔄 **Data Updates:** Detect duplicate articles and update them instead of creating duplicates.
- 🔔 **Notifications:** Send an email if an error occurs during the crawling process.
- 🎨 **Enhanced Terminal Output:** Uses PyFiglet for improved readability.

---

## 🚀 Installation & Setup

### 1️⃣ Requirements

🔹 Python 3 🔹 PostgreSQL 🔹 Required Python Libraries (installed manually)

### 2️⃣ Install Dependencies

Install required Python libraries:

```sh
pip install requests beautifulsoup4 psycopg2 pyfiglet
```

### 3️⃣ Create `.env` File

Set up your database credentials by creating a `.env` file:

```sh
DB_NAME=web_crawler
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
```

Additionally, set up email credentials in `.env` for notifications.

---

## 🛠 Usage

### Start the Crawler

```sh
python main.py
```

### Example Terminal Output

```
Crawling URL: https://www.heise.de/newsticker/archiv/2025/03
Found Articles: 25
Heise-Article 1
Heise-Article 2
...
```

If no articles are found, the crawl process will stop, and an email notification will be sent.
</table>

---

## 🏗 Database Schema

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

---



## 📩 Error Notifications

If any errors occur, an email notification will be sent.

---

## 📂 Project Structure

```
📂 Heise-News-Crawler
├── 📄 .gitignore         # Git ignore file
├── 📄 .env               # Environment variables (email & database config)
├── 📄 ToDo               # Task list
├── 📄 main.py            # Main crawler script
├── 📄 api.py             # API functionalities
├── 📄 notification.py    # Email notification handler
└── 📂 templates/         # HTML email templates
```

---

## 📜 License
Happy Crawling! 🎉

-- by 01.03.2025
