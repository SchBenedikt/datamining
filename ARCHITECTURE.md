# System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED CRAWLER SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐                        ┌──────────────────┐
│  Heise Crawlers  │                        │  Chip Crawlers   │
├──────────────────┤                        ├──────────────────┤
│ Archive Crawler  │                        │ Archive Crawler  │
│ (backward)       │                        │ (forward)        │
│                  │                        │                  │
│ Live Crawler     │                        │ Live Crawler     │
│ (every 5 min)    │                        │ (every 10 min)   │
└────────┬─────────┘                        └────────┬─────────┘
         │                                           │
         │ source='heise'                source='chip'│
         │                                           │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   PostgreSQL DB      │
              │   ┌──────────────┐   │
              │   │   articles   │   │
              │   │  (unified)   │   │
              │   │              │   │
              │   │ • source     │   │
              │   │ • title      │   │
              │   │ • url        │   │
              │   │ • date       │   │
              │   │ • author     │   │
              │   │ • ...        │   │
              │   └──────────────┘   │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Streamlit   │  │ Discord Bot │  │ Export      │
│ Dashboard   │  │             │  │ Tools       │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ • Filter by │  │ • Heise     │  │ • CSV       │
│   source    │  │   stats     │  │ • XLSX      │
│ • Analytics │  │ • Chip      │  │ • JSON      │
│ • Search    │  │   stats     │  │             │
│ • Export    │  │ • Today's   │  │ with source │
│             │  │   articles  │  │ column      │
└─────────────┘  └─────────────┘  └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE                               │
├─────────────────────────────────────────────────────────────────┤
│  heise-archive-crawler   │  heise-live-crawler                  │
│  chip-archive-crawler    │  chip-live-crawler                   │
│  streamlit-dashboard     │  discord-bot                         │
└─────────────────────────────────────────────────────────────────┘


DATA FLOW:
──────────

1. Crawlers fetch articles from their respective sources
2. Articles are stored in unified DB with source identifier
3. Streamlit loads all articles and filters by source
4. Discord bot queries database for both sources
5. Users can export filtered data with source info


KEY FEATURES:
─────────────

✓ Unified Database       - Single source of truth
✓ Source Filtering       - View Heise, Chip, or both
✓ Live Monitoring        - Continuous article collection
✓ Real-time Stats        - Discord bot updates
✓ Export with Source     - CSV, XLSX, JSON
✓ Docker Management      - Easy deployment
✓ Backward Compatible    - Works with existing data


CRAWLING STRATEGY:
──────────────────

Heise Archive:
  Start: 2025/06 → Goes backward (2025/05, 2025/04, ...)
  
Heise Live:
  Checks: Current month archive every 5 minutes
  
Chip Archive:
  Start: Page 1 → Goes forward (Page 2, 3, 4, ...)
  
Chip Live:
  Checks: Page 1 (newest) every 10 minutes


DEPLOYMENT OPTIONS:
───────────────────

Option 1: Docker (Recommended)
  docker-compose up -d
  
Option 2: Manual
  Terminal 1: cd heise && python3 current_crawler.py
  Terminal 2: cd chip && python3 current_crawler.py
  Terminal 3: cd visualization && streamlit run streamlit_app.py
  Terminal 4: cd heise && python3 bot.py


MONITORING:
───────────

• Docker Logs:     docker-compose logs -f [service]
• Streamlit:       http://localhost:8501
• Discord Bot:     Real-time stats in Discord channel
• Export:          Download data from Streamlit


EXTENSIBILITY:
──────────────

Adding a new source (e.g., Golem):
1. Create golem/main.py and golem/current_crawler.py
2. Set source='golem' in insert statements
3. Add service to docker-compose.yml
4. Streamlit automatically picks up new source
5. Discord bot includes it in statistics
```
