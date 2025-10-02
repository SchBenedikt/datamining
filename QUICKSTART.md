# Quick Start Guide

Welcome to the unified News Mining System! This guide will help you get started quickly.

## Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.11+ with pip (for manual installation)
- PostgreSQL database (accessible from your environment)
- Discord Bot Token (optional, for bot features)
- Google API Key (optional, for AI features in Streamlit)

## Setup in 5 Minutes

### Step 1: Clone the Repository

```bash
git clone https://github.com/SchBenedikt/datamining.git
cd datamining
```

### Step 2: Create Environment File

Create a `.env` file in the root directory:

```bash
cp .env.example .env  # If example exists
# OR create manually:
nano .env
```

Add these required variables:

```env
# Database (Required)
DB_NAME=web_crawler
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=192.168.1.100
DB_PORT=5432

# Email Notifications (Required for error alerts)
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ALERT_EMAIL=admin@example.com

# Discord Bot (Optional)
DISCORD_TOKEN=your_bot_token_here
CHANNEL_ID=1234567890

# Google AI (Optional - for Streamlit AI features)
GOOGLE_API_KEY=your_google_api_key
```

### Step 3: Choose Your Deployment Method

#### Option A: Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

#### Option B: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r visualization/requirements_streamlit.txt

# Start services in separate terminals:

# Terminal 1: Heise live crawler
cd heise
python3 current_crawler.py

# Terminal 2: Chip live crawler  
cd chip
python3 current_crawler.py

# Terminal 3: Streamlit dashboard
cd visualization
streamlit run streamlit_app.py

# Terminal 4: Discord bot (optional)
cd heise
python3 bot.py
```

### Step 4: Access the Dashboard

Open your browser:
```
http://localhost:8501
```

## What You'll See

### Streamlit Dashboard

1. **Navigation** - Choose from multiple analysis pages
2. **Source Filter** - Select Heise, Chip, or both
3. **Statistics** - View article counts, authors, categories
4. **Export** - Download data as CSV, XLSX, or JSON
5. **Analytics** - AI-powered insights and trends

### Discord Bot (if configured)

Every 10 minutes, the bot posts statistics:
- Today's article count (Heise & Chip)
- Total article count (Heise & Chip)  
- Author statistics

## Common Tasks

### Start Only Live Crawlers

```bash
docker-compose up -d heise-live-crawler chip-live-crawler
```

### View Specific Service Logs

```bash
docker-compose logs -f heise-live-crawler
```

### Restart a Service

```bash
docker-compose restart streamlit-dashboard
```

### Export Data from Streamlit

1. Open dashboard at http://localhost:8501
2. Use source filter to select desired sources
3. Click "📥 Daten exportieren" in sidebar
4. Choose format (CSV, Excel, JSON)
5. Click download button

### Check Service Status

```bash
docker-compose ps
```

## Troubleshooting

### Issue: "Cannot connect to database"

**Solution:**
1. Check if PostgreSQL is running
2. Verify DB_HOST, DB_PORT, DB_USER, DB_PASSWORD in .env
3. Ensure database firewall allows connections

### Issue: "No data in dashboard"

**Solution:**
1. Check if crawlers are running: `docker-compose ps`
2. View crawler logs: `docker-compose logs heise-live-crawler`
3. Wait a few minutes for first articles to be collected

### Issue: "Discord bot not posting"

**Solution:**
1. Verify DISCORD_TOKEN in .env
2. Check CHANNEL_ID is correct
3. Ensure bot has permissions in Discord channel
4. View bot logs: `docker-compose logs discord-bot`

### Issue: "Port 8501 already in use"

**Solution:**
Change port in docker-compose.yml:
```yaml
streamlit-dashboard:
  ports:
    - "8080:8501"  # Change 8080 to any available port
```

## Next Steps

1. **Explore the Dashboard** - Try different analysis pages
2. **Monitor Crawlers** - Check logs to see articles being collected
3. **Export Data** - Download some data to analyze externally
4. **Read Documentation** - Check other .md files for details:
   - `INTEGRATION_GUIDE.md` - Technical details
   - `ARCHITECTURE.md` - System design
   - `DOCKER_SETUP.md` - Advanced Docker usage

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐
│   Heise     │     │    Chip     │
│  Crawlers   │     │  Crawlers   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                │
         ┌──────▼──────┐
         │  PostgreSQL │
         └──────┬──────┘
                │
       ┌────────┼────────┐
       │        │        │
   ┌───▼──┐ ┌──▼──┐ ┌──▼─────┐
   │Stream│ │Discord│ │Export │
   │ lit  │ │  Bot  │ │ Tools │
   └──────┘ └───────┘ └────────┘
```

## Support

- Check logs: `docker-compose logs -f [service-name]`
- Review documentation: All .md files in repository
- Verify configuration: .env file settings
- Test database connection: Try connecting with psql or pgAdmin

## Summary

✅ Database configured  
✅ Environment variables set  
✅ Services running  
✅ Dashboard accessible  
✅ Data being collected  

**You're all set!** The system will now continuously collect articles from both Heise and Chip.

For more information, see:
- `README.md` - Full documentation
- `INTEGRATION_GUIDE.md` - Integration details
- `ARCHITECTURE.md` - System architecture
