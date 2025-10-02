# Unified Crawler System - Integration Guide

## Overview

This document describes the integration of the Heise and Chip crawlers into a unified system that shares a common database and provides a single interface for data access and visualization.

## Key Features

### 1. Unified Database Schema
- Both Heise and Chip articles are stored in the same `articles` table
- A new `source` column distinguishes between articles from different sources
- Automatic migration for existing databases (adds `source` column if missing)

### 2. Live Crawlers
- **Heise Live Crawler** (`heise/current_crawler.py`): Checks for new articles every 5 minutes (300 seconds)
- **Chip Live Crawler** (`chip/current_crawler.py`): Checks for new articles every 10 minutes (600 seconds)
- Both automatically detect and skip duplicate articles

### 3. Unified Streamlit Dashboard
- Single application that works with both data sources
- Source filtering: View Heise only, Chip only, or both
- All analytics work across both sources
- Export functionality includes source information

### 4. Enhanced Discord Bot
- Shows statistics for both sources
- Displays separate counts for Heise and Chip
- Updates every 10 minutes
- Shows today's articles and total articles per source

## Database Schema Changes

### New Column
```sql
ALTER TABLE articles ADD COLUMN source TEXT DEFAULT 'heise';
```

### Full Schema
- `id` - Serial primary key
- `title` - Article title
- `url` - Unique article URL
- `date` - Publication date
- `author` - Author(s)
- `category` - Category
- `keywords` - Keywords
- `word_count` - Word count
- `editor_abbr` - Editor abbreviation
- `site_name` - Site name
- **`source`** - Source identifier ('heise' or 'chip')
- Additional Chip-specific fields: `description`, `type`, `page_level1`, `page_level2`, `page_level3`, `page_template`

## How to Use

### Starting the Crawlers

1. **Heise Archive Crawler** (backward crawling from newest to oldest):
   ```bash
   cd heise
   python3 main.py
   ```

2. **Heise Live Crawler** (checks every 5 minutes):
   ```bash
   cd heise
   python3 current_crawler.py
   ```

3. **Chip Archive Crawler** (forward crawling from page 1):
   ```bash
   cd chip
   python3 main.py
   ```

4. **Chip Live Crawler** (checks every 10 minutes):
   ```bash
   cd chip
   python3 current_crawler.py
   ```

### Starting the Dashboard

```bash
cd visualization
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

In the dashboard:
- Use the "Quelle auswählen" filter in the sidebar to select Heise, Chip, or both
- All visualizations and statistics automatically update based on the selected sources
- Export data with source information included

### Starting the Discord Bot

```bash
cd heise
python3 bot.py
```

The bot will post updates every 10 minutes showing:
- Today's article count (total and per source)
- Total article count (total and per source)
- Author statistics

## Migration from Previous Version

If you have an existing database:

1. The crawlers will automatically add the `source` column when run
2. Existing articles will default to 'heise' source
3. No data loss occurs during migration
4. All existing functionality remains unchanged

## Technical Implementation

### Source Column Handling

**Heise crawlers** (`heise/main.py` and `heise/current_crawler.py`):
```python
base_columns = [..., "source"]
base_values = [..., "heise"]
```

**Chip crawlers** (`chip/main.py` and `chip/current_crawler.py`):
```python
base_columns = [..., "source"]
base_values = [..., "chip"]
```

### Streamlit Source Filtering

```python
available_sources = df['source'].unique().tolist()
selected_sources = st.sidebar.multiselect(
    "Quelle auswählen",
    options=available_sources,
    default=available_sources
)
df = df[df['source'].isin(selected_sources)]
```

### Discord Bot Source Queries

```python
heise_count, chip_count = await get_source_counts()
today_articles, today_authors, heise_today, chip_today = await get_today_counts()
```

## Troubleshooting

### Issue: Source column not found
**Solution**: Run any crawler once - it will automatically add the column

### Issue: Old articles showing as 'heise' when they're from Chip
**Solution**: This is expected for articles crawled before the integration. New articles will have correct source values.

### Issue: Discord bot showing only Heise statistics
**Solution**: Make sure you're running the updated bot.py with get_source_counts() function

### Issue: Streamlit not showing source filter
**Solution**: Clear cache with "🔄 Cache leeren" button in sidebar or restart the app

## Best Practices

1. **Run live crawlers as background services** to continuously monitor for new articles
2. **Use the source filter** to focus on specific sources when needed
3. **Export data regularly** with source information for analysis
4. **Monitor the Discord bot** for real-time statistics
5. **Check logs** from both crawlers to ensure they're working correctly

## Future Enhancements

Potential improvements for the system:
- Add more news sources (e.g., Golem, Heise+, etc.)
- Implement crawler status monitoring in Streamlit
- Add source-specific analytics in the dashboard
- Create scheduled tasks for automated crawler management
- Add crawler performance metrics per source

## Support

For issues or questions:
1. Check the main README.md
2. Review crawler logs for errors
3. Verify database connectivity
4. Ensure all environment variables are set in .env file
