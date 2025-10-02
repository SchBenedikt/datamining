# Migration Guide: From Unified Table to Separate Tables

## Overview

This guide explains the migration from a single unified `articles` table to two separate tables (`heise` and `chip`) for better data organization and performance.

## What Changed

### Database Structure

**Before:**
- Single `articles` table with `source` column to distinguish between Heise and Chip articles
- All articles stored in one table with mixed schemas

**After:**
- `heise` table: Dedicated table for Heise.de articles
- `chip` table: Dedicated table for Chip.de articles
- Each table has schema optimized for its source

### Benefits

1. **Better Performance**: Queries are faster with smaller, dedicated tables
2. **Clear Separation**: Each source has its own space
3. **Optimized Schemas**: Each table has only the columns it needs
4. **Easier Maintenance**: Source-specific changes don't affect other tables
5. **Feature Parity**: Both sources now have identical capabilities (including export)

## Migration Steps

### Option 1: Fresh Start (Recommended for Development)

1. **Backup existing data** (if needed):
   ```bash
   cd heise
   python3 export_articles.py  # Choose SQL export
   ```

2. **Drop old table** (optional):
   ```sql
   DROP TABLE IF EXISTS articles;
   ```

3. **Run crawlers** to populate new tables:
   ```bash
   # Terminal 1: Heise crawler
   cd heise && python3 main.py
   
   # Terminal 2: Chip crawler
   cd chip && python3 main.py
   ```

### Option 2: Migrate Existing Data

If you have existing data in an `articles` table with a `source` column:

```sql
-- Create new tables
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

-- Migrate Heise articles
INSERT INTO heise (title, url, date, author, category, keywords, word_count, editor_abbr, site_name)
SELECT title, url, date, author, category, keywords, word_count, editor_abbr, site_name
FROM articles
WHERE COALESCE(source, 'heise') = 'heise'
ON CONFLICT (url) DO NOTHING;

-- Migrate Chip articles
INSERT INTO chip (url, title, author, date, keywords, description, type, page_level1, page_level2, page_level3, page_template)
SELECT url, title, author, date, keywords, description, type, page_level1, page_level2, page_level3, page_template
FROM articles
WHERE source = 'chip'
ON CONFLICT (url) DO NOTHING;

-- Optional: Drop old table after verifying migration
-- DROP TABLE articles;
```

## Updated File Structure

```
📂 datamining
├── 📂 heise/
│   ├── 📄 main.py                  ✓ Updated to use 'heise' table
│   ├── 📄 current_crawler.py       ✓ Updated to use 'heise' table
│   ├── 📄 export_articles.py       ✓ Updated to export from 'heise' table
│   └── 📄 bot.py                   ✓ Updated to query both tables
├── 📂 chip/
│   ├── 📄 main.py                  ✓ Updated to use 'chip' table
│   ├── 📄 current_crawler.py       ✓ Updated to use 'chip' table
│   └── 📄 export_articles.py       ✓ NEW: Export Chip data
├── 📂 visualization/
│   └── 📄 streamlit_app.py         ✓ Updated to merge data from both tables
├── 📂 data/                        ✓ NEW: Directory for exports
└── 📄 .gitignore                   ✓ Updated to exclude export files
```

## Testing the Changes

### 1. Verify Database Tables

```sql
-- Check if tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name IN ('heise', 'chip');

-- Count records
SELECT 'heise' as source, COUNT(*) FROM heise
UNION ALL
SELECT 'chip' as source, COUNT(*) FROM chip;
```

### 2. Test Crawlers

```bash
# Test Heise crawler
cd heise
python3 main.py
# Should create/use 'heise' table

# Test Chip crawler
cd chip
python3 main.py
# Should create/use 'chip' table
```

### 3. Test Export Functionality

```bash
# Export Heise data
cd heise
python3 export_articles.py
# Choose format (CSV, Excel, JSON, SQL)

# Export Chip data
cd chip
python3 export_articles.py
# Choose format (CSV, Excel, JSON, SQL)
```

### 4. Test Streamlit Dashboard

```bash
cd visualization
streamlit run streamlit_app.py
```

**Verify:**
- Data from both sources appears
- Source filter works (Heise/Chip selection)
- All visualizations work correctly
- Export functions work

### 5. Test Discord Bot

```bash
cd heise
python3 bot.py
```

**Verify:**
- Bot shows counts for both Heise and Chip
- Today's statistics are accurate
- Updates every 10 minutes

## Troubleshooting

### Issue: "Table 'heise' does not exist"

**Solution**: Run the crawler to create the table:
```bash
cd heise
python3 main.py
```

### Issue: "No data in Streamlit dashboard"

**Solutions:**
1. Check database connection in `.env` file
2. Verify tables have data:
   ```sql
   SELECT COUNT(*) FROM heise;
   SELECT COUNT(*) FROM chip;
   ```
3. Check Streamlit logs for errors

### Issue: "Export fails with 'no such table'"

**Solution**: The export scripts are now source-specific:
- Use `heise/export_articles.py` for Heise data
- Use `chip/export_articles.py` for Chip data

### Issue: Discord bot shows 0 articles

**Solution**: Ensure crawlers have run and populated tables:
```bash
# Check table counts
psql -U $DB_USER -d $DB_NAME -c "SELECT COUNT(*) FROM heise; SELECT COUNT(*) FROM chip;"
```

## Rollback Plan

If you need to rollback to the unified table structure:

1. **Export all data** from both tables
2. **Recreate unified table** with source column
3. **Import data** back with appropriate source values
4. **Restore old code** from previous commit

## Additional Notes

- **Docker**: Docker configuration automatically uses the new table structure
- **API**: If using the API endpoint, ensure it queries both tables
- **Backups**: Always backup your database before major changes
- **Performance**: Monitor query performance; separate tables should be faster
- **Future Sources**: Easy to add more sources by creating new tables

## Support

If you encounter issues:
1. Check the logs in crawler output
2. Verify `.env` configuration
3. Test database connectivity
4. Review `IMPLEMENTATION_SUMMARY.md` for technical details
5. Open an issue on GitHub with error details
