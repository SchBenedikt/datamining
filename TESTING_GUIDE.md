# 🧪 Testing Guide

This document provides instructions for testing the new features and verifying that everything works correctly.

## Prerequisites

1. **PostgreSQL Database**: Ensure PostgreSQL is installed and running
2. **Python 3.8+**: Python with required packages
3. **Configuration**: `.env` file in root directory

## 🏁 Quick Test Checklist

- [ ] Database initialization works
- [ ] Heise crawler starts without errors
- [ ] Chip crawler starts without errors
- [ ] Streamlit dashboard loads
- [ ] Auto-refresh works
- [ ] All modules read from root .env

## 📋 Detailed Testing Steps

### 1. Test Database Initialization

**Test automatic initialization:**
```bash
cd /home/runner/work/datamining/datamining
python3 init_database.py
```

**Expected output:**
```
2024-10-02 21:30:00 [INFO] Starting database initialization...
2024-10-02 21:30:00 [SUCCESS] Table 'heise' initialized successfully
2024-10-02 21:30:00 [SUCCESS] Table 'chip' initialized successfully
2024-10-02 21:30:00 [SUCCESS] Crawl state tables initialized successfully
2024-10-02 21:30:00 [SUCCESS] Database initialization completed successfully!
```

**Verify tables exist:**
```bash
psql -U postgres -d datamining -c "\dt"
```

**Expected tables:**
- heise
- chip
- heise_crawl_state
- chip_crawl_state

---

### 2. Test Configuration Loading

**Test that all modules load from root .env:**

```bash
# Test heise/main.py
cd heise
python3 -c "from main import db_params; print('Heise DB params:', db_params)"

# Test chip/main.py
cd ../chip
python3 -c "from main import db_params; print('Chip DB params:', db_params)"

# Test visualization
cd ../visualization
python3 -c "from streamlit_app import DB_PARAMS; print('Streamlit DB params:', DB_PARAMS)"
```

**Expected:** All should show the same database parameters from root `.env`

---

### 3. Test Heise Crawler

```bash
cd heise
python3 main.py
```

**Expected output includes:**
```
[INFO] Initializing database...
[SUCCESS] Database initialization completed successfully!
[INFO] Starting Heise crawler...
```

**Stop with:** `Ctrl+C`

---

### 4. Test Chip Crawler

```bash
cd chip
python3 main.py
```

**Expected output includes:**
```
[INFO] Initializing database...
[SUCCESS] Database initialization completed successfully!
[INFO] Starting Chip crawler...
```

**Stop with:** `Ctrl+C`

---

### 5. Test Streamlit Dashboard

```bash
cd visualization
streamlit run streamlit_app.py
```

**Expected:**
- Dashboard loads at http://localhost:8501
- No errors about missing .env or database connection
- All navigation items work
- Data loads from database

**Test navigation pages:**
- [ ] 📊 Dashboard
- [ ] 📈 Time Analysis
- [ ] 🔑 Keywords
- [ ] 🔍 Search
- [ ] 🕸️ Network
- [ ] 🤖 AI Analysis

---

### 6. Test Auto-Refresh Feature

1. Open Streamlit dashboard
2. In sidebar, check **"Enable Auto-Refresh"**
3. Select interval (e.g., 60s)
4. Wait for the specified interval
5. Verify page reloads automatically

**Expected:** 
- Info message: "🔄 Auto-refresh every 60s"
- Page reloads after 60 seconds
- Last update timestamp changes

---

### 7. Test Manual Refresh

1. In sidebar, click **"🔄 Refresh Data"**
2. Verify data reloads
3. Check last update timestamp changes

---

### 8. Test Database Connection Status

**In main content area, check the "Database" metric:**
- Should show "🟢 Connected" if database is accessible
- Should show "🔴 Disconnected" or "🔴 Error" if not

---

### 9. Test Data Source Filter

1. In sidebar, use "Data Source" multiselect
2. Select only "heise" or only "chip"
3. Verify that metrics update correctly
4. Verify that only selected source data is shown

---

### 10. Test Export Functionality

**Test heise export:**
```bash
cd heise
python3 export_articles.py csv
```

**Test chip export:**
```bash
cd chip
python3 export_articles.py csv
```

**Expected:** Files created in `data/` directory

---

## 🔍 Verification Checklist

### Configuration
- [ ] `.env` file exists in root directory
- [ ] `.env` contains all required fields (DB_NAME, DB_USER, etc.)
- [ ] No `.env` files in subdirectories (heise/, chip/, visualization/)

### Database
- [ ] PostgreSQL is running
- [ ] Database "datamining" exists
- [ ] Tables heise and chip exist
- [ ] Crawl state tables exist
- [ ] Can connect to database with credentials from .env

### Python Files
- [ ] All Python files pass syntax check
- [ ] All modules import without errors
- [ ] All modules load .env from root directory

### Streamlit Dashboard
- [ ] Dashboard loads without errors
- [ ] 6 navigation pages are available
- [ ] Metrics display in main area (not sidebar)
- [ ] Auto-refresh option available
- [ ] Manual refresh button works
- [ ] Last update timestamp visible
- [ ] Database status indicator works

### Crawlers
- [ ] heise/main.py starts without errors
- [ ] chip/main.py starts without errors
- [ ] Database is initialized automatically on first run
- [ ] Crawlers can connect to database

---

## 🐛 Common Issues and Solutions

### Issue: "No module named 'psycopg2'"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Database connection failed"
**Solution:**
1. Check PostgreSQL is running: `sudo systemctl status postgresql`
2. Verify credentials in `.env`
3. Test connection: `python3 init_database.py`

### Issue: "No data available" in Streamlit
**Solution:**
1. Run crawlers to populate database
2. Check database has data: `psql -U postgres -d datamining -c "SELECT COUNT(*) FROM heise;"`
3. Verify .env configuration

### Issue: "CHANNEL_ID must be an integer" (bot.py)
**Solution:**
Add CHANNEL_ID to .env: `CHANNEL_ID=123456789`

### Issue: Auto-refresh not working
**Solution:**
1. Ensure "Enable Auto-Refresh" is checked
2. Wait for the selected interval
3. Check browser console for errors

---

## 📊 Expected Behavior

### First Run (No Data)
```
1. Database is created/initialized
2. Tables are empty
3. Streamlit shows "No data available"
4. Run crawlers to collect data
```

### After Running Crawlers
```
1. Database contains articles
2. Streamlit shows data and metrics
3. All pages display correctly
4. Charts and visualizations render
```

### Auto-Refresh Enabled
```
1. Page reloads at specified interval
2. New data is fetched from database
3. Metrics and charts update
4. Last update timestamp changes
```

---

## 🎯 Performance Testing

### Cache Testing
1. Load dashboard
2. Note load time
3. Reload page within 60 seconds
4. Verify faster load (from cache)
5. Wait 61+ seconds
6. Reload page
7. Verify data is refreshed (cache expired)

### Database Load Testing
1. Enable auto-refresh with 30s interval
2. Monitor database connections
3. Verify connections are properly closed
4. Check for connection leaks

---

## 📝 Test Results Template

```markdown
## Test Results - [Date]

### Environment
- OS: 
- Python Version: 
- PostgreSQL Version: 
- Browser: 

### Test Results
- [ ] Database initialization: PASS/FAIL
- [ ] Heise crawler: PASS/FAIL
- [ ] Chip crawler: PASS/FAIL
- [ ] Streamlit dashboard: PASS/FAIL
- [ ] Auto-refresh: PASS/FAIL
- [ ] Navigation: PASS/FAIL
- [ ] Data source filter: PASS/FAIL
- [ ] Export functionality: PASS/FAIL

### Issues Found
1. 
2. 
3. 

### Notes
- 
```

---

## 🚀 Next Steps

After successful testing:
1. ✅ Document any issues found
2. ✅ Update .env.example if needed
3. ✅ Add any missing documentation
4. ✅ Deploy to production (if applicable)
5. ✅ Monitor logs for errors
6. ✅ Set up regular database backups

---

## 📚 Additional Resources

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Setup instructions
- [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) - Changes overview
- [README.md](README.md) - Project documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
