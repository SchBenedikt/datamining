#!/bin/bash
# Starte den API-Server
python3 api.py &
PID_API=$!

# Starte den Discord-Bot
python3 bot.py &
PID_BOT=$!

# Starte den Current Crawler
python3 current_crawler.py &
PID_CRAWLER=$!

# Falls du notification.py separat testen möchtest (normalerweise als Modul importiert)
# python3 notification.py &
# PID_NOTIFICATION=$!

echo "Alle Prozesse gestartet."
echo "API PID: $PID_API, BOT PID: $PID_BOT, CRAWLER PID: $PID_CRAWLER"

# Warten, bis alle Prozesse beendet werden (oder du drückst Strg+C)
wait
