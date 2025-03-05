import discord
import asyncpg
import os
from discord.ext import tasks
from dotenv import load_dotenv
import time
from datetime import datetime, timezone
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

intents = discord.Intents.default()
bot = discord.Client(intents=intents)

last_message = None  # Speichert die letzte Nachricht

async def get_entry_count():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        count = await conn.fetchval("SELECT COUNT(*) FROM articles;")  # Deine Tabelle hier
        await conn.close()
        return count
    except Exception as e:
        print(f"❌ Fehler bei der DB-Abfrage: {e}")
        return None
async def get_author_count():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        authors = await conn.fetch("SELECT author FROM articles;")  # Ersetze `your_table`
        await conn.close()

        author_list = []
        for row in authors:
            author_list.extend(row["author"].split(","))  # Autoren aufsplitten

        unique_authors = set(a.strip() for a in author_list)  # Doppelte entfernen, Leerzeichen trimmen
        return len(unique_authors)  # Anzahl der einzigartigen Autoren
    except Exception as e:
        print(f"❌ Fehler bei der Autoren-Abfrage: {e}")
        return None
async def get_today_counts():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        today_date = datetime.now(timezone.utc).date()  # Heutiges Datum in UTC

        # Alle Einträge von heute abrufen
        rows = await conn.fetch("SELECT date, author FROM articles;")  # Ersetze `your_table`
        await conn.close()

        article_count = 0
        author_list = []

        for row in rows:
            # Datum aus der DB in UTC umwandeln
            article_date = datetime.fromisoformat(row["date"]).date()
            if article_date == today_date:
                article_count += 1
                author_list.extend(row["author"].split(","))  # Autoren splitten

        unique_authors_today = set(a.strip() for a in author_list)  # Doppelte entfernen
        return article_count, len(unique_authors_today)
    except Exception as e:
        print(f"❌ Fehler bei der heutigen Abfrage: {e}")
        return None, None

@tasks.loop(minutes=10)
async def update_entry_count():
    global last_message
    print("🔄 update_entry_count läuft...")

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print("⚠️ Fehler: Kanal nicht gefunden.")
        return

    count = await get_entry_count()
    author_count = await get_author_count()
    today_articles, today_authors = await get_today_counts()

    if count is None or author_count is None or today_articles is None or today_authors is None:
        return

    embed = discord.Embed(
        title="📊 **Datenbank-Statistiken**",
        color=0x3498db
    )

    # Unsichtbarer Abstand für bessere Trennung - vergrößert
    spacer = "\u200B" * 60  

    # Linke Spalte: HEUTE (als Heading in Fettschrift)
    embed.add_field(
        name="▬▬▬ 📅 **HEUTE** ▬▬▬",
        value=f"**Artikel:** {today_articles}\n **Autoren:** {today_authors}",
        inline=True
    )

    # Unsichtbarer Abstand
    embed.add_field(name=spacer, value=spacer, inline=True)

    # Rechte Spalte: GESAMT (als Heading in Fettschrift)
    embed.add_field(
        name="▬▬▬ 🗓️ **GESAMT** ▬▬▬",
        value=f"Artikel: {count}\nAutoren: {author_count}",
        inline=True
    )
    embed.add_field(name=spacer, value=spacer, inline=True)
    # Zeitangabe
    embed.add_field(name="🕒 **Letzte Aktualisierung**", value=f"<t:{int(time.time())}:R>", inline=False)
    embed.set_footer(text="Daten werden alle 10 Minuten aktualisiert.")

    try:
        if last_message:
            await last_message.edit(embed=embed)
        else:
            last_message = await channel.send(embed=embed)
    except discord.NotFound:
        last_message = await channel.send(embed=embed)

@bot.event
async def on_ready():
    print(f"Bot ist eingeloggt als {bot.user}")
    if not update_entry_count.is_running():
        update_entry_count.start()

bot.run(TOKEN)
