import discord
import asyncpg
import os
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
import time
from datetime import datetime, timezone, timedelta

# Load environment variables from root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

TOKEN = os.getenv("DISCORD_TOKEN")
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": os.getenv("DB_NAME", "datamining"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

# Bot mit Slash Commands
intents = discord.Intents.default()
intents.message_content = True

class MyBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
    
    async def setup_hook(self):
        # Synce die Commands mit Discord
        await self.tree.sync()

bot = MyBot()

last_message = None  # Speichert die letzte Statistik-Nachricht

async def get_entry_count():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        heise_count = await conn.fetchval("SELECT COUNT(*) FROM heise;")
        chip_count = await conn.fetchval("SELECT COUNT(*) FROM chip;")
        await conn.close()
        return heise_count + chip_count
    except Exception as e:
        print(f"❌ Fehler bei der DB-Abfrage: {e}")
        return None

async def get_source_counts():
    """Get article counts per source"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        heise_count = await conn.fetchval("SELECT COUNT(*) FROM heise;")
        chip_count = await conn.fetchval("SELECT COUNT(*) FROM chip;")
        await conn.close()
        return heise_count, chip_count
    except Exception as e:
        print(f"❌ Fehler bei der Quellen-Abfrage: {e}")
        return None, None

async def get_author_count():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        heise_authors = await conn.fetch("SELECT author FROM heise;")
        chip_authors = await conn.fetch("SELECT author FROM chip;")
        await conn.close()

        author_list = []
        for row in heise_authors:
            if row["author"]:
                author_list.extend(row["author"].split(","))  # Autoren aufsplitten
        for row in chip_authors:
            if row["author"]:
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

        # Alle Einträge von heute abrufen aus beiden Tabellen
        heise_rows = await conn.fetch("SELECT date, author FROM heise;")
        chip_rows = await conn.fetch("SELECT date, author FROM chip;")
        await conn.close()

        article_count = 0
        heise_today = 0
        chip_today = 0
        author_list = []

        for row in heise_rows:
            # Datum aus der DB in UTC umwandeln
            article_date = datetime.fromisoformat(row["date"]).date()
            if article_date == today_date:
                article_count += 1
                heise_today += 1
                if row["author"]:
                    author_list.extend(row["author"].split(","))  # Autoren splitten

        for row in chip_rows:
            # Datum aus der DB in UTC umwandeln
            article_date = datetime.fromisoformat(row["date"]).date()
            if article_date == today_date:
                article_count += 1
                chip_today += 1
                if row["author"]:
                    author_list.extend(row["author"].split(","))  # Autoren splitten

        unique_authors_today = set(a.strip() for a in author_list)  # Doppelte entfernen
        return article_count, len(unique_authors_today), heise_today, chip_today
    except Exception as e:
        print(f"❌ Fehler bei der heutigen Abfrage: {e}")
        return None, None, None, None

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
    heise_count, chip_count = await get_source_counts()
    today_articles, today_authors, heise_today, chip_today = await get_today_counts()

    if count is None or author_count is None or today_articles is None or today_authors is None:
        return

    embed = discord.Embed(
        title="📊 **Datenbank-Statistiken**",
        color=0x3498db
    )

    # Unsichtbarer Abstand für bessere Trennung - vergrößert
    spacer = "\u200B" * 60  

    # Linke Spalte: HEUTE (als Heading in Fettschrift)
    today_value = f"**Artikel:** {today_articles}\n**Autoren:** {today_authors}\n"
    if heise_today is not None and chip_today is not None:
        today_value += f"\n📰 Heise: {heise_today}\n🔧 Chip: {chip_today}"
    
    embed.add_field(
        name="▬▬▬ 📅 **HEUTE** ▬▬▬",
        value=today_value,
        inline=True
    )

    # Unsichtbarer Abstand
    embed.add_field(name=spacer, value=spacer, inline=True)

    # Rechte Spalte: GESAMT (als Heading in Fettschrift)
    total_value = f"Artikel: {count}\nAutoren: {author_count}\n"
    if heise_count is not None and chip_count is not None:
        total_value += f"\n📰 Heise: {heise_count}\n🔧 Chip: {chip_count}"
    
    embed.add_field(
        name="▬▬▬ 🗓️ **GESAMT** ▬▬▬",
        value=total_value,
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
    print(f"✅ Bot ist eingeloggt als {bot.user}")
    print(f"📝 Prefix: {bot.command_prefix}")
    print(f"📋 Verfügbare Befehle: {', '.join([cmd.name for cmd in bot.commands])}")
    if not update_entry_count.is_running():
        update_entry_count.start()

# ==================== COMMANDS ====================

@bot.command(name='commands', aliases=['befehle', 'hilfe'])
async def commands_help(ctx):
    """Zeigt alle verfügbaren Befehle"""
    embed = discord.Embed(
        title="📚 Bot-Befehle",
        description="Hier sind alle verfügbaren Befehle:",
        color=0x00ff00
    )
    
    embed.add_field(
        name="📊 Statistiken",
        value="`!stats` - Zeigt Datenbank-Statistiken\n`!today` - Artikel von heute",
        inline=False
    )
    
    embed.add_field(
        name="🔍 Suche",
        value="`!search <Begriff>` - Suche nach Artikeln\n`!latest [Anzahl]` - Neueste Artikel (Standard: 5)\n`!author <Name>` - Artikel eines Autors\n`!category <Kategorie>` - Artikel nach Kategorie",
        inline=False
    )
    
    embed.add_field(
        name="📅 Zeitfilter",
        value="`!week` - Artikel der letzten 7 Tage\n`!month` - Artikel des letzten Monats",
        inline=False
    )
    
    embed.add_field(
        name="🎯 Quelle",
        value="`!heise [Anzahl]` - Neueste Heise-Artikel\n`!chip [Anzahl]` - Neueste Chip-Artikel",
        inline=False
    )
    
    embed.set_footer(text="Tipp: Verwende ! vor jedem Befehl")
    await ctx.send(embed=embed)

@bot.command(name='stats', aliases=['statistik', 'stat'])
async def stats_command(ctx):
    """Zeigt detaillierte Statistiken"""
    count = await get_entry_count()
    author_count = await get_author_count()
    heise_count, chip_count = await get_source_counts()
    today_articles, today_authors, heise_today, chip_today = await get_today_counts()
    
    if count is None:
        await ctx.send("❌ Fehler beim Abrufen der Statistiken!")
        return
    
    embed = discord.Embed(
        title="📊 Datenbank-Statistiken",
        color=0x3498db,
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(
        name="📰 Gesamt",
        value=f"**Artikel:** {count:,}\n**Autoren:** {author_count:,}\n📰 Heise: {heise_count:,}\n🔧 Chip: {chip_count:,}",
        inline=True
    )
    
    embed.add_field(
        name="📅 Heute",
        value=f"**Artikel:** {today_articles}\n**Autoren:** {today_authors}\n📰 Heise: {heise_today}\n🔧 Chip: {chip_today}",
        inline=True
    )
    
    embed.set_footer(text=f"Angefordert von {ctx.author.name}")
    await ctx.send(embed=embed)

@bot.command(name='latest', aliases=['neueste', 'new'])
async def latest_command(ctx, limit: int = 5):
    """Zeigt die neuesten Artikel"""
    if limit > 20:
        await ctx.send("❌ Maximal 20 Artikel auf einmal!")
        return
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Union query für beide Tabellen
        query = """
        (SELECT title, url, date, author, category, 'Heise' as source FROM heise)
        UNION ALL
        (SELECT title, url, date, author, category, 'Chip' as source FROM chip)
        ORDER BY date DESC
        LIMIT $1
        """
        
        articles = await conn.fetch(query, limit)
        await conn.close()
        
        if not articles:
            await ctx.send("📭 Keine Artikel gefunden!")
            return
        
        embed = discord.Embed(
            title=f"🆕 Die {len(articles)} neuesten Artikel",
            color=0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            source_emoji = "📰" if article['source'] == 'Heise' else "🔧"
            date_str = article['date'].strftime('%d.%m.%Y %H:%M') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {source_emoji} {article['title'][:80]}",
                value=f"👤 {article['author'] or 'N/A'} | 📅 {date_str}\n🏷️ {article['category'] or 'N/A'}\n[🔗 Zum Artikel]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='search', aliases=['suche', 'find'])
async def search_command(ctx, *, query: str):
    """Sucht nach Artikeln"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        search_query = """
        (SELECT title, url, date, author, category, 'Heise' as source FROM heise 
         WHERE LOWER(title) LIKE LOWER($1) OR LOWER(keywords) LIKE LOWER($1))
        UNION ALL
        (SELECT title, url, date, author, category, 'Chip' as source FROM chip
         WHERE LOWER(title) LIKE LOWER($1) OR LOWER(keywords) LIKE LOWER($1))
        ORDER BY date DESC
        LIMIT 10
        """
        
        articles = await conn.fetch(search_query, f'%{query}%')
        await conn.close()
        
        if not articles:
            await ctx.send(f"📭 Keine Artikel für '{query}' gefunden!")
            return
        
        embed = discord.Embed(
            title=f"🔍 Suchergebnisse für '{query}'",
            description=f"**{len(articles)} Artikel gefunden**",
            color=0x9b59b6,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            source_emoji = "📰" if article['source'] == 'Heise' else "🔧"
            date_str = article['date'].strftime('%d.%m.%Y') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {source_emoji} {article['title'][:70]}",
                value=f"📅 {date_str} | 👤 {article['author'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='author', aliases=['autor'])
async def author_command(ctx, *, author_name: str):
    """Zeigt Artikel eines bestimmten Autors"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        query = """
        (SELECT title, url, date, category, 'Heise' as source FROM heise 
         WHERE LOWER(author) LIKE LOWER($1))
        UNION ALL
        (SELECT title, url, date, category, 'Chip' as source FROM chip
         WHERE LOWER(author) LIKE LOWER($1))
        ORDER BY date DESC
        LIMIT 10
        """
        
        articles = await conn.fetch(query, f'%{author_name}%')
        await conn.close()
        
        if not articles:
            await ctx.send(f"📭 Keine Artikel von '{author_name}' gefunden!")
            return
        
        embed = discord.Embed(
            title=f"✍️ Artikel von {author_name}",
            description=f"**{len(articles)} Artikel gefunden**",
            color=0x1abc9c,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            source_emoji = "📰" if article['source'] == 'Heise' else "🔧"
            date_str = article['date'].strftime('%d.%m.%Y') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {source_emoji} {article['title'][:70]}",
                value=f"📅 {date_str} | 🏷️ {article['category'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='category', aliases=['kategorie', 'cat'])
async def category_command(ctx, *, category_name: str):
    """Zeigt Artikel einer bestimmten Kategorie"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        query = """
        (SELECT title, url, date, author, 'Heise' as source FROM heise 
         WHERE LOWER(category) LIKE LOWER($1))
        UNION ALL
        (SELECT title, url, date, author, 'Chip' as source FROM chip
         WHERE LOWER(category) LIKE LOWER($1))
        ORDER BY date DESC
        LIMIT 10
        """
        
        articles = await conn.fetch(query, f'%{category_name}%')
        await conn.close()
        
        if not articles:
            await ctx.send(f"📭 Keine Artikel in Kategorie '{category_name}' gefunden!")
            return
        
        embed = discord.Embed(
            title=f"🏷️ Kategorie: {category_name}",
            description=f"**{len(articles)} Artikel gefunden**",
            color=0xf39c12,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            source_emoji = "📰" if article['source'] == 'Heise' else "🔧"
            date_str = article['date'].strftime('%d.%m.%Y') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {source_emoji} {article['title'][:70]}",
                value=f"📅 {date_str} | 👤 {article['author'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='today', aliases=['heute'])
async def today_command(ctx):
    """Zeigt alle Artikel von heute"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        today_date = datetime.now(timezone.utc).date()
        
        query = """
        (SELECT title, url, date, author, category, 'Heise' as source FROM heise 
         WHERE DATE(date) = $1)
        UNION ALL
        (SELECT title, url, date, author, category, 'Chip' as source FROM chip
         WHERE DATE(date) = $1)
        ORDER BY date DESC
        """
        
        articles = await conn.fetch(query, today_date)
        await conn.close()
        
        if not articles:
            await ctx.send("📭 Heute wurden noch keine Artikel veröffentlicht!")
            return
        
        embed = discord.Embed(
            title=f"📅 Artikel von heute ({today_date.strftime('%d.%m.%Y')})",
            description=f"**{len(articles)} Artikel**",
            color=0x2ecc71,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            source_emoji = "📰" if article['source'] == 'Heise' else "🔧"
            time_str = article['date'].strftime('%H:%M') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {source_emoji} {article['title'][:70]}",
                value=f"🕐 {time_str} | 👤 {article['author'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='week', aliases=['woche'])
async def week_command(ctx, limit: int = 10):
    """Zeigt Artikel der letzten 7 Tage"""
    if limit > 20:
        limit = 20
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        query = """
        (SELECT title, url, date, author, category, 'Heise' as source FROM heise 
         WHERE date >= $1)
        UNION ALL
        (SELECT title, url, date, author, category, 'Chip' as source FROM chip
         WHERE date >= $1)
        ORDER BY date DESC
        LIMIT $2
        """
        
        articles = await conn.fetch(query, week_ago, limit)
        await conn.close()
        
        if not articles:
            await ctx.send("📭 Keine Artikel in den letzten 7 Tagen!")
            return
        
        embed = discord.Embed(
            title=f"📆 Artikel der letzten 7 Tage",
            description=f"**{len(articles)} Artikel (Top {limit})**",
            color=0x3498db,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            source_emoji = "📰" if article['source'] == 'Heise' else "🔧"
            date_str = article['date'].strftime('%d.%m. %H:%M') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {source_emoji} {article['title'][:70]}",
                value=f"📅 {date_str} | 🏷️ {article['category'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='heise')
async def heise_command(ctx, limit: int = 5):
    """Zeigt neueste Heise-Artikel"""
    if limit > 15:
        limit = 15
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        articles = await conn.fetch("SELECT title, url, date, author, category FROM heise ORDER BY date DESC LIMIT $1", limit)
        await conn.close()
        
        if not articles:
            await ctx.send("📭 Keine Heise-Artikel gefunden!")
            return
        
        embed = discord.Embed(
            title=f"📰 Neueste Heise-Artikel",
            description=f"**{len(articles)} Artikel**",
            color=0xc0392b,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            date_str = article['date'].strftime('%d.%m. %H:%M') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {article['title'][:70]}",
                value=f"📅 {date_str} | 👤 {article['author'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='chip')
async def chip_command(ctx, limit: int = 5):
    """Zeigt neueste Chip-Artikel"""
    if limit > 15:
        limit = 15
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        articles = await conn.fetch("SELECT title, url, date, author, category FROM chip ORDER BY date DESC LIMIT $1", limit)
        await conn.close()
        
        if not articles:
            await ctx.send("📭 Keine Chip-Artikel gefunden!")
            return
        
        embed = discord.Embed(
            title=f"🔧 Neueste Chip-Artikel",
            description=f"**{len(articles)} Artikel**",
            color=0x16a085,
            timestamp=datetime.now(timezone.utc)
        )
        
        for i, article in enumerate(articles, 1):
            date_str = article['date'].strftime('%d.%m. %H:%M') if article['date'] else 'N/A'
            
            embed.add_field(
                name=f"{i}. {article['title'][:70]}",
                value=f"📅 {date_str} | 👤 {article['author'] or 'N/A'}\n[🔗 Link]({article['url']})",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='reload', hidden=True)
@commands.is_owner()
async def reload_commands(ctx):
    """Lädt alle Befehle neu (nur für Bot-Owner)"""
    try:
        # Liste aller Befehle anzeigen
        commands_list = [cmd.name for cmd in bot.commands]
        
        embed = discord.Embed(
            title="🔄 Befehle neu geladen",
            description=f"**{len(commands_list)} Befehle verfügbar:**",
            color=0x2ecc71
        )
        
        embed.add_field(
            name="Verfügbare Befehle",
            value=", ".join(f"`{cmd}`" for cmd in sorted(commands_list)),
            inline=False
        )
        
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Fehler beim Neuladen: {str(e)}")

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
    heise_count, chip_count = await get_source_counts()
    today_articles, today_authors, heise_today, chip_today = await get_today_counts()

    if count is None or author_count is None or today_articles is None or today_authors is None:
        return

    embed = discord.Embed(
        title="📊 **Datenbank-Statistiken**",
        color=0x3498db
    )

    # Unsichtbarer Abstand für bessere Trennung - vergrößert
    spacer = "\u200B" * 60  

    # Linke Spalte: HEUTE (als Heading in Fettschrift)
    today_value = f"**Artikel:** {today_articles}\n**Autoren:** {today_authors}\n"
    if heise_today is not None and chip_today is not None:
        today_value += f"\n📰 Heise: {heise_today}\n🔧 Chip: {chip_today}"
    
    embed.add_field(
        name="▬▬▬ 📅 **HEUTE** ▬▬▬",
        value=today_value,
        inline=True
    )

    # Unsichtbarer Abstand
    embed.add_field(name=spacer, value=spacer, inline=True)

    # Rechte Spalte: GESAMT (als Heading in Fettschrift)
    total_value = f"Artikel: {count}\nAutoren: {author_count}\n"
    if heise_count is not None and chip_count is not None:
        total_value += f"\n📰 Heise: {heise_count}\n🔧 Chip: {chip_count}"
    
    embed.add_field(
        name="▬▬▬ 🗓️ **GESAMT** ▬▬▬",
        value=total_value,
        inline=True
    )
    embed.add_field(name=spacer, value=spacer, inline=True)
    # Zeitangabe
    embed.add_field(name="🕒 **Letzte Aktualisierung**", value=f"<t:{int(time.time())}:R>", inline=False)
    embed.set_footer(text="Daten werden alle 10 Minuten aktualisiert. Benutze !help für Befehle.")

    try:
        if last_message:
            await last_message.edit(embed=embed)
        else:
            last_message = await channel.send(embed=embed)
    except discord.NotFound:
        last_message = await channel.send(embed=embed)

@bot.event
async def on_ready():
    print(f"✅ Bot ist eingeloggt als {bot.user}")
    print(f"📝 Prefix: {bot.command_prefix}")
    print(f"📋 Verfügbare Befehle: {', '.join([cmd.name for cmd in bot.commands])}")
    
    # Lösche alle alten Slash Commands
    try:
        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()
        print("🗑️ Alle alten Slash Commands gelöscht")
    except Exception as e:
        print(f"⚠️ Fehler beim Löschen der Slash Commands: {e}")
    
    if not update_entry_count.is_running():
        update_entry_count.start()

bot.run(TOKEN)
