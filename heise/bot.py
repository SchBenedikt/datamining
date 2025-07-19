import discord
import asyncpg
import os
from discord.ext import tasks, commands
from discord import app_commands
from dotenv import load_dotenv
import time
from datetime import datetime, timezone, timedelta

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
bot = commands.Bot(command_prefix='!', intents=intents)

last_message = None  # Speichert die letzte Nachricht

async def get_entry_count():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        count = await conn.fetchval("SELECT COUNT(*) FROM articles;")
        await conn.close()
        return count
    except Exception as e:
        print(f"❌ Fehler bei der DB-Abfrage: {e}")
        return None

async def get_author_count():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        authors = await conn.fetch("SELECT author FROM articles;")  
        await conn.close()

        author_list = []
        for row in authors:
            author_list.extend(row["author"].split(","))

        unique_authors = set(a.strip() for a in author_list)
        return len(unique_authors)
    except Exception as e:
        print(f"❌ Fehler bei der Autoren-Abfrage: {e}")
        return None

async def get_today_counts():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        today_date = datetime.now(timezone.utc).date()

        rows = await conn.fetch("SELECT date, author FROM articles;")
        await conn.close()

        article_count = 0
        author_list = []

        for row in rows:
            article_date = datetime.fromisoformat(row["date"]).date()
            if article_date == today_date:
                article_count += 1
                author_list.extend(row["author"].split(","))

        unique_authors_today = set(a.strip() for a in author_list)
        return article_count, len(unique_authors_today)
    except Exception as e:
        print(f"❌ Fehler bei der heutigen Abfrage: {e}")
        return None, None

async def search_articles_by_author(author_name, keywords=None, category=None, limit=10):
    """Sucht Artikel eines bestimmten Autors mit optionalen Filtern"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = ["author ILIKE $1"]
        params = [f"%{author_name}%"]
        param_count = 1
        
        if keywords:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{keywords}%")
        
        if category:
            param_count += 1
            conditions.append(f"category ILIKE ${param_count}")
            params.append(f"%{category}%")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT title, url, date, author, category 
            FROM articles 
            WHERE {where_clause}
            ORDER BY date DESC 
            LIMIT ${param_count}
        """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Autoren-Suche: {e}")
        return []

async def search_articles_by_keyword(keyword, category=None, limit=10):
    """Sucht Artikel nach Stichworten im Titel oder in den Keywords mit optionalem Kategorie-Filter"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = ["(title ILIKE $1 OR keywords ILIKE $1)"]
        params = [f"%{keyword}%"]
        param_count = 1
        
        if category:
            param_count += 1
            conditions.append(f"category ILIKE ${param_count}")
            params.append(f"%{category}%")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT title, url, date, author, category 
            FROM articles 
            WHERE {where_clause}
            ORDER BY date DESC 
            LIMIT ${param_count}
        """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Keyword-Suche: {e}")
        return []

async def search_articles_by_category(category, keywords=None, limit=10):
    """Sucht Artikel nach Kategorie mit optionalem Stichwort-Filter"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = ["category ILIKE $1"]
        params = [f"%{category}%"]
        param_count = 1
        
        if keywords:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{keywords}%")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT title, url, date, author, category 
            FROM articles 
            WHERE {where_clause}
            ORDER BY date DESC 
            LIMIT ${param_count}
        """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Kategorie-Suche: {e}")
        return []

async def get_recent_articles(limit=10):
    """Holt die neuesten Artikel"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        query = """
            SELECT title, url, date, author, category 
            FROM articles 
            ORDER BY date DESC 
            LIMIT $1
        """
        rows = await conn.fetch(query, limit)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der neuesten Artikel: {e}")
        return []

async def search_articles_by_title(title_content, limit=10):
    """Sucht Artikel nach Titel-Inhalt"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        query = """
            SELECT title, url, date, author, category 
            FROM articles 
            WHERE title ILIKE $1 
            ORDER BY date DESC 
            LIMIT $2
        """
        rows = await conn.fetch(query, f"%{title_content}%", limit)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Titel-Suche: {e}")
        return []

async def get_top_authors(category=None, keywords=None, limit=10):
    """Zeigt die aktivsten Autoren mit optionalen Filtern"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = ["author IS NOT NULL"]
        params = []
        param_count = 0
        
        if category:
            param_count += 1
            conditions.append(f"category ILIKE ${param_count}")
            params.append(f"%{category}%")
        
        if keywords:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{keywords}%")
        
        where_clause = " AND ".join(conditions)
        query = f"SELECT author FROM articles WHERE {where_clause}"
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        
        author_count = {}
        for row in rows:
            authors = row["author"].split(",")
            for author in authors:
                author = author.strip()
                if author:
                    author_count[author] = author_count.get(author, 0) + 1
        
        sorted_authors = sorted(author_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_authors[:limit]
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Top-Autoren: {e}")
        return []

async def search_articles_by_date_range(start_date, end_date, author_name=None, keywords=None, category=None, limit=10):
    """Sucht Artikel in einem bestimmten Zeitraum mit optionalen Filtern"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = ["date >= $1", "date <= $2"]
        params = [start_date, end_date]
        param_count = 2
        
        if author_name:
            param_count += 1
            conditions.append(f"author ILIKE ${param_count}")
            params.append(f"%{author_name}%")
        
        if keywords:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{keywords}%")
        
        if category:
            param_count += 1
            conditions.append(f"category ILIKE ${param_count}")
            params.append(f"%{category}%")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT title, url, date, author, category, word_count 
            FROM articles 
            WHERE {where_clause}
            ORDER BY date DESC 
            LIMIT ${param_count}
        """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Datums-Suche: {e}")
        return []

async def search_articles_by_word_count(min_words=None, max_words=None, author_name=None, keywords=None, category=None, limit=10):
    """Sucht Artikel nach Wortanzahl mit optionalen Filtern"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = []
        params = []
        param_count = 0
        
        if min_words is not None:
            param_count += 1
            conditions.append(f"word_count >= ${param_count}")
            params.append(min_words)
        
        if max_words is not None:
            param_count += 1
            conditions.append(f"word_count <= ${param_count}")
            params.append(max_words)
        
        if author_name:
            param_count += 1
            conditions.append(f"author ILIKE ${param_count}")
            params.append(f"%{author_name}%")
        
        if keywords:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{keywords}%")
        
        if category:
            param_count += 1
            conditions.append(f"category ILIKE ${param_count}")
            params.append(f"%{category}%")
        
        if not conditions:
            conditions.append("word_count IS NOT NULL")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT title, url, date, author, category, word_count 
            FROM articles 
            WHERE {where_clause}
            ORDER BY word_count DESC 
            LIMIT ${param_count}
        """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Wortanzahl-Suche: {e}")
        return []

async def search_articles_advanced(title_keyword=None, author_name=None, category=None, 
                                 start_date=None, end_date=None, min_words=None, max_words=None, limit=10):
    """Erweiterte Suche mit mehreren Filtern"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = []
        params = []
        param_count = 0
        
        if title_keyword:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{title_keyword}%")
        
        if author_name:
            param_count += 1
            conditions.append(f"author ILIKE ${param_count}")
            params.append(f"%{author_name}%")
        
        if category:
            param_count += 1
            conditions.append(f"category ILIKE ${param_count}")
            params.append(f"%{category}%")
        
        if start_date:
            param_count += 1
            conditions.append(f"date >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"date <= ${param_count}")
            params.append(end_date)
        
        if min_words is not None:
            param_count += 1
            conditions.append(f"word_count >= ${param_count}")
            params.append(min_words)
        
        if max_words is not None:
            param_count += 1
            conditions.append(f"word_count <= ${param_count}")
            params.append(max_words)
        
        if not conditions:
            conditions.append("TRUE")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT title, url, date, author, category, word_count 
            FROM articles 
            WHERE {where_clause}
            ORDER BY date DESC 
            LIMIT ${param_count}
        """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der erweiterten Suche: {e}")
        return []

async def get_articles_stats_by_period(days=30):
    """Statistiken für einen bestimmten Zeitraum"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        from_date = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        
        query = """
            SELECT COUNT(*) as article_count,
                   COUNT(DISTINCT author) as unique_authors,
                   AVG(word_count) as avg_word_count,
                   MAX(word_count) as max_word_count,
                   MIN(word_count) as min_word_count
            FROM articles 
            WHERE date >= $1 AND word_count IS NOT NULL
        """
        
        stats = await conn.fetchrow(query, str(from_date))
        await conn.close()
        return stats
    except Exception as e:
        print(f"❌ Fehler bei den Periode-Statistiken: {e}")
        return None

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
        title="📊 **Live Datenbank-Statistiken**",
        description="🔄 **Automatische Updates alle 10 Minuten**",
        color=0x3498db
    )

    embed.add_field(
        name="🌅 **Heute**",
        value=f"```📰 {today_articles:,} Artikel\n✍️ {today_authors:,} Autoren```",
        inline=True
    )

    embed.add_field(
        name="\u200B",
        value="\u200B",
        inline=True
    )

    embed.add_field(
        name="�️ **Gesamt**",
        value=f"```📚 {count:,} Artikel\n👥 {author_count:,} Autoren```",
        inline=True
    )
    
    embed.add_field(
        name="� **Letzte Aktualisierung**", 
        value=f"<t:{int(time.time())}:R>", 
        inline=False
    )
    
    embed.set_footer(
        text="🤖 Automatische Überwachung der Heise-Datenbank", 
        icon_url="https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
    )

    try:
        if last_message:
            await last_message.edit(embed=embed)
        else:
            last_message = await channel.send(embed=embed)
    except discord.NotFound:
        last_message = await channel.send(embed=embed)

async def get_all_keywords(limit=50):
    """Holt alle verfügbaren Schlagwörter aus der Datenbank"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch("SELECT keywords FROM articles WHERE keywords IS NOT NULL AND keywords != '';")
        await conn.close()
        
        keyword_count = {}
        for row in rows:
            keywords = row["keywords"].split(",")
            for keyword in keywords:
                keyword = keyword.strip()
                if keyword:
                    keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        
        # Sortiere nach Häufigkeit
        sorted_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:limit]
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Schlagwörter: {e}")
        return []

async def get_all_categories(limit=50):
    """Holt alle verfügbaren Kategorien aus der Datenbank"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch("SELECT category, COUNT(*) as count FROM articles WHERE category IS NOT NULL AND category != '' GROUP BY category ORDER BY count DESC LIMIT $1;", limit)
        await conn.close()
        return [(row["category"], row["count"]) for row in rows]
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Kategorien: {e}")
        return []

async def get_all_authors(limit=50):
    """Holt alle verfügbaren Autoren aus der Datenbank"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch("SELECT author FROM articles WHERE author IS NOT NULL AND author != '';")
        await conn.close()
        
        author_count = {}
        for row in rows:
            authors = row["author"].split(",")
            for author in authors:
                author = author.strip()
                if author:
                    author_count[author] = author_count.get(author, 0) + 1
        
        # Sortiere nach Häufigkeit
        sorted_authors = sorted(author_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_authors[:limit]
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Autoren: {e}")
        return []

# Slash Commands
@bot.tree.command(name="autor", description="Suche Artikel von einem bestimmten Autor")
@app_commands.describe(
    name="Name des Autors (z.B. 'Jan Mahn')",
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    stichwort="Optionales Stichwort für weitere Filterung",
    kategorie="Optionale Kategorie für weitere Filterung"
)
async def search_by_author(interaction: discord.Interaction, name: str, anzahl: int = 10, 
                          stichwort: str = None, kategorie: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    articles = await search_articles_by_author(name, stichwort, kategorie, anzahl)
    
    if not articles:
        filter_text = ""
        if stichwort or kategorie:
            filters = []
            if stichwort:
                filters.append(f"Stichwort: '{stichwort}'")
            if kategorie:
                filters.append(f"Kategorie: '{kategorie}'")
            filter_text = f" mit {' und '.join(filters)}"
        
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel von Autor '{name}'{filter_text} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Titel für das Embed erstellen
    title_parts = [f"📰 Artikel von '{name}'"]
    if stichwort:
        title_parts.append(f"� mit '{stichwort}'")
    if kategorie:
        title_parts.append(f"📂 in '{kategorie}'")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],  # Discord Titel-Limit
        description=f"🎯 **{len(articles)} Artikel gefunden**",
        color=0x2ecc71
    )
    
    for i, article in enumerate(articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:75]}{'...' if len(article['title']) > 75 else ''}**"
        
        meta_info = []
        meta_info.append(f"�️ `{date_str}`")
        meta_info.append(f"� `{article['category'] or 'Uncategorized'}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 10:
        embed.set_footer(text=f"📄 {len(articles) - 10} weitere Artikel verfügbar • Verfeinere deine Suche für spezifischere Ergebnisse")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="stichwort", description="Suche Artikel nach Stichworten")
@app_commands.describe(
    keyword="Stichwort zum Suchen",
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    kategorie="Optionale Kategorie für weitere Filterung"
)
async def search_by_keyword(interaction: discord.Interaction, keyword: str, anzahl: int = 10, 
                           kategorie: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    articles = await search_articles_by_keyword(keyword, kategorie, anzahl)
    
    if not articles:
        filter_text = ""
        if kategorie:
            filter_text = f" in Kategorie '{kategorie}'"
        
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel mit Stichwort '{keyword}'{filter_text} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Titel für das Embed erstellen
    title_parts = [f"🔍 Artikel mit '{keyword}'"]
    if kategorie:
        title_parts.append(f"📂 in '{kategorie}'")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],  # Discord Titel-Limit
        description=f"🎯 **{len(articles)} Artikel gefunden**",
        color=0x3498db
    )
    
    for i, article in enumerate(articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:75]}{'...' if len(article['title']) > 75 else ''}**"
        
        meta_info = []
        meta_info.append(f"�️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        meta_info.append(f"� `{article['category'] or 'Uncategorized'}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 10:
        embed.set_footer(text=f"📄 {len(articles) - 10} weitere Artikel verfügbar • Verfeinere deine Suche für spezifischere Ergebnisse")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="kategorie", description="Suche Artikel nach Kategorie")
@app_commands.describe(
    kategorie="Name der Kategorie",
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    stichwort="Optionales Stichwort für weitere Filterung"
)
async def search_by_category(interaction: discord.Interaction, kategorie: str, anzahl: int = 10, 
                            stichwort: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    articles = await search_articles_by_category(kategorie, stichwort, anzahl)
    
    if not articles:
        filter_text = ""
        if stichwort:
            filter_text = f" mit Stichwort '{stichwort}'"
        
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel in Kategorie '{kategorie}'{filter_text} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Titel für das Embed erstellen
    title_parts = [f"📂 Artikel aus '{kategorie}'"]
    if stichwort:
        title_parts.append(f"🔍 mit '{stichwort}'")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],  # Discord Titel-Limit
        description=f"🎯 **{len(articles)} Artikel gefunden**",
        color=0x9b59b6
    )
    
    for i, article in enumerate(articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:75]}{'...' if len(article['title']) > 75 else ''}**"
        
        meta_info = []
        meta_info.append(f"�️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 10:
        embed.set_footer(text=f"📄 {len(articles) - 10} weitere Artikel verfügbar • Verfeinere deine Suche für spezifischere Ergebnisse")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="titel", description="Suche Artikel nach Titel-Inhalt")
@app_commands.describe(
    inhalt="Suchbegriff im Titel (z.B. 'Nextcloud')",
    anzahl="Anzahl der Ergebnisse (Standard: 10)"
)
async def search_by_title(interaction: discord.Interaction, inhalt: str, anzahl: int = 10):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    articles = await search_articles_by_title(inhalt, anzahl)
    
    if not articles:
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel mit '{inhalt}' im Titel gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"📝 Artikel mit '{inhalt}' im Titel",
        description=f"🎯 **{len(articles)} Artikel gefunden**",
        color=0x1abc9c
    )
    
    for i, article in enumerate(articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:75]}{'...' if len(article['title']) > 75 else ''}**"
        
        meta_info = []
        meta_info.append(f"�️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        meta_info.append(f"� `{article['category'] or 'Uncategorized'}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 10:
        embed.set_footer(text=f"📄 {len(articles) - 10} weitere Artikel verfügbar • Verfeinere deine Suche für spezifischere Ergebnisse")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="neueste", description="Zeige die neuesten Artikel")
@app_commands.describe(anzahl="Anzahl der Artikel (Standard: 10)")
async def get_latest_articles(interaction: discord.Interaction, anzahl: int = 10):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    articles = await get_recent_articles(anzahl)
    
    if not articles:
        embed = discord.Embed(
            title="❌ Fehler",
            description="Keine Artikel gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title="🆕 Neueste Artikel",
        description=f"🎯 **Die {len(articles)} neuesten Artikel**",
        color=0xf39c12
    )
    
    for i, article in enumerate(articles, 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:75]}{'...' if len(article['title']) > 75 else ''}**"
        
        meta_info = []
        meta_info.append(f"�️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        meta_info.append(f"� `{article['category'] or 'Uncategorized'}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="topautoren", description="Zeige die aktivsten Autoren")
@app_commands.describe(
    anzahl="Anzahl der Autoren (Standard: 10)",
    kategorie="Optionale Kategorie für Filterung",
    stichwort="Optionales Stichwort für Filterung"
)
async def get_top_authors_command(interaction: discord.Interaction, anzahl: int = 10, 
                                 kategorie: str = None, stichwort: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    authors = await get_top_authors(kategorie, stichwort, anzahl)
    
    if not authors:
        filter_text = ""
        if kategorie or stichwort:
            filters = []
            if kategorie:
                filters.append(f"Kategorie: '{kategorie}'")
            if stichwort:
                filters.append(f"Stichwort: '{stichwort}'")
            filter_text = f" für {' und '.join(filters)}"
        
        embed = discord.Embed(
            title="❌ Keine Autoren gefunden",
            description=f"Keine Autoren{filter_text} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Titel für das Embed erstellen
    title_parts = ["🏆 Top Autoren"]
    if kategorie:
        title_parts.append(f"📂 in '{kategorie}'")
    if stichwort:
        title_parts.append(f"🔍 für '{stichwort}'")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],  # Discord Titel-Limit
        description=f"🎯 **Die {len(authors)} aktivsten Autoren**",
        color=0xe67e22
    )
    
    for i, (author, count) in enumerate(authors, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"`{i:02d}.`"
        
        # Verbesserte Darstellung mit modernem Design
        author_formatted = f"**{author}**"
        article_text = "Artikel" if count == 1 else "Artikel"
        
        embed.add_field(
            name=f"{medal} {author_formatted}",
            value=f"� `{count} {article_text}`",
            inline=True
        )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="zeitraum", description="Suche Artikel aus einem bestimmten Zeitraum")
@app_commands.describe(
    von="Startdatum (YYYY-MM-DD, z.B. 2025-01-01)", 
    bis="Enddatum (YYYY-MM-DD, z.B. 2025-05-25)", 
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    autor="Optionaler Autor-Filter",
    stichwort="Optionales Stichwort für weitere Filterung",
    kategorie="Optionale Kategorie für weitere Filterung"
)
async def search_by_date_range(interaction: discord.Interaction, von: str, bis: str, anzahl: int = 10,
                              autor: str = None, stichwort: str = None, kategorie: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    try:
        start_date = datetime.strptime(von, "%Y-%m-%d").date()
        end_date = datetime.strptime(bis, "%Y-%m-%d").date()
        if start_date > end_date:
            raise ValueError("Startdatum muss vor Enddatum liegen")
    except ValueError as e:
        embed = discord.Embed(
            title="❌ Ungültiges Datum",
            description=f"Bitte verwende das Format YYYY-MM-DD (z.B. 2025-01-01)\nFehler: {str(e)}",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    articles = await search_articles_by_date_range(str(start_date), str(end_date), autor, stichwort, kategorie, anzahl)
    
    if not articles:
        filter_text = ""
        if autor or stichwort or kategorie:
            filters = []
            if autor:
                filters.append(f"Autor: '{autor}'")
            if stichwort:
                filters.append(f"Stichwort: '{stichwort}'")
            if kategorie:
                filters.append(f"Kategorie: '{kategorie}'")
            filter_text = f" mit {' und '.join(filters)}"
        
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel zwischen {von} und {bis}{filter_text} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Titel für das Embed erstellen
    title_parts = [f"📅 Artikel von {von} bis {bis}"]
    if autor:
        title_parts.append(f"👤 von '{autor}'")
    if stichwort:
        title_parts.append(f"🔍 mit '{stichwort}'")
    if kategorie:
        title_parts.append(f"📂 in '{kategorie}'")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],  # Discord Titel-Limit
        description=f"🎯 **{len(articles)} Artikel gefunden**",
        color=0x16a085
    )
    
    for i, article in enumerate(articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:75]}{'...' if len(article['title']) > 75 else ''}**"
        
        meta_info = []
        meta_info.append(f"🗓️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        meta_info.append(f"📁 `{article['category'] or 'Uncategorized'}`")
        if article.get('word_count'):
            meta_info.append(f"� `{article['word_count']} Wörter`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 10:
        embed.set_footer(text=f"📄 {len(articles) - 10} weitere Artikel verfügbar • Verwende Filter für spezifischere Ergebnisse")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="wortanzahl", description="Suche Artikel nach Wortanzahl")
@app_commands.describe(
    min_woerter="Mindest-Wortanzahl (optional)", 
    max_woerter="Höchst-Wortanzahl (optional)", 
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    autor="Optionaler Autor-Filter",
    stichwort="Optionales Stichwort für weitere Filterung",
    kategorie="Optionale Kategorie für weitere Filterung"
)
async def search_by_word_count(interaction: discord.Interaction, min_woerter: int = None, max_woerter: int = None, anzahl: int = 10,
                              autor: str = None, stichwort: str = None, kategorie: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    if min_woerter is None and max_woerter is None:
        embed = discord.Embed(
            title="❌ Parameter fehlen",
            description="Bitte gebe mindestens min_woerter oder max_woerter an.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    articles = await search_articles_by_word_count(min_woerter, max_woerter, autor, stichwort, kategorie, anzahl)
    
    if not articles:
        range_text = ""
        if min_woerter and max_woerter:
            range_text = f"zwischen {min_woerter} und {max_woerter} Wörtern"
        elif min_woerter:
            range_text = f"mit mindestens {min_woerter} Wörtern"
        elif max_woerter:
            range_text = f"mit höchstens {max_woerter} Wörtern"
        
        filter_text = ""
        if autor or stichwort or kategorie:
            filters = []
            if autor:
                filters.append(f"Autor: '{autor}'")
            if stichwort:
                filters.append(f"Stichwort: '{stichwort}'")
            if kategorie:
                filters.append(f"Kategorie: '{kategorie}'")
            filter_text = f" und {' und '.join(filters)}"
        
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel {range_text}{filter_text} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Range Text für Titel
    range_text = ""
    if min_woerter and max_woerter:
        range_text = f"{min_woerter}-{max_woerter} Wörter"
    elif min_woerter:
        range_text = f"≥{min_woerter} Wörter"
    elif max_woerter:
        range_text = f"≤{max_woerter} Wörter"
    
    # Titel für das Embed erstellen
    title_parts = [f"📊 Artikel mit {range_text}"]
    if autor:
        title_parts.append(f"👤 von '{autor}'")
    if stichwort:
        title_parts.append(f"🔍 mit '{stichwort}'")
    if kategorie:
        title_parts.append(f"📂 in '{kategorie}'")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],  # Discord Titel-Limit
        description=f"🎯 **{len(articles)} Artikel gefunden**",
        color=0x8e44ad
    )
    
    for i, article in enumerate(articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        word_count = article['word_count'] or 0
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:70]}{'...' if len(article['title']) > 70 else ''}**"
        
        meta_info = []
        meta_info.append(f"�️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        meta_info.append(f"📊 `{word_count} Wörter`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 10:
        embed.set_footer(text=f"📄 {len(articles) - 10} weitere Artikel verfügbar • Passe Wortanzahl-Filter für präzisere Ergebnisse an")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="erweitert", description="Erweiterte Suche mit mehreren Filtern")
@app_commands.describe(
    titel_stichwort="Stichwort im Titel (optional)",
    autor_name="Autorname (optional)",
    kategorie="Kategorie (optional)",
    von_datum="Startdatum YYYY-MM-DD (optional)",
    bis_datum="Enddatum YYYY-MM-DD (optional)",
    min_woerter="Mindest-Wortanzahl (optional)",
    max_woerter="Höchst-Wortanzahl (optional)",
    anzahl="Anzahl der Ergebnisse (Standard: 10)"
)
async def advanced_search(interaction: discord.Interaction, 
                         titel_stichwort: str = None, autor_name: str = None, kategorie: str = None,
                         von_datum: str = None, bis_datum: str = None, 
                         min_woerter: int = None, max_woerter: int = None, anzahl: int = 10):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    if not any([titel_stichwort, autor_name, kategorie, von_datum, bis_datum, min_woerter, max_woerter]):
        embed = discord.Embed(
            title="❌ Keine Filter angegeben",
            description="Bitte gebe mindestens einen Suchfilter an.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    start_date = None
    end_date = None
    if von_datum:
        try:
            start_date = datetime.strptime(von_datum, "%Y-%m-%d").date()
        except ValueError:
            embed = discord.Embed(
                title="❌ Ungültiges Startdatum",
                description="Bitte verwende das Format YYYY-MM-DD (z.B. 2025-01-01)",
                color=0xe74c3c
            )
            await interaction.followup.send(embed=embed)
            return
    
    if bis_datum:
        try:
            end_date = datetime.strptime(bis_datum, "%Y-%m-%d").date()
        except ValueError:
            embed = discord.Embed(
                title="❌ Ungültiges Enddatum",
                description="Bitte verwende das Format YYYY-MM-DD (z.B. 2025-05-25)",
                color=0xe74c3c
            )
            await interaction.followup.send(embed=embed)
            return
    
    if start_date and end_date and start_date > end_date:
        embed = discord.Embed(
            title="❌ Ungültiger Zeitraum",
            description="Startdatum muss vor Enddatum liegen.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    articles = await search_articles_advanced(titel_stichwort, autor_name, kategorie,
                                            str(start_date) if start_date else None,
                                            str(end_date) if end_date else None,
                                            min_woerter, max_woerter, anzahl)
    
    if not articles:
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description="Keine Artikel entsprechen den angegebenen Suchkriterien.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    filters = []
    if titel_stichwort:
        filters.append(f"🔍 Titel: '{titel_stichwort}'")
    if autor_name:
        filters.append(f"👤 Autor: '{autor_name}'")
    if kategorie:
        filters.append(f"📂 Kategorie: '{kategorie}'")
    if von_datum and bis_datum:
        filters.append(f"📅 Zeitraum: {von_datum} - {bis_datum}")
    elif von_datum:
        filters.append(f"📅 Ab: {von_datum}")
    elif bis_datum:
        filters.append(f"📅 Bis: {bis_datum}")
    if min_woerter and max_woerter:
        filters.append(f"📊 Wörter: {min_woerter}-{max_woerter}")
    elif min_woerter:
        filters.append(f"📊 Min. Wörter: {min_woerter}")
    elif max_woerter:
        filters.append(f"📊 Max. Wörter: {max_woerter}")
    
    filter_text = "\n".join(filters)
    
    embed = discord.Embed(
        title="🔍 Erweiterte Suche",
        description=f"**🎯 Filter aktiv:**\n{filter_text}\n\n**✨ {len(articles)} Artikel gefunden**",
        color=0x2c3e50
    )
    
    for i, article in enumerate(articles[:8], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        
        # Modernere Formatierung mit besserer visueller Struktur
        title_formatted = f"**{article['title'][:65]}{'...' if len(article['title']) > 65 else ''}**"
        
        meta_info = []
        meta_info.append(f"🗓️ `{date_str}`")
        if article.get('author'):
            meta_info.append(f"✍️ `{article['author']}`")
        if article.get('word_count'):
            meta_info.append(f"📊 `{article['word_count']} Wörter`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(articles) > 8:
        embed.set_footer(text=f"📄 {len(articles) - 8} weitere Artikel verfügbar • Nutze spezifischere Filter für gezieltere Ergebnisse")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="statistik", description="Zeige Statistiken für einen bestimmten Zeitraum")
@app_commands.describe(tage="Anzahl Tage rückwirkend (Standard: 30)")
async def get_period_stats(interaction: discord.Interaction, tage: int = 30):
    await interaction.response.defer()
    
    if tage > 365:
        tage = 365
    
    stats = await get_articles_stats_by_period(tage)
    
    if not stats or stats['article_count'] == 0:
        embed = discord.Embed(
            title="📊 Keine Daten",
            description=f"Keine Artikel in den letzten {tage} Tagen gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"📊 Statistiken der letzten {tage} Tage",
        description="📈 **Datenbank-Übersicht**",
        color=0x27ae60
    )
    
    embed.add_field(
        name="📰 Artikel gesamt",
        value=f"```{stats['article_count']:,}```",
        inline=True
    )
    
    embed.add_field(
        name="👥 Einzigartige Autoren",
        value=f"```{stats['unique_authors']:,}```",
        inline=True
    )
    
    embed.add_field(
        name="📈 Ø Artikel/Tag",
        value=f"```{stats['article_count'] / tage:.1f}```",
        inline=True
    )
    
    if stats['avg_word_count']:
        embed.add_field(
            name="📊 Ø Wortanzahl",
            value=f"```{int(stats['avg_word_count']):,} Wörter```",
            inline=True
        )
        
        embed.add_field(
            name="� Längster Artikel",
            value=f"```{stats['max_word_count']:,} Wörter```",
            inline=True
        )
        
        embed.add_field(
            name="� Kürzester Artikel",
            value=f"```{stats['min_word_count']:,} Wörter```",
            inline=True
        )
    
    from_date = (datetime.now(timezone.utc) - timedelta(days=tage)).strftime("%Y-%m-%d")
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    embed.set_footer(text=f"Zeitraum: {from_date} bis {to_date}")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="hilfe", description="Zeige alle verfügbaren Commands")
async def show_help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Bot Commands Übersicht",
        description="Hier sind alle verfügbaren Slash Commands:",
        color=0x3498db
    )
    
    embed.add_field(
        name="📝 `/autor`",
        value="Suche Artikel von einem bestimmten Autor\n`/autor name:Jan Mahn stichwort:KI kategorie:Technik`",
        inline=False
    )
    
    embed.add_field(
        name="🔍 `/stichwort`",
        value="Suche Artikel nach Stichworten im Titel/Keywords\n`/stichwort keyword:KI kategorie:Technik anzahl:10`",
        inline=False
    )
    
    embed.add_field(
        name="📂 `/kategorie`",
        value="Suche Artikel nach Kategorie\n`/kategorie kategorie:Technik stichwort:KI anzahl:10`",
        inline=False
    )
    
    embed.add_field(
        name="📝 `/titel`",
        value="Suche Artikel nach Titel-Inhalt\n`/titel inhalt:Nextcloud anzahl:10`",
        inline=False
    )
    
    embed.add_field(
        name="🆕 `/neueste`",
        value="Zeige die neuesten Artikel\n`/neueste anzahl:5`",
        inline=False
    )
    
    embed.add_field(
        name="🏆 `/topautoren`",
        value="Zeige die aktivsten Autoren\n`/topautoren kategorie:Technik stichwort:KI anzahl:15`",
        inline=False
    )
    
    embed.add_field(
        name="📅 `/zeitraum`",
        value="Suche Artikel aus einem Zeitraum\n`/zeitraum von:2025-01-01 bis:2025-05-25 autor:Mahn kategorie:Technik`",
        inline=False
    )
    
    embed.add_field(
        name="📊 `/wortanzahl`",
        value="Suche nach Wortanzahl\n`/wortanzahl min_woerter:500 max_woerter:1000 autor:Mahn stichwort:KI`",
        inline=False
    )
    
    embed.add_field(
        name="🔍 `/erweitert`",
        value="Erweiterte Suche mit mehreren Filtern\n`/erweitert titel_stichwort:KI autor_name:Mahn`",
        inline=False
    )
    
    embed.add_field(
        name="📊 `/statistik`",
        value="Zeige Statistiken für einen Zeitraum\n`/statistik tage:30`",
        inline=False
    )
    
    embed.add_field(
        name="🏷️ `/schlagwoerter`",
        value="Zeige alle verfügbaren Schlagwörter\n`/schlagwoerter anzahl:30`",
        inline=False
    )
    
    embed.add_field(
        name="📂 `/kategorien`",
        value="Zeige alle verfügbaren Kategorien\n`/kategorien anzahl:20`",
        inline=False
    )
    
    embed.add_field(
        name="✍️ `/autoren`",
        value="Zeige alle verfügbaren Autoren\n`/autoren anzahl:30`",
        inline=False
    )
    
    embed.add_field(
        name="❓ `/hilfe`",
        value="Zeige diese Hilfe an",
        inline=False
    )
    
    embed.set_footer(text="💡 Tipp: Die meisten Parameter sind optional. Probiere verschiedene Kombinationen aus!")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="schlagwoerter", description="Zeige alle verfügbaren Schlagwörter")
@app_commands.describe(anzahl="Anzahl der Schlagwörter (Standard: 30)")
async def list_keywords(interaction: discord.Interaction, anzahl: int = 30):
    await interaction.response.defer()
    
    if anzahl > 50:
        anzahl = 50
    
    keywords = await get_all_keywords(anzahl)
    
    if not keywords:
        embed = discord.Embed(
            title="❌ Keine Schlagwörter gefunden",
            description="Keine Schlagwörter in der Datenbank verfügbar.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title="🏷️ Verfügbare Schlagwörter",
        description=f"🎯 **Die {len(keywords)} häufigsten Schlagwörter**",
        color=0x3498db
    )
    
    # Teile die Schlagwörter in Gruppen auf für bessere Darstellung
    for i in range(0, len(keywords), 10):
        keyword_group = keywords[i:i+10]
        field_value = ""
        
        for j, (keyword, count) in enumerate(keyword_group, i+1):
            field_value += f"`{j:02d}.` **{keyword}** • `{count} Artikel`\n"
        
        group_num = (i // 10) + 1
        embed.add_field(
            name=f"📑 Gruppe {group_num}",
            value=field_value,
            inline=True
        )
    
    embed.set_footer(text="💡 Verwende /stichwort keyword:Schlagwort um nach einem spezifischen Schlagwort zu suchen!")
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="kategorien", description="Zeige alle verfügbaren Kategorien")
@app_commands.describe(anzahl="Anzahl der Kategorien (Standard: 20)")
async def list_categories(interaction: discord.Interaction, anzahl: int = 20):
    await interaction.response.defer()
    
    if anzahl > 50:
        anzahl = 50
    
    categories = await get_all_categories(anzahl)
    
    if not categories:
        embed = discord.Embed(
            title="❌ Keine Kategorien gefunden",
            description="Keine Kategorien in der Datenbank verfügbar.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title="📂 Verfügbare Kategorien",
        description=f"🎯 **Die {len(categories)} aktivsten Kategorien**",
        color=0x9b59b6
    )
    
    for i, (category, count) in enumerate(categories, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"`{i:02d}.`"
        
        embed.add_field(
            name=f"{medal} **{category}**",
            value=f"📰 `{count} Artikel`",
            inline=True
        )
    
    embed.set_footer(text="💡 Verwende /kategorie kategorie:Name um nach einer spezifischen Kategorie zu suchen!")
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="autoren", description="Zeige alle verfügbaren Autoren")
@app_commands.describe(anzahl="Anzahl der Autoren (Standard: 30)")
async def list_authors(interaction: discord.Interaction, anzahl: int = 30):
    await interaction.response.defer()
    
    if anzahl > 50:
        anzahl = 50
    
    authors = await get_all_authors(anzahl)
    
    if not authors:
        embed = discord.Embed(
            title="❌ Keine Autoren gefunden",
            description="Keine Autoren in der Datenbank verfügbar.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title="✍️ Verfügbare Autoren",
        description=f"🎯 **Die {len(authors)} aktivsten Autoren**",
        color=0x2ecc71
    )
    
    # Teile die Autoren in Gruppen auf für bessere Darstellung
    for i in range(0, len(authors), 8):
        author_group = authors[i:i+8]
        field_value = ""
        
        for j, (author, count) in enumerate(author_group, i+1):
            medal = "🥇" if j == 1 else "🥈" if j == 2 else "🥉" if j == 3 else f"`{j:02d}.`"
            field_value += f"{medal} **{author}** • `{count} Artikel`\n"
        
        group_num = (i // 8) + 1
        embed.add_field(
            name=f"👥 Gruppe {group_num}",
            value=field_value,
            inline=True
        )
    
    embed.set_footer(text="💡 Verwende /autor name:Autorname um nach einem spezifischen Autor zu suchen!")
    await interaction.followup.send(embed=embed)

@bot.event
async def on_ready():
    print(f"Bot ist eingeloggt als {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    
    if not update_entry_count.is_running():
        update_entry_count.start()

if __name__ == "__main__":
    bot.run(TOKEN)
