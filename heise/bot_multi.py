import discord
import asyncpg
import os
from discord.ext import tasks, commands
from discord import app_commands
from dotenv import load_dotenv
import time
from datetime import datetime, timezone, timedelta
from typing import Literal

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

last_message_heise = None
last_message_chip = None

# Typ für Quelle (Source)
SourceType = Literal["heise", "chip", "beide"]

def get_category_field(table: str) -> str:
    """Gibt das richtige Kategoriefeld für die Tabelle zurück"""
    if table == "chip":
        return "page_level1 as category"
    return "category"

def get_category_condition(table: str, param_num: int) -> str:
    """Gibt die richtige WHERE-Bedingung für Kategorien zurück"""
    if table == "chip":
        return f"(page_level1 ILIKE ${param_num} OR page_level2 ILIKE ${param_num} OR page_level3 ILIKE ${param_num})"
    return f"category ILIKE ${param_num}"

async def get_entry_count(table="heise"):
    """Zählt Einträge in der angegebenen Tabelle"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table};")
        await conn.close()
        return count
    except Exception as e:
        print(f"❌ Fehler bei der DB-Abfrage ({table}): {e}")
        return None

async def get_author_count(table="heise"):
    """Zählt einzigartige Autoren"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        authors = await conn.fetch(f"SELECT author FROM {table} WHERE author IS NOT NULL;")  
        await conn.close()

        author_list = []
        for row in authors:
            if row["author"]:
                author_list.extend(row["author"].split(","))

        unique_authors = set(a.strip() for a in author_list if a.strip())
        return len(unique_authors)
    except Exception as e:
        print(f"❌ Fehler bei der Autoren-Abfrage ({table}): {e}")
        return None

async def get_today_counts(table="heise"):
    """Zählt heutige Artikel und Autoren"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        today_date = datetime.now(timezone.utc).date()

        rows = await conn.fetch(f"SELECT date, author FROM {table} WHERE date IS NOT NULL;")
        await conn.close()

        article_count = 0
        author_list = []

        for row in rows:
            try:
                article_date = datetime.fromisoformat(row["date"]).date()
                if article_date == today_date:
                    article_count += 1
                    if row["author"]:
                        author_list.extend(row["author"].split(","))
            except (ValueError, AttributeError):
                continue

        unique_authors_today = set(a.strip() for a in author_list if a.strip())
        return article_count, len(unique_authors_today)
    except Exception as e:
        print(f"❌ Fehler bei der heutigen Abfrage ({table}): {e}")
        return None, None

async def search_articles(table="heise", author_name=None, keywords=None, category=None, 
                         title_content=None, start_date=None, end_date=None,
                         min_words=None, max_words=None, limit=10):
    """Universal-Suchfunktion für Artikel"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = []
        params = []
        param_count = 0
        
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
            conditions.append(get_category_condition(table, param_count))
            params.append(f"%{category}%")
        
        if title_content:
            param_count += 1
            conditions.append(f"title ILIKE ${param_count}")
            params.append(f"%{title_content}%")
        
        if start_date:
            param_count += 1
            conditions.append(f"date >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"date <= ${param_count}")
            params.append(end_date)
        
        if min_words is not None and table == "heise":  # word_count nur bei heise
            param_count += 1
            conditions.append(f"word_count >= ${param_count}")
            params.append(min_words)
        
        if max_words is not None and table == "heise":
            param_count += 1
            conditions.append(f"word_count <= ${param_count}")
            params.append(max_words)
        
        if not conditions:
            conditions.append("TRUE")
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions)
        
        # Spalten je nach Tabelle
        if table == "heise":
            query = f"""
                SELECT title, url, date, author, category, word_count 
                FROM {table} 
                WHERE {where_clause}
                ORDER BY date DESC 
                LIMIT ${param_count}
            """
        else:  # chip
            query = f"""
                SELECT title, url, date, author, {get_category_field(table)}, description
                FROM {table} 
                WHERE {where_clause}
                ORDER BY date DESC 
                LIMIT ${param_count}
            """
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        return rows
    except Exception as e:
        print(f"❌ Fehler bei der Suche ({table}): {e}")
        return []

async def get_top_authors(table="heise", category=None, keywords=None, limit=10):
    """Zeigt die aktivsten Autoren"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        conditions = ["author IS NOT NULL"]
        params = []
        param_count = 0
        
        if category:
            param_count += 1
            conditions.append(get_category_condition(table, param_count))
            params.append(f"%{category}%")
        
        if keywords:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR keywords ILIKE ${param_count})")
            params.append(f"%{keywords}%")
        
        where_clause = " AND ".join(conditions)
        query = f"SELECT author FROM {table} WHERE {where_clause}"
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        
        author_count = {}
        for row in rows:
            if row["author"]:
                authors = row["author"].split(",")
                for author in authors:
                    author = author.strip()
                    if author:
                        author_count[author] = author_count.get(author, 0) + 1
        
        sorted_authors = sorted(author_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_authors[:limit]
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Top-Autoren ({table}): {e}")
        return []

async def get_all_categories(table="heise", limit=50):
    """Holt alle verfügbaren Kategorien"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        if table == "chip":
            # Für chip alle drei Level zusammenfassen
            query = f"""
                SELECT page_level1 as category, COUNT(*) as count 
                FROM {table} 
                WHERE page_level1 IS NOT NULL AND page_level1 != '' 
                GROUP BY page_level1 
                ORDER BY count DESC 
                LIMIT $1
            """
        else:
            query = f"""
                SELECT category, COUNT(*) as count 
                FROM {table} 
                WHERE category IS NOT NULL AND category != '' 
                GROUP BY category 
                ORDER BY count DESC 
                LIMIT $1
            """
        
        rows = await conn.fetch(query, limit)
        await conn.close()
        return [(row["category"], row["count"]) for row in rows]
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Kategorien ({table}): {e}")
        return []

@tasks.loop(minutes=10)
async def update_entry_count():
    global last_message_heise, last_message_chip
    print("🔄 update_entry_count läuft...")

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print("⚠️ Fehler: Kanal nicht gefunden.")
        return

    # Update für beide Quellen
    for table, last_msg_var in [("heise", "last_message_heise"), ("chip", "last_message_chip")]:
        count = await get_entry_count(table)
        author_count = await get_author_count(table)
        today_articles, today_authors = await get_today_counts(table)

        if count is None or author_count is None or today_articles is None or today_authors is None:
            continue

        source_name = "Heise" if table == "heise" else "Chip"
        color = 0x3498db if table == "heise" else 0xe67e22
        
        embed = discord.Embed(
            title=f"📊 **{source_name} Datenbank-Statistiken**",
            description="🔄 **Automatische Updates alle 10 Minuten**",
            color=color
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
            name="🗄️ **Gesamt**",
            value=f"```📚 {count:,} Artikel\n👥 {author_count:,} Autoren```",
            inline=True
        )
        
        embed.add_field(
            name="🕒 **Letzte Aktualisierung**", 
            value=f"<t:{int(time.time())}:R>", 
            inline=False
        )
        
        embed.set_footer(
            text=f"🤖 Automatische Überwachung der {source_name}-Datenbank", 
            icon_url="https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
        )

        try:
            last_msg = globals()[last_msg_var]
            if last_msg:
                await last_msg.edit(embed=embed)
            else:
                new_msg = await channel.send(embed=embed)
                globals()[last_msg_var] = new_msg
        except discord.NotFound:
            new_msg = await channel.send(embed=embed)
            globals()[last_msg_var] = new_msg
        except Exception as e:
            print(f"❌ Fehler beim Update ({table}): {e}")

# Slash Commands mit Quelle-Parameter
@bot.tree.command(name="autor", description="Suche Artikel von einem bestimmten Autor")
@app_commands.describe(
    name="Name des Autors (z.B. 'Jan Mahn')",
    quelle="Quelle: heise, chip oder beide",
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    stichwort="Optionales Stichwort für weitere Filterung",
    kategorie="Optionale Kategorie für weitere Filterung"
)
async def search_by_author(interaction: discord.Interaction, name: str, 
                          quelle: Literal["heise", "chip", "beide"] = "heise",
                          anzahl: int = 10, stichwort: str = None, kategorie: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    all_articles = []
    tables = ["heise", "chip"] if quelle == "beide" else [quelle]
    
    for table in tables:
        articles = await search_articles(table, author_name=name, keywords=stichwort, 
                                        category=kategorie, limit=anzahl)
        for article in articles:
            all_articles.append((article, table))
    
    # Nach Datum sortieren
    all_articles.sort(key=lambda x: x[0]['date'] if x[0]['date'] else "", reverse=True)
    all_articles = all_articles[:anzahl]
    
    if not all_articles:
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel von Autor '{name}' in {quelle} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    title_parts = [f"📰 Artikel von '{name}'"]
    if quelle:
        title_parts.append(f"📍 aus {quelle}")
    
    embed = discord.Embed(
        title=" ".join(title_parts)[:256],
        description=f"🎯 **{len(all_articles)} Artikel gefunden**",
        color=0x2ecc71
    )
    
    for i, (article, source) in enumerate(all_articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        source_icon = "🔵" if source == "heise" else "🟠"
        
        title_formatted = f"**{article['title'][:70]}{'...' if len(article['title']) > 70 else ''}**"
        
        meta_info = [f"{source_icon} `{source.upper()}`", f"🗓️ `{date_str}`"]
        if article.get('category'):
            meta_info.append(f"📁 `{article['category']}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    if len(all_articles) > 10:
        embed.set_footer(text=f"📄 {len(all_articles) - 10} weitere Artikel verfügbar")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="stichwort", description="Suche Artikel nach Stichworten")
@app_commands.describe(
    keyword="Stichwort zum Suchen",
    quelle="Quelle: heise, chip oder beide",
    anzahl="Anzahl der Ergebnisse (Standard: 10)",
    kategorie="Optionale Kategorie"
)
async def search_by_keyword(interaction: discord.Interaction, keyword: str,
                           quelle: Literal["heise", "chip", "beide"] = "heise",
                           anzahl: int = 10, kategorie: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    all_articles = []
    tables = ["heise", "chip"] if quelle == "beide" else [quelle]
    
    for table in tables:
        articles = await search_articles(table, keywords=keyword, category=kategorie, limit=anzahl)
        for article in articles:
            all_articles.append((article, table))
    
    all_articles.sort(key=lambda x: x[0]['date'] if x[0]['date'] else "", reverse=True)
    all_articles = all_articles[:anzahl]
    
    if not all_articles:
        embed = discord.Embed(
            title="🔍 Keine Artikel gefunden",
            description=f"Keine Artikel mit '{keyword}' in {quelle} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"🔍 Artikel mit '{keyword}' aus {quelle}",
        description=f"🎯 **{len(all_articles)} Artikel gefunden**",
        color=0x3498db
    )
    
    for i, (article, source) in enumerate(all_articles[:10], 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        source_icon = "🔵" if source == "heise" else "🟠"
        
        title_formatted = f"**{article['title'][:70]}{'...' if len(article['title']) > 70 else ''}**"
        
        meta_info = [f"{source_icon} `{source.upper()}`", f"🗓️ `{date_str}`"]
        if article.get('category'):
            meta_info.append(f"📁 `{article['category']}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="neueste", description="Zeige die neuesten Artikel")
@app_commands.describe(
    quelle="Quelle: heise, chip oder beide",
    anzahl="Anzahl der Artikel (Standard: 10)"
)
async def get_latest_articles(interaction: discord.Interaction,
                             quelle: Literal["heise", "chip", "beide"] = "heise",
                             anzahl: int = 10):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    all_articles = []
    tables = ["heise", "chip"] if quelle == "beide" else [quelle]
    
    for table in tables:
        articles = await search_articles(table, limit=anzahl)
        for article in articles:
            all_articles.append((article, table))
    
    all_articles.sort(key=lambda x: x[0]['date'] if x[0]['date'] else "", reverse=True)
    all_articles = all_articles[:anzahl]
    
    if not all_articles:
        embed = discord.Embed(
            title="❌ Fehler",
            description="Keine Artikel gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"🆕 Neueste Artikel aus {quelle}",
        description=f"🎯 **Die {len(all_articles)} neuesten Artikel**",
        color=0xf39c12
    )
    
    for i, (article, source) in enumerate(all_articles, 1):
        date_str = article['date'][:10] if article['date'] else "Unbekannt"
        source_icon = "🔵" if source == "heise" else "🟠"
        
        title_formatted = f"**{article['title'][:70]}{'...' if len(article['title']) > 70 else ''}**"
        
        meta_info = [f"{source_icon} `{source.upper()}`", f"🗓️ `{date_str}`"]
        if article.get('category'):
            meta_info.append(f"📁 `{article['category']}`")
        
        embed.add_field(
            name=f"{i:02d}. {title_formatted}",
            value=f"{' • '.join(meta_info)}\n🔗 **[Artikel öffnen]({article['url']})**",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="topautoren", description="Zeige die aktivsten Autoren")
@app_commands.describe(
    quelle="Quelle: heise, chip oder beide",
    anzahl="Anzahl der Autoren (Standard: 10)",
    kategorie="Optionale Kategorie",
    stichwort="Optionales Stichwort"
)
async def get_top_authors_command(interaction: discord.Interaction,
                                 quelle: Literal["heise", "chip", "beide"] = "heise",
                                 anzahl: int = 10, kategorie: str = None, stichwort: str = None):
    await interaction.response.defer()
    
    if anzahl > 25:
        anzahl = 25
    
    all_authors = {}
    tables = ["heise", "chip"] if quelle == "beide" else [quelle]
    
    for table in tables:
        authors = await get_top_authors(table, kategorie, stichwort, anzahl * 2)
        for author, count in authors:
            all_authors[author] = all_authors.get(author, 0) + count
    
    sorted_authors = sorted(all_authors.items(), key=lambda x: x[1], reverse=True)[:anzahl]
    
    if not sorted_authors:
        embed = discord.Embed(
            title="❌ Keine Autoren gefunden",
            description=f"Keine Autoren in {quelle} gefunden.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"🏆 Top Autoren aus {quelle}",
        description=f"🎯 **Die {len(sorted_authors)} aktivsten Autoren**",
        color=0xe67e22
    )
    
    for i, (author, count) in enumerate(sorted_authors, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"`{i:02d}.`"
        
        embed.add_field(
            name=f"{medal} **{author}**",
            value=f"📰 `{count} Artikel`",
            inline=True
        )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="kategorien", description="Zeige alle verfügbaren Kategorien")
@app_commands.describe(
    quelle="Quelle: heise oder chip",
    anzahl="Anzahl der Kategorien (Standard: 20)"
)
async def list_categories(interaction: discord.Interaction,
                         quelle: Literal["heise", "chip"] = "heise",
                         anzahl: int = 20):
    await interaction.response.defer()
    
    if anzahl > 50:
        anzahl = 50
    
    categories = await get_all_categories(quelle, anzahl)
    
    if not categories:
        embed = discord.Embed(
            title="❌ Keine Kategorien gefunden",
            description=f"Keine Kategorien in {quelle} verfügbar.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"📂 Verfügbare Kategorien aus {quelle}",
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
    
    embed.set_footer(text=f"💡 Verwende /stichwort mit quelle:{quelle} für gezielte Suche!")
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="hilfe", description="Zeige alle verfügbaren Commands")
async def show_help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Bot Commands Übersicht",
        description="🔵 = Heise | 🟠 = Chip | Wähle 'beide' für kombinierte Suche",
        color=0x3498db
    )
    
    embed.add_field(
        name="📝 `/autor`",
        value="Suche Artikel von einem Autor\n`/autor name:Mahn quelle:beide`",
        inline=False
    )
    
    embed.add_field(
        name="🔍 `/stichwort`",
        value="Suche nach Stichworten\n`/stichwort keyword:KI quelle:chip`",
        inline=False
    )
    
    embed.add_field(
        name="🆕 `/neueste`",
        value="Neueste Artikel anzeigen\n`/neueste quelle:beide anzahl:15`",
        inline=False
    )
    
    embed.add_field(
        name="🏆 `/topautoren`",
        value="Aktivste Autoren\n`/topautoren quelle:heise`",
        inline=False
    )
    
    embed.add_field(
        name="📂 `/kategorien`",
        value="Alle Kategorien anzeigen\n`/kategorien quelle:chip`",
        inline=False
    )
    
    embed.add_field(
        name="❓ `/hilfe`",
        value="Diese Hilfe anzeigen",
        inline=False
    )
    
    embed.set_footer(text="💡 Nutze quelle:beide für Suche in Heise + Chip!")
    
    await interaction.response.send_message(embed=embed)

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
