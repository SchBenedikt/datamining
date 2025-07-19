#!/usr/bin/env python3
"""
Starter-Skript für die Streamlit-basierte Heise Mining Anwendung
"""

import subprocess
import sys
import os

def install_requirements():
    """Installiert die erforderlichen Pakete"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("✅ Alle Abhängigkeiten wurden erfolgreich installiert!")
    except subprocess.CalledProcessError:
        print("❌ Fehler beim Installieren der Abhängigkeiten")
        sys.exit(1)

def start_streamlit():
    """Startet die Streamlit-Anwendung"""
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port=8501",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Anwendung gestoppt")
    except Exception as e:
        print(f"❌ Fehler beim Starten: {e}")

def main():
    """Hauptfunktion"""
    print("🚀 Heise Mining Streamlit App")
    print("="*50)
    
    # Prüfe ob requirements_streamlit.txt existiert
    if not os.path.exists("requirements_streamlit.txt"):
        print("❌ requirements_streamlit.txt nicht gefunden!")
        sys.exit(1)
    
    # Installiere Abhängigkeiten
    print("📦 Installiere Abhängigkeiten...")
    install_requirements()
    
    # Starte Streamlit
    print("🌐 Starte Streamlit-Anwendung...")
    print("📱 Öffne http://localhost:8501 in deinem Browser")
    print("🔄 Drücke Ctrl+C zum Beenden")
    print("="*50)
    
    start_streamlit()

if __name__ == "__main__":
    main()
