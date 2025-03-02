import os
from notification import send_notification

def main():
    subject = "Test Email"
    body = "Dies ist eine Testnachricht."
    # Verwende die in .env gespeicherte E-Mail-Adresse
    to_email = os.getenv('ALERT_EMAIL')
    send_notification(subject, body, to_email)

if __name__ == "__main__":
    main()
