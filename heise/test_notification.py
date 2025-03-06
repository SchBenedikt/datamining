import os
from notification import send_notification

def main():
    subject = "Test Email"
    body = "Dies ist eine Testnachricht."
    # use.env for ALERT_EMAIL
    to_email = os.getenv('ALERT_EMAIL')
    send_notification(subject, body, to_email)

if __name__ == "__main__":
    main()
