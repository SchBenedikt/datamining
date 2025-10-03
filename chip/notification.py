import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def send_notification(subject, body, to_email, debug_info=""):
    from_email = os.getenv('EMAIL_USER')
    from_password = os.getenv('EMAIL_PASSWORD')

    # Betreff um aktuelle Zeit ergänzen
    full_subject = f"{subject} - {os.path.basename(__file__)} - {os.getenv('ALERT_EMAIL')}"
    
    msg = MIMEMultipart("alternative")
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = full_subject

    # Plaintext-Version als Fallback
    text = f"{body}\n\nDebug-Informationen:\n{debug_info}\n\nGesendet von Deinem Crawling-System."
    
    # HTML-Version für ansprechendes Design
    html = f"""\
<html>
  <head>
    <style>
      body {{ font-family: Arial, sans-serif; }}
      .header {{ background-color: #007BFF; color: #ffffff; padding: 10px; text-align: center; }}
      .content {{ margin: 20px; }}
      .footer {{ font-size: 0.8rem; color: #777777; margin-top: 20px; text-align: center; }}
      pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #dddddd; }}
    </style>
  </head>
  <body>
    <div class="header">
      <h2>{subject}</h2>
    </div>
    <div class="content">
      <p>{body}</p>
      <hr>
      <p><strong>Debug-Informationen:</strong></p>
      <pre>{debug_info}</pre>
    </div>
    <div class="footer">
      Gesendet von Deinem Crawling-System.
    </div>
  </body>
</html>
"""
    # Beide Versionen dem Message-Objekt hinzufügen
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')
    msg.attach(part1)
    msg.attach(part2)

    try:
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.strato.de')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")
