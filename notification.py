import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_notification(subject, body, to_email):
    from_email = os.getenv('EMAIL_USER', 'your_email@example.com')
    from_password = os.getenv('EMAIL_PASSWORD', 'your_email_password')

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.example.com'), int(os.getenv('SMTP_PORT', 587)))
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")
