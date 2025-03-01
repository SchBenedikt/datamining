import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from the .env file located in the project directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

def send_notification(subject, body, to_email):
    from_email = os.getenv('EMAIL_USER', 'your_email@example.com')
    from_password = os.getenv('EMAIL_PASSWORD', 'your_email_password')

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.example.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")
