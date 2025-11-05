# email_sender.py

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import os
from dotenv import load_dotenv


def send_email_with_attachment(to_email: str, subject: str, body: str, attachment_path: str) -> None:
    """Send an email with a PDF attachment.

    Args:
        to_email: Recipient email address.
        subject: Email subject line.
        body: Plain text message body.
        attachment_path: Path to the PDF file to attach.
    """

    # Load credentials from .env file
    load_dotenv()
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        raise ValueError("Email credentials not found. Set SENDER_EMAIL and SENDER_PASSWORD in your .env file")

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Attach the file
    with open(attachment_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(attachment_path)}"')
        msg.attach(part)

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)

    print(f"✅ Email with report sent successfully to {to_email}")
