# notifications.py
import os
import logging
import requests
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GMAIL_EMAIL = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")



# --- Telegram Notification ---
def send_telegram_message(message: str, file_path: str = None):
    """
    Sends a Telegram message.
    Optionally sends a file along with the message.
    """
    try:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not bot_token or not chat_id:
            logger.error("Telegram BOT_TOKEN or CHAT_ID not set.")
            return

        if file_path and os.path.exists(file_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
            with open(file_path, "rb") as f:
                response = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"},
                    files={"document": f}
                )
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            response = requests.post(
                url,
                data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
            )

        response.raise_for_status()
        logger.info("Telegram message sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


# --- Email Notification ---
def send_email(subject: str, body: str, attachment: str = None):
    """
    Sends an email.
    Optionally attaches a file.
    """
    try:
        gmail_address = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        recipient_email = os.getenv("EMAIL_RECEIVER")

        if not all([gmail_address, gmail_password, recipient_email]):
            logger.error("Email credentials not set in environment.")
            return

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = gmail_address
        msg['To'] = recipient_email
        msg.set_content(body)

        # Attach file if provided
        if attachment and os.path.exists(attachment):
            with open(attachment, "rb") as f:
                file_data = f.read()
                file_name = os.path.basename(attachment)
                msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(gmail_address, gmail_password)
            smtp.send_message(msg)

        logger.info(f"Email sent to {recipient_email}")

    except Exception as e:
        logger.error(f"Failed to send email: {e}")


