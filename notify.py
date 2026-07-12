"""Email notifications for scan summaries (Gmail SMTP, same pattern as ca-jepa)."""
import logging
import smtplib
from email.mime.text import MIMEText

from config import GMAIL_APP_PASSWORD, GMAIL_SENDER, SCAN_EMAIL_ENABLED, SCAN_EMAIL_RECIPIENT

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str) -> None:
    """Send a plaintext email via Gmail SMTP. No-ops if disabled or unconfigured."""
    if not SCAN_EMAIL_ENABLED:
        return
    if not GMAIL_APP_PASSWORD or not GMAIL_SENDER or not SCAN_EMAIL_RECIPIENT:
        logger.warning("GMAIL_APP_PASSWORD/GMAIL_SENDER/SCAN_EMAIL_RECIPIENT not set — skipping email")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_SENDER
    msg["To"] = SCAN_EMAIL_RECIPIENT

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
