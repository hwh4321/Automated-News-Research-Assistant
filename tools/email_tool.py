"""Email sending tool via SMTP."""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_core.tools import tool

from config import config


@tool
def send_email(to: str, subject: str, body: str, html: bool = False) -> dict[str, str]:
    """发送电子邮件。将报告或摘要通过邮件发送给指定收件人。

    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
        html: 是否为HTML格式邮件
    """
    msg = MIMEMultipart()
    msg["From"] = config.smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html" if html else "plain", "utf-8"))

    with _connect() as server:
        server.login(config.smtp_user, config.smtp_password)
        server.send_message(msg)

    return {"status": "sent", "to": to, "subject": subject}


def _connect() -> smtplib.SMTP:
    if config.smtp_port == 465:
        return smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=30)
    server = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30)
    server.starttls()
    return server
