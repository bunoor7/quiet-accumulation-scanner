"""Optional Telegram delivery for scan results.

Credentials come only from environment variables, never from code or
the command line, so they can live in GitHub Actions secrets:

- TELEGRAM_BOT_TOKEN  the token BotFather gave you
- TELEGRAM_CHAT_ID    the channel, e.g. "@my_channel" (public) or
                      "-100..." (private); the bot must be an admin there
"""
from __future__ import annotations

import os
from typing import Optional

import requests

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 4096  # Telegram's hard limit per message


class TelegramConfigError(RuntimeError):
    """Raised when Telegram delivery is requested but not configured."""


class TelegramSendError(RuntimeError):
    """Raised when Telegram rejects or cannot receive the message."""


def credentials_from_env() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise TelegramConfigError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set"
        )
    return token, chat_id


def send_message(text: str, token: Optional[str] = None, chat_id: Optional[str] = None,
                 timeout_s: float = 15.0) -> None:
    if token is None or chat_id is None:
        token, chat_id = credentials_from_env()
    if len(text) > MAX_LEN:
        text = text[: MAX_LEN - 1] + "…"
    try:
        resp = requests.post(
            API_URL.format(token=token),
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=timeout_s,
        )
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        # Never echo the URL: it contains the bot token.
        raise TelegramSendError(f"telegram: request failed ({type(exc).__name__})") from None
    if not payload.get("ok"):
        raise TelegramSendError(f"telegram: {payload.get('description', 'unknown error')}")
