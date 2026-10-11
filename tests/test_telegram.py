"""Telegram delivery tests: requests.post is mocked, no network."""
from __future__ import annotations

import pytest

from accscan import cli, telegram


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_send_message_posts_to_the_channel(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["url"], seen["json"] = url, json
        return _Resp({"ok": True})

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    telegram.send_message("مرحبا", token="123:ABC", chat_id="@chan")
    assert seen["url"] == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert seen["json"]["chat_id"] == "@chan"
    assert seen["json"]["text"] == "مرحبا"


def test_send_message_truncates_to_telegram_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(telegram.requests, "post",
                        lambda url, json=None, timeout=None: seen.update(json) or _Resp({"ok": True}))
    telegram.send_message("x" * 5000, token="t", chat_id="c")
    assert len(seen["text"]) == telegram.MAX_LEN


def test_send_message_raises_on_telegram_error(monkeypatch):
    monkeypatch.setattr(telegram.requests, "post",
                        lambda *a, **k: _Resp({"ok": False, "description": "chat not found"}))
    with pytest.raises(telegram.TelegramSendError, match="chat not found"):
        telegram.send_message("x", token="t", chat_id="c")


def test_network_error_never_leaks_the_token(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("https://api.telegram.org/botSECRET/sendMessage")

    monkeypatch.setattr(telegram.requests, "post", boom)
    with pytest.raises(telegram.TelegramSendError) as info:
        telegram.send_message("x", token="SECRET", chat_id="c")
    assert "SECRET" not in str(info.value)


def test_missing_credentials_raise_config_error(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(telegram.TelegramConfigError):
        telegram.send_message("x")


def test_cli_telegram_test_sends_and_exits(monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(telegram, "send_message", lambda text, **k: sent.append(text))
    assert cli.run(["--telegram-test"]) == 0
    assert len(sent) == 1
    assert "بنجاح" in capsys.readouterr().out


def test_emit_posts_only_real_signals(monkeypatch):
    sent = []
    monkeypatch.setattr(telegram, "send_message", lambda text, **k: sent.append(text))
    args = cli.build_arg_parser().parse_args(["--telegram"])
    cli._emit("لا توجد إشارة", is_signal=False, args=args)
    assert sent == []
    cli._emit("تقرير الإشارة", is_signal=True, args=args)
    assert len(sent) == 1 and "تقرير الإشارة" in sent[0]
