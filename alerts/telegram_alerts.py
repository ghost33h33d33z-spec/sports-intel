"""
alerts/telegram_alerts.py

Sends rich, formatted alerts to Telegram when a value bet is found.
Each message includes all the context you need to make a decision:
odds, edge, EV, form, H2H, sharp money signal, injury warnings.

Uses Telegram MarkdownV2 formatting (bold, italic, monospace, emoji).

Setup:
  1. Create a bot via @BotFather — get the token
  2. Get your chat ID via @userinfobot
  3. Set TG_TOKEN and TG_CHAT in .env
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from config import cfg
from core.http_client import HTTPClient
from models.value_detector import ValueBet

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"

# Emojis for outcomes
OUTCOME_EMOJI = {
    "home_win": "🏠",
    "away_win": "✈️",
    "draw": "🤝",
}

# Sport icons
SPORT_EMOJI = {
    "soccer": "⚽",
    "basketball": "🏀",
    "americanfootball": "🏈",
    "tennis": "🎾",
    "mma": "🥊",
    "boxing": "🥊",
}


def _sport_icon(sport: str) -> str:
    for key, icon in SPORT_EMOJI.items():
        if sport.startswith(key):
            return icon
    return "🎯"


def _escape_md(text: str) -> str:
    """
    Telegram MarkdownV2 requires escaping these special characters.
    Missing escapes = message fails to send silently.
    """
    specials = r"\_*[]()~`>#+-=|{}.!"
    for char in specials:
        text = text.replace(char, f"\\{char}")
    return text


def _confidence_bar(confidence: float, length: int = 10) -> str:
    """Visual confidence bar. 0.75 -> '███████░░░'"""
    filled = round(confidence * length)
    return "█" * filled + "░" * (length - filled)


def _format_value_bet(bet: ValueBet, scan_num: int) -> str:
    """
    Build a richly formatted Telegram message for one value bet.
    Uses MarkdownV2 formatting.
    """
    icon = _sport_icon(bet.sport)
    outcome_icon = OUTCOME_EMOJI.get(bet.outcome, "🎯")
    sport_label = bet.sport.replace("_", " ").title()
    outcome_label = bet.outcome.replace("_", " ").title()

    # Edge and EV strings
    ev_str = f"+{bet.expected_value:.1%}" if bet.expected_value > 0 else f"{bet.expected_value:.1%}"
    ev_label = "🟢 POSITIVE EV" if bet.expected_value > 0 else "🔴 NEGATIVE EV"

    # Confidence bar
    conf_bar = _confidence_bar(bet.confidence)
    conf_pct = f"{bet.confidence:.0%}"

    # Sharp / injury warnings
    warnings = []
    if bet.sharp_money_signal:
        warnings.append("⚡ Sharp money detected \\(line shortened significantly\\)")
    if bet.injury_concern:
        warnings.append("🚑 Injury concern flagged for this team")
    warnings_block = "\n".join(warnings) if warnings else "✅ No major concerns"

    # Odds comparison block
    our_odds_equiv = f"{1/bet.our_prob:.2f}" if bet.our_prob > 0 else "N/A"

    lines = [
        f"{icon} *{_escape_md(sport_label)}*",
        "",
        f"🆚 *{_escape_md(bet.home_team)} vs {_escape_md(bet.away_team)}*",
        f"⏰ Kickoff: {_escape_md(bet.kick_off_time)} \\({bet.hours_to_kickoff:.1f}h away\\)",
        "",
        f"━━━━━━━━━━━━━━━━",
        f"{outcome_icon} *BET: {_escape_md(outcome_label)}*",
        f"━━━━━━━━━━━━━━━━",
        "",
        f"📊 *Odds Analysis*",
        f"  Best available: `{bet.best_odds:.3f}`  \\({_escape_md(bet.sharpest_book)}\\)",
        f"  Market implies: `{bet.market_implied_prob:.1%}` probability",
        f"  Our estimate:   `{bet.our_prob:.1%}` probability",
        f"  Our fair odds:  `{our_odds_equiv}`",
        "",
        f"💹 *Value Metrics*",
        f"  Edge:  `{bet.edge:+.2%}`",
        f"  {ev_label}: `{ev_str}`",
        "",
        f"🎯 *Model Confidence*",
        f"  `{conf_bar}` {conf_pct}",
        "",
        f"⚠️ *Warnings*",
        warnings_block,
        "",
        f"📎 Scan \\#{scan_num} \\| {_escape_md(datetime.now(timezone.utc).strftime('%H:%M UTC'))}",
    ]

    return "\n".join(lines)


def _format_scan_summary(bets: List[ValueBet], scan_num: int,
                          events_checked: int, elapsed_sec: float) -> str:
    """
    Short summary message sent after each scan even if no value bets found.
    Keeps you informed the bot is still running.
    """
    if not bets:
        return (
            f"🔄 *Scan \\#{scan_num} Complete*\n"
            f"Checked {events_checked} events in {elapsed_sec:.0f}s\n"
            f"No value bets found this cycle\\."
        )

    sport_counts: dict = {}
    for b in bets:
        icon = _sport_icon(b.sport)
        sport_counts[icon] = sport_counts.get(icon, 0) + 1

    sport_breakdown = "  ".join(f"{icon} ×{n}" for icon, n in sport_counts.items())

    top_3 = bets[:3]
    top_lines = []
    for b in top_3:
        outcome_icon = OUTCOME_EMOJI.get(b.outcome, "🎯")
        top_lines.append(
            f"  {outcome_icon} {_escape_md(b.home_team[:12])} vs "
            f"{_escape_md(b.away_team[:12])} @ `{b.best_odds:.2f}` "
            f"EV: `{b.expected_value:+.1%}`"
        )

    return (
        f"✅ *Scan \\#{scan_num} — {len(bets)} Value Bet{'s' if len(bets) != 1 else ''} Found*\n"
        f"Events checked: {events_checked} \\| Time: {elapsed_sec:.0f}s\n"
        f"{sport_breakdown}\n\n"
        f"*Top picks:*\n" + "\n".join(top_lines)
    )


class AlertSystem:
    """
    Manages all alert delivery. Currently supports Telegram.
    Easily extensible to Discord, email, etc.
    """

    def __init__(self, http: HTTPClient):
        self.http = http
        self.token = cfg.api.telegram_token
        self.chat_id = cfg.api.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram not configured — alerts will log to console only")

    async def send_value_bet(self, bet: ValueBet, scan_num: int) -> bool:
        """Send a single value bet alert."""
        message = _format_value_bet(bet, scan_num)
        success = await self._send(message, parse_mode="MarkdownV2")
        if success:
            logger.info(f"Alert sent: {bet.summary_line()}")
        return success

    async def send_value_bets(self, bets: List[ValueBet], scan_num: int) -> None:
        """Send all value bets, with a short pause between messages."""
        for bet in bets:
            await self.send_value_bet(bet, scan_num)
            if len(bets) > 1:
                await asyncio.sleep(1.0)   # avoid hitting Telegram rate limit

    async def send_scan_summary(self, bets: List[ValueBet], scan_num: int,
                                  events_checked: int, elapsed_sec: float) -> None:
        """Send the per-scan summary message."""
        message = _format_scan_summary(bets, scan_num, events_checked, elapsed_sec)
        await self._send(message, parse_mode="MarkdownV2")

    async def send_startup(self) -> None:
        """Send a message when the bot starts up."""
        msg = (
            "🚀 *Sports Intel Bot Online*\n"
            f"Sports monitored: {len(cfg.sports.monitored)}\n"
            f"Scan interval: {cfg.betting.scan_interval}s\n"
            f"Min edge: {cfg.betting.min_edge:.0%} \\| "
            f"Min confidence: {cfg.betting.min_confidence:.0%}\n"
            f"Pre\\-game window: {cfg.betting.pregame_hours_before}h before KO"
        )
        await self._send(msg, parse_mode="MarkdownV2")

    async def send_error(self, error_msg: str) -> None:
        """Send an error notification — useful for monitoring."""
        msg = f"⚠️ *Error*\n`{_escape_md(str(error_msg)[:500])}`"
        await self._send(msg, parse_mode="MarkdownV2")

    async def _send(self, text: str, parse_mode: str = "MarkdownV2") -> bool:
        """
        Send a message via Telegram Bot API.
        Falls back to console log if Telegram is not configured.
        """
        if not self.enabled:
            logger.info(f"\n--- ALERT ---\n{text}\n---")
            return True

        url = f"{TELEGRAM_API}/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        result = await self.http.post_json(url, json=payload)
        if result and result.get("ok"):
            return True
        else:
            # If MarkdownV2 fails (e.g. bad escaping), retry as plain text
            logger.warning(f"Telegram send failed (MarkdownV2), retrying plain text")
            plain_payload = {
                "chat_id": self.chat_id,
                "text": text.replace("\\", "").replace("*", "").replace("`", ""),
                "disable_web_page_preview": True,
            }
            result2 = await self.http.post_json(url, json=plain_payload)
            return bool(result2 and result2.get("ok"))
