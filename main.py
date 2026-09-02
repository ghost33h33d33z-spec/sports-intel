#!/usr/bin/env python3
"""
main.py — Sports Betting Intel Bot (On-Demand Edition)

THREE MODES — set in config.py → ScanConfig.mode:

  "manual"    — Bot idles. Press ENTER in terminal or send /scan on Telegram
                to trigger a full scan. Nothing runs until you ask.

  "alert"     — Lightweight background check every 5 min. Only fires an alert
                if something truly substantial is found (≥8% edge + sharp signal
                or arbitrage opportunity). You won't hear from it unless it matters.

  "scheduled" — Original timed loop (every 30min by default). Not recommended
                if you're frugal about API quota.

TELEGRAM COMMANDS (when bot is running):
  /scan           — run a full scan right now
  /status         — show quota, last scan time, mode
  /top            — show top 3 bets from last scan (no new API call)
  /quota          — show API usage for today
  /help           — command list

Run:
    source .env && python main.py

Background:
    nohup python main.py > logs/output.log 2>&1 &
    tail -f logs/output.log
    pkill -f "python main.py"
"""

import asyncio
import logging
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from alerts.telegram_alerts import AlertSystem
from analyzers.pregame_analyzer import MatchAnalysis, PreGameAnalyzer
from config import cfg
from core.http_client import HTTPClient
from models.value_detector import ValueBet, ValueDetector
from scrapers.live_stream import LiveOddsStream
from scrapers.odds_aggregator import Event, OddsAggregator
from scrapers.web_scraper import WebScraper


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    log = logging.getLogger()
    log.setLevel(getattr(logging, cfg.log.level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(
        cfg.log.log_file, maxBytes=cfg.log.max_bytes,
        backupCount=cfg.log.backup_count, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return logging.getLogger("main")


logger = setup_logging()


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class ScanPipeline:
    def __init__(self):
        self.http       = HTTPClient(retries=3, backoff_base=2.0, timeout=20, rate_gap=1.5)
        self.aggregator = OddsAggregator(self.http)
        self.scraper    = WebScraper(self.http)
        self.analyzer   = PreGameAnalyzer(self.http)
        self.detector   = ValueDetector()
        self.alerts     = AlertSystem(self.http)
        self.stream     = LiveOddsStream()

        self.scan_count: int = 0
        self.last_scan_time: Optional[datetime] = None
        self.last_value_bets: List[ValueBet] = []
        self.all_analyses: List[MatchAnalysis] = []

    async def run_scan(self, use_firecrawl: bool = False) -> List[ValueBet]:
        """Full pipeline scan. use_firecrawl=True on manual scans only."""
        self.scan_count += 1
        start = time.monotonic()
        logger.info(f"═══ Scan #{self.scan_count} starting ═══")

        # 1. Odds — prefer stream cache, fall back to REST APIs
        stream_events = self.stream.get_latest()
        if stream_events:
            logger.info(f"  Using {len(stream_events)} events from live stream cache")
            events: List[Event] = stream_events
        else:
            logger.info("  Fetching from APIs...")
            events = await self.aggregator.fetch_all()

        if not events:
            logger.warning("No events returned from any source")
            elapsed = time.monotonic() - start
            await self.alerts.send_scan_summary([], self.scan_count, 0, elapsed)
            return []

        # 2. Supplement with scraping (free static, Firecrawl only if manual)
        logger.info(f"  Scraping supplementary data (firecrawl={use_firecrawl})...")
        try:
            scraped = await self.scraper.scrape_all(use_firecrawl=use_firecrawl)
            injury_news = [s.content for s in scraped if s.data_type == "news_headline"]
        except Exception as e:
            logger.warning(f"Scraping error (non-fatal): {e}")
            injury_news = []

        # 3. Sharp move signals from live stream
        sharp_moves = self.stream.get_sharp_moves(since_minutes=30)
        if sharp_moves:
            logger.info(f"  {len(sharp_moves)} sharp moves in last 30min")

        # 4. Pre-game analysis
        logger.info(f"  Analysing {len(events)} events...")
        analyses = await self.analyzer.analyse_events(events, injury_news)
        self.all_analyses.extend(analyses)

        # 5. AI value detection
        value_bets = self.detector.find_value_bets(analyses)
        self.last_value_bets = value_bets
        self.last_scan_time = datetime.now(timezone.utc)

        elapsed = time.monotonic() - start
        logger.info(
            f"Scan #{self.scan_count}: {len(events)} events → "
            f"{len(analyses)} analysed → {len(value_bets)} value bets | {elapsed:.1f}s"
        )

        # 6. Alerts — send all bets + summary
        if value_bets:
            await self.alerts.send_value_bets(value_bets, self.scan_count)
        await self.alerts.send_scan_summary(value_bets, self.scan_count, len(events), elapsed)

        if value_bets:
            logger.info("Top picks:")
            for i, b in enumerate(value_bets[:5], 1):
                logger.info(f"  {i}. {b.summary_line()}")

        return value_bets

    async def run_alert_check(self) -> None:
        """
        Lightweight background check.
        Only fetches odds from stream cache (no API calls).
        Only sends alert if edge ≥ substantial threshold AND sharp signal present.
        """
        stream_events = self.stream.get_latest()
        if not stream_events:
            logger.debug("Alert check: no stream data available")
            return

        analyses = await self.analyzer.analyse_events(stream_events, [])
        bets = self.detector.find_value_bets(analyses)

        substantial = [
            b for b in bets
            if b.edge >= cfg.betting.substantial_edge and
            (b.sharp_money_signal or not cfg.scan.alert_mode_sharp_required)
        ]

        if substantial:
            logger.info(f"SUBSTANTIAL BET FOUND: {substantial[0].summary_line()}")
            for bet in substantial:
                await self.alerts.send_value_bet(bet, self.scan_count)
        else:
            logger.debug(f"Alert check: {len(bets)} bets, none substantial")

    def status_text(self) -> str:
        last = self.last_scan_time.strftime("%H:%M UTC") if self.last_scan_time else "never"
        quota = self.aggregator.quota.status()
        return (
            f"Mode: {cfg.scan.mode}\n"
            f"Last scan: {last}\n"
            f"Scans run: {self.scan_count}\n"
            f"Last scan bets: {len(self.last_value_bets)}\n"
            f"{quota}"
        )


# ─── Telegram command listener ────────────────────────────────────────────────

class TelegramCommandListener:
    """
    Polls Telegram for commands sent to the bot.
    Handles: /scan, /status, /top, /quota, /help

    This uses long-polling (getUpdates) — no webhook server needed.
    """

    def __init__(self, pipeline: ScanPipeline):
        self.pipeline = pipeline
        self.http = pipeline.http
        self.token = cfg.api.telegram_token
        self.chat_id = cfg.api.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        self._last_update_id = 0

    async def poll_forever(self) -> None:
        """Long-poll Telegram for incoming messages."""
        if not self.enabled:
            logger.info("Telegram not configured — command listener disabled")
            return

        logger.info("Telegram command listener started")
        while True:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
            except Exception as e:
                logger.error(f"Telegram poll error: {e}")
            await asyncio.sleep(2)

    async def _get_updates(self) -> list:
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        data = await self.http.get_json(url, params={
            "offset": self._last_update_id + 1,
            "timeout": 30,
            "allowed_updates": ["message"],
        })
        if not data or not data.get("ok"):
            return []
        updates = data.get("result", [])
        if updates:
            self._last_update_id = updates[-1]["update_id"]
        return updates

    async def _handle_update(self, update: dict) -> None:
        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = msg.get("text", "").strip().lower()

        # Only respond to the configured chat
        if chat_id != self.chat_id:
            return

        logger.info(f"Telegram command: {text}")

        if text.startswith("/scan"):
            await self.pipeline.alerts._send("🔄 Starting scan...")
            bets = await self.pipeline.run_scan(use_firecrawl=True)
            if not bets:
                await self.pipeline.alerts._send("✅ Scan complete. No value bets found.")

        elif text.startswith("/status"):
            await self.pipeline.alerts._send(
                f"📊 Status\n\n{self.pipeline.status_text()}")

        elif text.startswith("/top"):
            if not self.pipeline.last_value_bets:
                await self.pipeline.alerts._send("No bets from last scan yet. Run /scan first.")
            else:
                top = self.pipeline.last_value_bets[:3]
                lines = [f"{i+1}. {b.summary_line()}" for i, b in enumerate(top)]
                await self.pipeline.alerts._send("🏆 Top picks (last scan):\n\n" + "\n".join(lines))

        elif text.startswith("/quota"):
            await self.pipeline.alerts._send(
                f"📈 API Quota\n\n{self.pipeline.aggregator.quota.status()}")

        elif text.startswith("/help"):
            await self.pipeline.alerts._send(
                "📖 Commands:\n"
                "/scan    — run a full scan now\n"
                "/status  — mode, last scan, quota\n"
                "/top     — top 3 bets from last scan\n"
                "/quota   — API usage today\n"
                "/help    — this message"
            )


# ─── Terminal scan trigger (press Enter) ──────────────────────────────────────

async def terminal_listener(pipeline: ScanPipeline) -> None:
    """
    In manual mode: wait for Enter key in terminal to trigger a scan.
    Non-blocking — runs as a separate asyncio task.
    """
    loop = asyncio.get_event_loop()
    logger.info("Press ENTER at any time to trigger a scan")
    while True:
        try:
            await loop.run_in_executor(None, input)
            logger.info("Manual trigger — starting scan...")
            await pipeline.run_scan(use_firecrawl=True)
        except Exception:
            await asyncio.sleep(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("━" * 55)
    logger.info("  Sports Betting Intel Bot")
    logger.info(f"  Mode: {cfg.scan.mode}")
    logger.info(f"  Sports: {', '.join(cfg.sports.monitored)}")
    logger.info(f"  Min edge: {cfg.betting.min_edge:.0%}  "
                f"Substantial: {cfg.betting.substantial_edge:.0%}")
    logger.info(f"  TheOddsAPI daily quota: {cfg.quota.theodds_daily_max} calls")
    logger.info("━" * 55)

    if not cfg.api.the_odds_api_key and not cfg.api.sharpapi_key and not cfg.api.boltodds_key:
        logger.error("No API keys configured. Set at least one of: THE_ODDS_API_KEY, SHARPAPI_IO_KEY, BOLTODDS_KEY")
        return

    pipeline = ScanPipeline()
    tg = TelegramCommandListener(pipeline)

    await pipeline.alerts.send_startup()

    # Start live stream in background (if SPRO_KEY set)
    tasks = []
    if cfg.api.spro_key:
        logger.info("Starting live WebSocket stream (Spro.agency)...")
        tasks.append(asyncio.create_task(pipeline.stream.run()))

    # Telegram command listener (always on if configured)
    if tg.enabled:
        tasks.append(asyncio.create_task(tg.poll_forever()))

    # ── Mode-specific behaviour ───────────────────────────────────────────────

    if cfg.scan.mode == "manual":
        logger.info("MANUAL MODE — idle. Press Enter or send /scan on Telegram.")
        tasks.append(asyncio.create_task(terminal_listener(pipeline)))
        # Just keep running (Telegram listener + stream handle everything)
        await asyncio.gather(*tasks)

    elif cfg.scan.mode == "alert":
        logger.info(f"ALERT MODE — background check every {cfg.scan.alert_check_interval}s. "
                    f"Alerts only for ≥{cfg.betting.substantial_edge:.0%} edge.")
        # Run one initial scan
        await pipeline.run_scan()
        for task in tasks:
            asyncio.create_task(task)

        consecutive_errors = 0
        while True:
            try:
                await asyncio.sleep(cfg.scan.alert_check_interval)
                await pipeline.run_alert_check()
                consecutive_errors = 0
            except KeyboardInterrupt:
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Alert check error #{consecutive_errors}: {e}")
                if consecutive_errors >= 5:
                    await pipeline.alerts.send_error(f"5 consecutive check failures: {e}")

    elif cfg.scan.mode == "scheduled":
        logger.info(f"SCHEDULED MODE — scanning every {cfg.scan.scheduled_interval}s")
        for task in tasks:
            asyncio.create_task(task)

        consecutive_errors = 0
        while True:
            try:
                await pipeline.run_scan()
                await pipeline.detector.maybe_retrain(pipeline.all_analyses)
                consecutive_errors = 0
                logger.info(f"💤 Next scan in {cfg.scan.scheduled_interval}s...")
                await asyncio.sleep(cfg.scan.scheduled_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Scan error #{consecutive_errors}: {e}", exc_info=True)
                if consecutive_errors == 5:
                    await pipeline.alerts.send_error(f"5 failures: {e}")
                backoff = min(cfg.scan.scheduled_interval * 2 ** min(consecutive_errors, 3), 1800)
                await asyncio.sleep(backoff)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Stopped")
