"""
scrapers/live_stream.py

Spro.agency WebSocket client for real-time odds streaming.
Runs as a background task — updates a shared in-memory store that the
main pipeline reads from instead of making REST calls every scan.

This is the most frugal approach:
  - One persistent WebSocket connection = zero per-scan API calls
  - The stream pushes updates when lines move, not on a poll cycle
  - Background task updates self.latest_events automatically

Usage (in main.py):
    stream = LiveOddsStream()
    asyncio.create_task(stream.run())
    # Later:
    events = stream.get_latest()
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import cfg
from scrapers.odds_aggregator import Event, OutcomeOdds

logger = logging.getLogger(__name__)

SPRO_URI = "wss://spro.agency/api"
SPRO_REST = "https://spro.agency/api/get_games"


@dataclass
class StreamedOddsUpdate:
    """A single odds update pushed by the stream."""
    event_id: str
    home_team: str
    away_team: str
    sport: str
    outcome: str
    bookmaker: str
    old_odds: float
    new_odds: float
    movement_pct: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_sharp_move(self) -> bool:
        """Sharp move = significant shortening (odds dropped >4%)."""
        return self.movement_pct < -4.0

    @property
    def is_steam_move(self) -> bool:
        """Steam move = rapid shortening across multiple books (>6%)."""
        return self.movement_pct < -6.0


class LiveOddsStream:
    """
    Persistent WebSocket connection to Spro.agency.
    Runs in the background and updates self.latest_events as lines move.

    Falls back gracefully if no SPRO_KEY is set.
    """

    def __init__(self):
        self.key = cfg.api.spro_key
        self.enabled = bool(self.key)
        self.latest_events: Dict[str, Event] = {}     # event_id -> Event
        self.recent_moves: List[StreamedOddsUpdate] = []  # last 50 moves
        self._running = False
        self._prev_odds: Dict[str, Dict[str, float]] = {}  # event_id -> outcome -> odds

    async def run(self) -> None:
        """
        Main loop. Reconnects automatically on disconnect.
        Run as: asyncio.create_task(stream.run())
        """
        if not self.enabled:
            logger.info("Spro.agency stream disabled (no SPRO_KEY set)")
            return

        self._running = True
        logger.info("Starting Spro.agency live stream...")

        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"Stream error: {e}")
            logger.info("Stream disconnected — reconnecting in 5s...")
            await asyncio.sleep(5)

    def stop(self) -> None:
        self._running = False

    def get_latest(self) -> List[Event]:
        """Return current snapshot of all tracked events."""
        return list(self.latest_events.values())

    def get_sharp_moves(self, since_minutes: int = 30) -> List[StreamedOddsUpdate]:
        """Return sharp/steam moves in the last N minutes."""
        cutoff = datetime.now(timezone.utc).timestamp() - since_minutes * 60
        return [
            m for m in self.recent_moves
            if m.timestamp.timestamp() > cutoff and m.is_sharp_move
        ]

    async def fetch_games_rest(self) -> List[dict]:
        """
        Alternative to WebSocket: one-shot REST call to get current game list.
        Used during manual scans when the stream isn't running.
        """
        if not self.enabled:
            return []
        try:
            from core.http_client import HTTPClient
            http = HTTPClient()
            data = await http.get_json(
                SPRO_REST,
                params={"key": self.key},
            )
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Spro REST fetch failed: {e}")
            return []

    # ── Private ───────────────────────────────────────────────────────────────

    async def _connect_and_listen(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.error("pip install websockets to use live stream")
            self._running = False
            return

        uri = f"{SPRO_URI}?key={self.key}"

        async with websockets.connect(uri, max_size=None) as ws:
            # First message is an acknowledgement
            ack = await ws.recv()
            logger.info(f"Spro stream connected: {ack}")

            # Subscribe to our sports and key sportsbooks
            subscribe = {
                "action": "subscribe",
                "filters": {
                    "sports": ["NFL", "NBA", "Soccer", "MMA"],
                    "sportsbooks": ["draftkings", "fanduel", "betmgm", "pinnacle"],
                    "markets": ["Moneyline", "Spread"],
                },
            }
            await ws.send(json.dumps(subscribe))
            logger.info("Subscribed to Spro stream")

            # Listen indefinitely
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    msg = json.loads(raw)
                    self._process_message(msg)
                except asyncio.TimeoutError:
                    # Send a ping to keep connection alive
                    await ws.send(json.dumps({"action": "ping"}))
                except Exception as e:
                    logger.debug(f"Stream message error: {e}")
                    break

    def _process_message(self, msg: dict) -> None:
        """
        Process one incoming WebSocket message.
        Updates self.latest_events and self.recent_moves.
        """
        msg_type = msg.get("type") or msg.get("action") or ""

        if msg_type in ("odds_update", "line_move", "update"):
            self._handle_odds_update(msg)
        elif msg_type in ("snapshot", "games"):
            self._handle_snapshot(msg)
        elif msg_type == "ping":
            pass  # heartbeat
        else:
            logger.debug(f"Unknown stream message type: {msg_type}")

    def _handle_odds_update(self, msg: dict) -> None:
        """Handle a single odds movement update."""
        event_id = msg.get("event_id") or msg.get("game_id") or ""
        if not event_id:
            return

        outcome = msg.get("outcome") or msg.get("market") or "moneyline"
        bookmaker = msg.get("sportsbook") or msg.get("book") or "unknown"
        new_odds = float(msg.get("odds") or msg.get("price") or 0)
        if new_odds <= 0:
            return

        # Detect movement
        prev_key = f"{event_id}:{outcome}:{bookmaker}"
        old_odds = self._prev_odds.get(prev_key, new_odds)
        movement_pct = round((new_odds - old_odds) / old_odds * 100, 2) if old_odds else 0.0
        self._prev_odds[prev_key] = new_odds

        move = StreamedOddsUpdate(
            event_id=event_id,
            home_team=msg.get("home_team", ""),
            away_team=msg.get("away_team", ""),
            sport=msg.get("sport", ""),
            outcome=outcome,
            bookmaker=bookmaker,
            old_odds=old_odds,
            new_odds=new_odds,
            movement_pct=movement_pct,
        )
        self.recent_moves.append(move)
        # Keep only last 200 moves
        if len(self.recent_moves) > 200:
            self.recent_moves = self.recent_moves[-200:]

        if move.is_sharp_move:
            logger.info(
                f"⚡ Sharp move: {move.home_team} vs {move.away_team} | "
                f"{outcome} @ {new_odds:.2f} ({movement_pct:+.1f}%) [{bookmaker}]"
            )

    def _handle_snapshot(self, msg: dict) -> None:
        """Handle a full snapshot of current games."""
        games = msg.get("data") or msg.get("games") or []
        count = 0
        for game in games:
            event_id = game.get("id") or game.get("event_id") or ""
            if not event_id:
                continue
            # We store as a minimal Event for compatibility with the pipeline
            sport = self._map_sport(game.get("sport", ""))
            if sport not in cfg.sports.monitored:
                continue

            try:
                commence_raw = game.get("start_time") or game.get("commence_time") or ""
                commence_time = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
                hours = (commence_time - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours < 0 or hours > cfg.betting.pregame_hours_before:
                    continue

                event = Event(
                    event_id=event_id,
                    sport=sport,
                    home_team=game.get("home_team", ""),
                    away_team=game.get("away_team", ""),
                    commence_time=commence_time,
                    hours_to_kickoff=round(hours, 2),
                    source="spro_stream",
                )

                # Parse odds if present in snapshot
                for book_data in game.get("odds", []):
                    book = book_data.get("sportsbook", "unknown")
                    ml = book_data.get("moneyline", {})
                    if "home" in ml and "away" in ml:
                        if "h2h" not in event.markets:
                            event.markets["h2h"] = {}
                        for label, price in [("home_win", ml["home"]), ("away_win", ml["away"])]:
                            price = float(price)
                            if cfg.betting.min_odds < price < cfg.betting.max_odds:
                                if label not in event.markets["h2h"]:
                                    event.markets["h2h"][label] = OutcomeOdds()
                                event.markets["h2h"][label].books[book] = price
                                event.markets["h2h"][label].num_books += 1

                self.latest_events[event_id] = event
                count += 1
            except Exception as e:
                logger.debug(f"Snapshot parse error: {e}")

        logger.info(f"Stream snapshot: {count} events loaded")

    def _map_sport(self, raw: str) -> str:
        raw = raw.lower()
        if "nfl" in raw:
            return "americanfootball_nfl"
        if "nba" in raw:
            return "basketball_nba"
        if "soccer" in raw or "epl" in raw:
            return "soccer_epl"
        if "mma" in raw or "ufc" in raw:
            return "mma_mixed_martial_arts"
        return raw
