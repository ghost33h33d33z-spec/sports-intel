"""
scrapers/odds_aggregator.py

Multi-source odds aggregator. Priority order (most frugal first):

  1. SharpAPI    — multi-book, sharp signals, low cost per call
  2. BoltOdds    — real-time REST fallback
  3. TheOddsAPI  — used sparingly (quota guard: max 20 calls/day)

All three sources normalise into the same Event dataclass so the
rest of the system doesn't care where data came from.

Quota tracker writes to data/quota.json so the daily count
persists across restarts.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import cfg
from core.http_client import HTTPClient

logger = logging.getLogger(__name__)

THEODDS_BASE  = "https://api.the-odds-api.com/v4"
SHARPAPI_BASE = "https://api.sharpapi.io/api/v1"
BOLTODDS_BASE = "https://api.boltodds.com/v1"   # adjust if different


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class OutcomeOdds:
    best_odds: float = 0.0
    avg_odds: float = 0.0
    num_books: int = 0
    books: Dict[str, float] = field(default_factory=dict)


@dataclass
class Event:
    event_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: datetime
    markets: Dict[str, Dict[str, OutcomeOdds]] = field(default_factory=dict)
    implied_probs: Dict[str, float] = field(default_factory=dict)
    overround: float = 0.0
    sharpest_book: str = ""
    hours_to_kickoff: float = 0.0
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "sport": self.sport,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "commence_time": self.commence_time.isoformat(),
            "markets": {
                mkt: {
                    label: {
                        "best_odds": o.best_odds,
                        "avg_odds": o.avg_odds,
                        "num_books": o.num_books,
                        "books": o.books,
                    }
                    for label, o in outcomes.items()
                }
                for mkt, outcomes in self.markets.items()
            },
            "implied_probs": self.implied_probs,
            "overround": self.overround,
            "sharpest_book": self.sharpest_book,
            "hours_to_kickoff": self.hours_to_kickoff,
            "source": self.source,
        }


# ─── Quota tracker ────────────────────────────────────────────────────────────

class QuotaTracker:
    """
    Tracks daily API call counts per service.
    Persists to data/quota.json so counts survive restarts.
    """

    def __init__(self):
        self._path = Path("data/quota.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                d = json.loads(self._path.read_text())
                # Reset if it's a new day
                if d.get("date") != str(date.today()):
                    return {"date": str(date.today()), "theodds": 0, "sharpapi": 0, "boltodds": 0}
                return d
            except Exception:
                pass
        return {"date": str(date.today()), "theodds": 0, "sharpapi": 0, "boltodds": 0}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2))

    def increment(self, service: str, count: int = 1) -> None:
        self._data[service] = self._data.get(service, 0) + count
        self._save()

    def get(self, service: str) -> int:
        return self._data.get(service, 0)

    def can_use_theodds(self) -> bool:
        return self.get("theodds") < cfg.quota.theodds_daily_max

    def status(self) -> str:
        return (
            f"Quota today — TheOddsAPI: {self.get('theodds')}/{cfg.quota.theodds_daily_max} | "
            f"SharpAPI: {self.get('sharpapi')} | BoltOdds: {self.get('boltodds')}"
        )


# ─── Aggregator ───────────────────────────────────────────────────────────────

class OddsAggregator:
    """
    Tries data sources in cost order and returns a unified list of Events.

    Strategy per scan:
      - SharpAPI first (cheapest for multi-book data)
      - BoltOdds as supplement/fallback
      - TheOddsAPI only if SharpAPI returned nothing AND quota allows
    """

    def __init__(self, http: HTTPClient):
        self.http = http
        self.quota = QuotaTracker()

    async def fetch_all(self) -> List[Event]:
        logger.info(self.quota.status())
        events: List[Event] = []

        # ── 1. SharpAPI (primary — multi-book, pre-priced) ────────────────────
        if cfg.api.sharpapi_key:
            sharp_events = await self._fetch_sharpapi()
            events.extend(sharp_events)
            logger.info(f"SharpAPI: {len(sharp_events)} events")

        # ── 2. BoltOdds (supplement) ──────────────────────────────────────────
        if cfg.api.boltodds_key and len(events) < 5:
            bolt_events = await self._fetch_boltodds()
            # Deduplicate by team names
            existing = {(e.home_team, e.away_team) for e in events}
            new_bolt = [e for e in bolt_events if (e.home_team, e.away_team) not in existing]
            events.extend(new_bolt)
            logger.info(f"BoltOdds: added {len(new_bolt)} new events")

        # ── 3. TheOddsAPI (fallback, quota-guarded) ───────────────────────────
        if not events and self.quota.can_use_theodds() and cfg.api.the_odds_api_key:
            logger.info("Falling back to TheOddsAPI...")
            for sport in cfg.sports.monitored:
                sport_events = await self._fetch_theodds_sport(sport)
                events.extend(sport_events)
                self.quota.increment("theodds")
            logger.info(f"TheOddsAPI: {len(events)} events (quota now {self.quota.get('theodds')})")
        elif not events and not self.quota.can_use_theodds():
            logger.warning("TheOddsAPI daily quota reached — skipping")

        logger.info(f"Total events after dedup: {len(events)}")
        return events

    # ── SharpAPI ──────────────────────────────────────────────────────────────

    async def _fetch_sharpapi(self) -> List[Event]:
        """
        Fetch from SharpAPI. Tries each configured sport/book combo.
        One call covers multiple sportsbooks simultaneously.
        """
        events: List[Event] = []
        headers = {"X-API-Key": cfg.api.sharpapi_key}

        # Fetch by league using sport keys we know work
        league_map = {
            "americanfootball_nfl": "nfl",
            "basketball_nba": "nba",
            "soccer_epl": "epl",
            "mma_mixed_martial_arts": "ufc",
        }

        for sport_key, league in league_map.items():
            if sport_key not in cfg.sports.monitored:
                continue
            params = {
                "league": league,
                "sportsbook": cfg.sports.sharpapi_books,
                "live": "false",
            }
            data = await self.http.get_json(
                f"{SHARPAPI_BASE}/odds",
                params=params,
                headers=headers,
            )
            self.quota.increment("sharpapi")

            if not data:
                continue

            # SharpAPI returns either a list or {"data": [...]}
            rows = data if isinstance(data, list) else data.get("data", [])
            for row in rows:
                event = self._parse_sharpapi_row(row, sport_key)
                if event:
                    events.append(event)

        return events

    def _parse_sharpapi_row(self, row: dict, sport_key: str) -> Optional[Event]:
        """Parse one SharpAPI odds row into an Event."""
        try:
            # SharpAPI field names (adjust if their schema differs)
            home = row.get("home_team") or row.get("home") or ""
            away = row.get("away_team") or row.get("away") or ""
            if not home or not away:
                return None

            commence_raw = row.get("commence_time") or row.get("start_time") or ""
            if not commence_raw:
                return None
            commence_time = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_to_kickoff = (commence_time - now).total_seconds() / 3600
            if hours_to_kickoff < 0 or hours_to_kickoff > cfg.betting.pregame_hours_before:
                return None

            event = Event(
                event_id=row.get("id") or f"{home}-{away}",
                sport=sport_key,
                home_team=home,
                away_team=away,
                commence_time=commence_time,
                hours_to_kickoff=round(hours_to_kickoff, 2),
                source="sharpapi",
            )

            # Parse odds — SharpAPI nests by sportsbook then outcome
            bookmakers = row.get("bookmakers") or row.get("sportsbooks") or []
            for book in bookmakers:
                book_name = book.get("key") or book.get("name") or "unknown"
                for market in book.get("markets", []):
                    mkt_key = market.get("key", "h2h")
                    if mkt_key not in event.markets:
                        event.markets[mkt_key] = {}
                    for outcome in market.get("outcomes", []):
                        label = self._label(outcome.get("name", ""), event)
                        price = float(outcome.get("price", 0))
                        if not (cfg.betting.min_odds < price < cfg.betting.max_odds):
                            continue
                        if label not in event.markets[mkt_key]:
                            event.markets[mkt_key][label] = OutcomeOdds()
                        entry = event.markets[mkt_key][label]
                        entry.books[book_name] = price
                        entry.num_books += 1

            self._finalise_event(event)
            if not self._has_coverage(event):
                return None
            return event

        except Exception as e:
            logger.debug(f"SharpAPI parse error: {e}")
            return None

    # ── BoltOdds ──────────────────────────────────────────────────────────────

    async def _fetch_boltodds(self) -> List[Event]:
        """Fetch from BoltOdds REST API."""
        headers = {"Authorization": f"Bearer {cfg.api.boltodds_key}"}
        events: List[Event] = []

        data = await self.http.get_json(
            f"{BOLTODDS_BASE}/odds",
            params={"live": "false", "limit": 50},
            headers=headers,
        )
        self.quota.increment("boltodds")

        if not data:
            return events

        rows = data if isinstance(data, list) else data.get("data", data.get("odds", []))
        for row in rows:
            sport_key = self._guess_sport(row.get("sport") or row.get("league") or "")
            if sport_key not in cfg.sports.monitored:
                continue
            event = self._parse_generic_row(row, sport_key, source="boltodds")
            if event:
                events.append(event)

        return events

    # ── TheOddsAPI (frugal fallback) ──────────────────────────────────────────

    async def _fetch_theodds_sport(self, sport_key: str) -> List[Event]:
        """One API call = one sport. Uses quota."""
        url = f"{THEODDS_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": cfg.api.the_odds_api_key,
            "regions": "us,uk",
            "markets": "h2h,spreads",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        raw = await self.http.get_json(url, params=params)
        if not raw or not isinstance(raw, list):
            return []

        events = []
        for item in raw:
            event = self._parse_theodds_item(item, sport_key)
            if event:
                events.append(event)
        return events

    def _parse_theodds_item(self, raw: dict, sport_key: str) -> Optional[Event]:
        try:
            commence_time = datetime.fromisoformat(
                raw["commence_time"].replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            hours_to_kickoff = (commence_time - now).total_seconds() / 3600
            if hours_to_kickoff < 0 or hours_to_kickoff > cfg.betting.pregame_hours_before:
                return None

            event = Event(
                event_id=raw.get("id", ""),
                sport=sport_key,
                home_team=raw.get("home_team", ""),
                away_team=raw.get("away_team", ""),
                commence_time=commence_time,
                hours_to_kickoff=round(hours_to_kickoff, 2),
                source="theodds_api",
            )

            for bookmaker in raw.get("bookmakers", []):
                book_name = bookmaker.get("key", "unknown")
                for market in bookmaker.get("markets", []):
                    mkt_key = market.get("key", "h2h")
                    if mkt_key not in event.markets:
                        event.markets[mkt_key] = {}
                    for outcome in market.get("outcomes", []):
                        label = self._label(outcome.get("name", ""), event)
                        price = float(outcome.get("price", 0))
                        if not (cfg.betting.min_odds < price < cfg.betting.max_odds):
                            continue
                        if label not in event.markets[mkt_key]:
                            event.markets[mkt_key][label] = OutcomeOdds()
                        entry = event.markets[mkt_key][label]
                        entry.books[book_name] = price
                        entry.num_books += 1

            self._finalise_event(event)
            if not self._has_coverage(event):
                return None
            return event
        except Exception as e:
            logger.debug(f"TheOddsAPI parse error: {e}")
            return None

    # ── Generic parser (works for unknown API schemas) ─────────────────────────

    def _parse_generic_row(self, row: dict, sport_key: str, source: str) -> Optional[Event]:
        try:
            home = row.get("home_team") or row.get("home") or ""
            away = row.get("away_team") or row.get("away") or ""
            if not home or not away:
                return None
            commence_raw = row.get("commence_time") or row.get("start_time") or ""
            if not commence_raw:
                return None
            commence_time = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours = (commence_time - now).total_seconds() / 3600
            if hours < 0 or hours > cfg.betting.pregame_hours_before:
                return None
            event = Event(
                event_id=row.get("id") or f"{home}-{away}",
                sport=sport_key,
                home_team=home,
                away_team=away,
                commence_time=commence_time,
                hours_to_kickoff=round(hours, 2),
                source=source,
            )
            self._finalise_event(event)
            return event if self._has_coverage(event) else None
        except Exception:
            return None

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _label(self, name: str, event: Event) -> str:
        if name == event.home_team:
            return "home_win"
        if name == event.away_team:
            return "away_win"
        if name.lower() in ("draw", "tie"):
            return "draw"
        return name.lower().replace(" ", "_")

    def _finalise_event(self, event: Event) -> None:
        """Compute best/avg odds, implied probs, overround, sharpest book."""
        for mkt, outcomes in event.markets.items():
            for label, od in outcomes.items():
                if od.books:
                    vals = list(od.books.values())
                    od.best_odds = max(vals)
                    od.avg_odds = round(sum(vals) / len(vals), 4)

        h2h = event.markets.get("h2h", {})
        probs: Dict[str, float] = {}
        book_totals: Dict[str, float] = {}

        for label, od in h2h.items():
            if od.best_odds > 1.0:
                probs[label] = round(1 / od.best_odds, 4)
            for book, price in od.books.items():
                book_totals[book] = book_totals.get(book, 0.0) + (1 / price if price > 0 else 0)

        event.implied_probs = probs
        event.overround = round(sum(probs.values()), 4)
        if book_totals:
            event.sharpest_book = min(book_totals, key=book_totals.get)

    def _has_coverage(self, event: Event) -> bool:
        h2h = event.markets.get("h2h", {})
        if not h2h:
            return False
        return max((o.num_books for o in h2h.values()), default=0) >= cfg.betting.min_books_for_consensus

    def _guess_sport(self, raw: str) -> str:
        raw = raw.lower()
        if "nfl" in raw or "football" in raw:
            return "americanfootball_nfl"
        if "nba" in raw or "basketball" in raw:
            return "basketball_nba"
        if "epl" in raw or "premier" in raw or "soccer" in raw:
            return "soccer_epl"
        if "ufc" in raw or "mma" in raw:
            return "mma_mixed_martial_arts"
        return raw
