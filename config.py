"""
config.py — Central config for all keys, thresholds, and behaviour.

Monster.bet stack fingerprint (for HTTP mimicry):
  - Next.js 14 App Router (SSR + RSC)
  - Cloudflare CDN / Bot Management
  - Railway hosting (iad1 edge)
  - Tailwind CSS / dark theme
  - Whop payments
  - Google Analytics (GTM-P3GBZGL5, G-FRL3PVRXEK)

We mimic a Next.js SPA client making fetch requests:
  - Accept: application/json, */*
  - x-nextjs-data: 1 header on API calls
  - Referrer set to monster.bet
"""
import os
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP FINGERPRINT (Next.js / Cloudflare-safe headers)
# ─────────────────────────────────────────────────────────────────────────────
NEXTJS_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.monster.bet/",
    "Origin": "https://www.monster.bet",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "x-nextjs-data": "1",
    "DNT": "1",
}

# Browser profiles that Cloudflare recognises as legit
BROWSER_PROFILES = ["chrome120", "chrome119", "edge99", "safari17_0", "chrome110"]


# ─────────────────────────────────────────────────────────────────────────────
#  API KEYS  — set via environment or edit here directly
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    # TheOddsAPI — 500 req/mo free — https://the-odds-api.com
    the_odds_api_key: str = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY", "bda33adca828c09dc3cac3a856aef176")
    )

    # SharpAPI — multi-book odds & sharp signals — https://sharpapi.io
    sharpapi_key: str = field(
        default_factory=lambda: os.getenv("SHARPAPI_IO_KEY", "sk_live_M5NjAFNP7HNJG7JsshgxrF")
    )

    # BoltOdds — real-time odds — https://boltodds.com
    boltodds_key: str = field(
        default_factory=lambda: os.getenv("BOLTODDS_KEY", "a229e557-4fe8-4db6-a425-8d2ba392d968")
    )

    # Spro.agency WebSocket stream — https://spro.agency
    spro_key: str = field(
        default_factory=lambda: os.getenv("SPRO_KEY", "")
    )

    # ParlayAPI — parlay builder — http://parlayapi.com
    parlay_key: str = field(
        default_factory=lambda: os.getenv("PARLAY_API_KEY", "b52eb251b46aab86e06dff0853452e2d")
    )

    # Firecrawl — JS-rendered scraping — https://firecrawl.dev
    firecrawl_key: str = field(
        default_factory=lambda: os.getenv("FIRECRAWL_KEY", "fc-0b42a75279de47259e3511f60e9c8ca0")
    )

    # API-Sports — team stats, form, H2H — https://api-sports.io (optional)
    api_sports_key: str = field(
        default_factory=lambda: os.getenv("API_SPORTS_KEY", "")
    )

    # Telegram alerts (optional but recommended)
    telegram_token: str = field(
        default_factory=lambda: os.getenv("TG_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.getenv("TG_CHAT", "")
    )


# ─────────────────────────────────────────────────────────────────────────────
#  API FRUGALITY — controls how aggressively we use paid quota
#  TheOddsAPI free = 500 req/mo. Each scan = 1 req per sport.
#  With 4 sports and manual-only scanning, you'll stay well within limits.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuotaConfig:
    # TheOddsAPI: max API calls per 24h (safety ceiling regardless of mode)
    theodds_daily_max: int = 20

    # SharpAPI: max calls per scan
    sharpapi_calls_per_scan: int = 3

    # Firecrawl: only use for JS-heavy pages (costs credits)
    firecrawl_enabled: bool = True

    # ParlayAPI: only call when a multi-leg parlay is being evaluated
    parlay_enabled: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN MODE
#  "manual"    — only scan when you press Enter / send Telegram /scan command
#  "alert"     — run a lightweight background check; only alert if something
#                substantial is found (big line move, arbitrage, etc.)
#  "scheduled" — scan on a fixed interval (legacy behaviour)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScanConfig:
    mode: str = "manual"          # "manual" | "alert" | "scheduled"
    scheduled_interval: int = 1800  # seconds — only used in "scheduled" mode

    # In "alert" mode: minimum edge % that triggers an unsolicited alert
    alert_mode_min_edge: float = 0.08     # 8% — only really exceptional finds
    alert_mode_sharp_required: bool = True  # must also have sharp money signal

    # In "alert" mode: background check interval (much lighter than full scan)
    alert_check_interval: int = 300       # 5 min lightweight check


# ─────────────────────────────────────────────────────────────────────────────
#  SPORTS — frugal default: 4 high-liquidity sports only
#  Add more when you have a paid TheOddsAPI plan
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SportsConfig:
    monitored: List[str] = field(default_factory=lambda: [
        "americanfootball_nfl",
        "basketball_nba",
        "soccer_epl",
        "mma_mixed_martial_arts",
    ])
    markets: List[str] = field(default_factory=lambda: ["h2h", "spreads"])
    bookmakers: List[str] = field(default_factory=lambda: [
        "draftkings", "fanduel", "betmgm", "pinnacle", "bet365",
    ])

    # SharpAPI sportsbooks (comma-separated string for API call)
    sharpapi_books: str = "draftkings,fanduel,betmgm,pinnacle"


# ─────────────────────────────────────────────────────────────────────────────
#  VALUE BET THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BettingConfig:
    min_confidence: float = 0.62
    min_edge: float = 0.04            # 4% edge to flag normally
    substantial_edge: float = 0.08   # 8% = "substantial" for alert mode
    min_odds: float = 1.30
    max_odds: float = 8.00
    pregame_hours_before: int = 24
    min_books_for_consensus: int = 3


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_path: str = "models/value_model.joblib"
    scaler_path: str = "models/scaler.joblib"
    min_samples_to_train: int = 100
    retrain_every_n_bets: int = 50


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LogConfig:
    level: str = "INFO"
    log_file: str = "logs/intel.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    api = APIConfig()
    quota = QuotaConfig()
    scan = ScanConfig()
    sports = SportsConfig()
    betting = BettingConfig()
    model = ModelConfig()
    log = LogConfig()


cfg = Config()
