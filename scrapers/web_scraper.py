"""
scrapers/web_scraper.py

Two scraping strategies:

1. Firecrawl API  — for JS-heavy pages (OddsPortal, BetExplorer, Flashscore).
   Handles React/Next.js SPAs by running a headless browser server-side.
   Costs Firecrawl credits — use sparingly (only on manual scans).

2. curl_cffi + parsel — for lightweight static pages (injury news, lineups).
   Free, fast, no credit cost.

monster.bet mimicry:
   We send the same Sec-Fetch / Origin / Referer headers their SPA uses
   when calling their own internal API endpoints. This makes our traffic
   indistinguishable from a legitimate browser session on their site.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from parsel import Selector

from config import cfg
from core.http_client import HTTPClient

logger = logging.getLogger(__name__)

FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"


@dataclass
class ScrapedItem:
    source_url: str
    sport: str
    data_type: str       # "fixture" | "news_headline" | "odds_hint" | "line_move"
    content: Dict[str, Any]


# ─── Firecrawl scraper (JS-rendered pages) ────────────────────────────────────

class FirecrawlScraper:
    """
    Uses the Firecrawl API to scrape JavaScript-heavy pages.
    Returns clean markdown or structured data from React SPAs.

    Conserve credits: only call during manual scans, not alert-mode checks.
    """

    def __init__(self, http: HTTPClient):
        self.http = http
        self.key = cfg.api.firecrawl_key
        self.enabled = bool(self.key) and cfg.quota.firecrawl_enabled

    async def scrape_url(self, url: str, extract_schema: Optional[dict] = None) -> Optional[str]:
        """
        Scrape a single URL using Firecrawl.
        Returns clean markdown text of the page.
        extract_schema: optional JSON schema for structured extraction.
        """
        if not self.enabled:
            logger.debug("Firecrawl disabled or no key")
            return None

        headers = {"Authorization": f"Bearer {self.key}"}
        payload: dict = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
        }
        if extract_schema:
            payload["extract"] = {"schema": extract_schema}

        result = await self.http.post_json(
            f"{FIRECRAWL_BASE}/scrape",
            json=payload,
            headers=headers,
        )

        if result and result.get("success"):
            return result.get("data", {}).get("markdown", "")
        logger.warning(f"Firecrawl scrape failed for {url}: {result}")
        return None

    async def scrape_oddsportal(self, sport: str = "basketball", league: str = "usa/nba") -> List[ScrapedItem]:
        """
        Scrape OddsPortal for odds comparison data.
        OddsPortal is a React SPA — requires Firecrawl or Playwright.
        """
        url = f"https://www.oddsportal.com/{sport}/{league}/"

        # Structured extraction schema — Firecrawl will use LLM to extract this
        schema = {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "home_team": {"type": "string"},
                            "away_team": {"type": "string"},
                            "date": {"type": "string"},
                            "odds_home": {"type": "string"},
                            "odds_draw": {"type": "string"},
                            "odds_away": {"type": "string"},
                            "num_bookmakers": {"type": "string"},
                        }
                    }
                }
            }
        }

        text = await self.scrape_url(url, extract_schema=schema)
        if not text:
            return []

        items = []
        # Parse the markdown table Firecrawl returns
        for line in text.splitlines():
            if " vs " in line.lower() or "|" in line:
                items.append(ScrapedItem(
                    source_url=url,
                    sport=sport,
                    data_type="odds_hint",
                    content={"raw_line": line.strip()},
                ))
        logger.info(f"Firecrawl/OddsPortal: {len(items)} lines from {url}")
        return items

    async def scrape_betexplorer(self, sport: str = "football", league: str = "england/premier-league") -> List[ScrapedItem]:
        """
        Scrape BetExplorer for historical odds and line movement.
        """
        url = f"https://www.betexplorer.com/{sport}/{league}/results/"
        text = await self.scrape_url(url)
        if not text:
            return []

        items = []
        for line in text.splitlines():
            if " - " in line and any(c.isdigit() for c in line):
                items.append(ScrapedItem(
                    source_url=url,
                    sport=sport,
                    data_type="fixture",
                    content={"raw_line": line.strip()},
                ))
        return items


# ─── Lightweight static scraper ───────────────────────────────────────────────

class WebScraper:
    """
    Fast static scraper for injury news and lineup info.
    Uses curl_cffi (no Firecrawl credits needed).
    """

    def __init__(self, http: HTTPClient):
        self.http = http
        self.firecrawl = FirecrawlScraper(http)

    async def scrape_all(self, use_firecrawl: bool = False) -> List[ScrapedItem]:
        """
        Run all scrapers. use_firecrawl=True only on manual scans.
        """
        items: List[ScrapedItem] = []

        # Always do the lightweight stuff
        injury_news = await self.scrape_injury_news()
        items.extend([
            ScrapedItem("bbc_sport", "soccer", "news_headline", {"headline": h["headline"]})
            for h in injury_news
        ])

        # Firecrawl only on manual scans (costs credits)
        if use_firecrawl and self.firecrawl.enabled:
            logger.info("Running Firecrawl scrapes (manual scan)...")
            portal_items = await self.firecrawl.scrape_oddsportal("basketball", "usa/nba")
            items.extend(portal_items)

        logger.info(f"WebScraper: {len(items)} total items")
        return items

    async def scrape_injury_news(self) -> List[Dict[str, str]]:
        """
        Pull injury/suspension news from BBC Sport (free, static HTML).
        Returns: [{"headline": str, "sport": str}]
        """
        url = "https://www.bbc.com/sport/football"
        html = await self.http.get_html(url)
        if not html:
            return []

        sel = Selector(text=html)
        headlines = []
        injury_keywords = [
            "injur", "suspend", "out ", "doubt", "miss", "ruled out",
            "lineup", "squad", "return", "fit for", "unavailable",
        ]

        for article in sel.css("article"):
            headline = (
                article.css("h3::text").get() or
                article.css("h2::text").get() or
                article.css(".gs-c-promo-heading__title::text").get()
            )
            if headline:
                text = headline.strip().lower()
                if any(kw in text for kw in injury_keywords):
                    headlines.append({"headline": headline.strip(), "sport": "soccer"})

        logger.info(f"Injury/news headlines scraped: {len(headlines)}")
        return headlines

    async def scrape_line_moves(self, sport: str = "nba") -> List[ScrapedItem]:
        """
        Scrape line movement from a public tracker.
        Uses Firecrawl for JS-rendered content.
        """
        if not self.firecrawl.enabled:
            return []
        url = f"https://www.sportsinsights.com/betting-tools/line-watcher/?sport={sport}"
        text = await self.firecrawl.scrape_url(url)
        if not text:
            return []

        items = []
        for line in text.splitlines():
            # Look for lines that mention odds movement
            if any(kw in line.lower() for kw in ["-", "+", "moved", "steam", "sharp"]):
                items.append(ScrapedItem(
                    source_url=url,
                    sport=sport,
                    data_type="line_move",
                    content={"raw": line.strip()},
                ))
        return items
