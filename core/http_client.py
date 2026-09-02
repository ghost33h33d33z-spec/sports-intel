"""
core/http_client.py

HTTP client that mimics the exact fingerprint monster.bet uses:
  - Next.js 14 App Router client (RSC fetch)
  - Cloudflare-passing headers (Sec-Fetch-*, Origin, Referer)
  - curl_cffi browser impersonation (TLS fingerprint matches real Chrome/Edge)
  - Per-domain rate limiting (avoid triggering Cloudflare rate rules)
  - Exponential backoff with browser rotation on 403/429
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from curl_cffi import requests as cffi_req

from config import BROWSER_PROFILES, NEXTJS_HEADERS

logger = logging.getLogger(__name__)


class RateLimiter:
    """Enforce minimum gap between requests to each domain."""

    def __init__(self, min_gap: float = 2.0):
        self.min_gap = min_gap
        self._last: Dict[str, float] = {}

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last.get(host, 0.0)
        if elapsed < self.min_gap:
            await asyncio.sleep(self.min_gap - elapsed + random.uniform(0.1, 0.4))
        self._last[host] = time.monotonic()


class HTTPClient:
    """
    Cloudflare-safe HTTP client.

    Mimics a Next.js SPA running on monster.bet:
    - TLS fingerprint via curl_cffi impersonation
    - Browser-realistic headers (Sec-Fetch, Origin, Referer)
    - Rotates profile on bot-detection responses

    Usage:
        client = HTTPClient()
        data   = await client.get_json("https://...", params={...})
        html   = await client.get_html("https://...")
        result = await client.post_json("https://...", json={...})
    """

    def __init__(
        self,
        retries: int = 3,
        backoff_base: float = 2.0,
        timeout: int = 20,
        rate_gap: float = 1.5,
    ):
        self.retries = retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.limiter = RateLimiter(min_gap=rate_gap)
        self._profile = random.choice(BROWSER_PROFILES)
        self._session = cffi_req.Session(impersonate=self._profile)

    # ── Public ────────────────────────────────────────────────────────────────

    async def get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Optional[Any]:
        resp = await self._get(url, params=params, extra_headers=headers)
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception as e:
            logger.error(f"JSON parse failed {url}: {e}")
            return None

    async def get_html(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Optional[str]:
        resp = await self._get(url, params=params, extra_headers=headers)
        return resp.text if resp else None

    async def post_json(
        self,
        url: str,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Optional[Any]:
        merged = {**NEXTJS_HEADERS, **(headers or {})}
        try:
            await self.limiter.wait(url)
            resp = self._session.post(
                url,
                data=data,
                json=json,
                headers=merged,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"POST failed {url}: {e}")
            return None

    # ── WebSocket helper (for Spro.agency stream) ─────────────────────────────
    # Returns the websockets connection; caller manages the loop.
    # Requires: pip install websockets
    async def ws_connect(self, uri: str):
        try:
            import websockets
            return await websockets.connect(uri, max_size=None)
        except ImportError:
            logger.error("websockets package not installed: pip install websockets")
            return None
        except Exception as e:
            logger.error(f"WebSocket connect failed {uri}: {e}")
            return None

    # ── Private ───────────────────────────────────────────────────────────────

    async def _get(
        self,
        url: str,
        params: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ) -> Optional[cffi_req.Response]:
        merged = {**NEXTJS_HEADERS, **(extra_headers or {})}

        for attempt in range(1, self.retries + 1):
            try:
                await self.limiter.wait(url)
                resp = self._session.get(
                    url,
                    params=params,
                    headers=merged,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp

            except cffi_req.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                logger.warning(f"HTTP {status} — {url} (attempt {attempt}/{self.retries})")
                if status in (403, 429, 503):
                    self._rotate()

            except Exception as e:
                logger.warning(f"Request error {url} attempt {attempt}: {e}")

            if attempt < self.retries:
                wait = self.backoff_base ** attempt + random.uniform(0, 1)
                await asyncio.sleep(wait)

        logger.error(f"All {self.retries} attempts failed: {url}")
        return None

    def _rotate(self) -> None:
        """Switch browser profile after bot-detection response."""
        self._profile = random.choice(BROWSER_PROFILES)
        self._session = cffi_req.Session(impersonate=self._profile)
        logger.debug(f"Rotated to browser profile: {self._profile}")
