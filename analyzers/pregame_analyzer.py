"""
analyzers/pregame_analyzer.py

Pre-game analysis engine. For each event it produces a MatchAnalysis
containing the signals the AI model uses to decide if there's value.

Signals computed:
  1. Team form          — last 5 results (W/D/L), points per game
  2. Head-to-head       — historical results between these two teams
  3. Line movement      — how much the odds have moved (sharp money indicator)
  4. Market consensus   — agreement between bookmakers
  5. Home/away bias     — home advantage adjustment
  6. Injury flag        — whether a key player is flagged in news
  7. Overround delta    — difference between our fair prob and the market prob

Data source: api-sports.io (free tier: 100 req/day)
Falls back to market-only analysis if API key is missing.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import cfg
from core.http_client import HTTPClient
from scrapers.odds_aggregator import Event

logger = logging.getLogger(__name__)

API_SPORTS_BASE = "https://v3.football.api-sports.io"  # change host per sport

# ─── Output data model ────────────────────────────────────────────────────────

@dataclass
class FormRecord:
    """Recent form for one team."""
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0
    points_per_game: float = 0.0
    form_string: str = ""       # e.g. "WWDLW" most recent last


@dataclass
class H2HRecord:
    """Head-to-head history between two teams."""
    total_played: int = 0
    home_team_wins: int = 0
    away_team_wins: int = 0
    draws: int = 0
    home_team_avg_goals: float = 0.0
    away_team_avg_goals: float = 0.0
    last_5_results: List[str] = field(default_factory=list)   # "home 2-1 away"


@dataclass
class LineMovement:
    """Tracks how the odds have moved (market signal)."""
    outcome: str                # "home_win" / "away_win" / "draw"
    opening_odds: float = 0.0  # first recorded odds
    current_odds: float = 0.0  # latest odds
    movement_pct: float = 0.0  # % change — negative = shortened (money came in)
    direction: str = "stable"  # "shortened", "drifted", "stable"
    is_sharp_signal: bool = False  # big shortening with no obvious news = sharp money


@dataclass
class MatchAnalysis:
    """
    Complete pre-game analysis for one event.
    This is what gets fed into the AI model as features.
    """
    event: Event

    # Form data
    home_form: Optional[FormRecord] = None
    away_form: Optional[FormRecord] = None

    # Head-to-head
    h2h: Optional[H2HRecord] = None

    # Line movement (one per main outcome)
    line_movements: List[LineMovement] = field(default_factory=list)

    # Derived signals (floats between 0.0 and 1.0 or raw counts)
    home_advantage_factor: float = 0.0    # 0 = no advantage, 1 = strong
    injury_flag_home: bool = False        # True if a key player flagged out
    injury_flag_away: bool = False
    market_confidence: float = 0.0       # How many books agree (0-1)

    # Pre-computed feature vector for the ML model
    features: Dict[str, float] = field(default_factory=dict)

    def to_feature_vector(self) -> Dict[str, float]:
        """
        Convert all analysis into a flat dict of named features.
        Every feature is a float — the ML model needs this.
        """
        f: Dict[str, float] = {}

        # ── Odds-based features ───────────────────────────────────────────────
        h2h_market = self.event.markets.get("h2h", {})
        f["home_best_odds"] = h2h_market.get("home_win", {}).best_odds if hasattr(
            h2h_market.get("home_win", {}), "best_odds") else 0.0
        f["away_best_odds"] = h2h_market.get("away_win", {}).best_odds if hasattr(
            h2h_market.get("away_win", {}), "best_odds") else 0.0
        f["draw_best_odds"] = h2h_market.get("draw", {}).best_odds if hasattr(
            h2h_market.get("draw", {}), "best_odds") else 0.0

        # Use implied probabilities directly as features
        probs = self.event.implied_probs
        f["implied_prob_home"] = probs.get("home_win", 0.0)
        f["implied_prob_away"] = probs.get("away_win", 0.0)
        f["implied_prob_draw"] = probs.get("draw", 0.0)
        f["overround"] = self.event.overround

        # ── Form features ─────────────────────────────────────────────────────
        if self.home_form:
            f["home_ppg"] = self.home_form.points_per_game
            f["home_win_rate"] = (self.home_form.wins / self.home_form.played
                                  if self.home_form.played else 0.0)
            f["home_goals_scored_avg"] = self.home_form.goals_for
            f["home_goals_conceded_avg"] = self.home_form.goals_against
        else:
            f["home_ppg"] = f["home_win_rate"] = 0.0
            f["home_goals_scored_avg"] = f["home_goals_conceded_avg"] = 0.0

        if self.away_form:
            f["away_ppg"] = self.away_form.points_per_game
            f["away_win_rate"] = (self.away_form.wins / self.away_form.played
                                  if self.away_form.played else 0.0)
            f["away_goals_scored_avg"] = self.away_form.goals_for
            f["away_goals_conceded_avg"] = self.away_form.goals_against
        else:
            f["away_ppg"] = f["away_win_rate"] = 0.0
            f["away_goals_scored_avg"] = f["away_goals_conceded_avg"] = 0.0

        # PPG difference — positive = home team in better form
        f["ppg_diff"] = f["home_ppg"] - f["away_ppg"]

        # ── H2H features ─────────────────────────────────────────────────────
        if self.h2h and self.h2h.total_played > 0:
            total = self.h2h.total_played
            f["h2h_home_win_rate"] = self.h2h.home_team_wins / total
            f["h2h_away_win_rate"] = self.h2h.away_team_wins / total
            f["h2h_draw_rate"] = self.h2h.draws / total
            f["h2h_total_played"] = float(total)
            f["h2h_avg_goals_home"] = self.h2h.home_team_avg_goals
            f["h2h_avg_goals_away"] = self.h2h.away_team_avg_goals
        else:
            for k in ["h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
                      "h2h_total_played", "h2h_avg_goals_home", "h2h_avg_goals_away"]:
                f[k] = 0.0

        # ── Line movement features ────────────────────────────────────────────
        for lm in self.line_movements:
            prefix = lm.outcome
            f[f"{prefix}_movement_pct"] = lm.movement_pct
            f[f"{prefix}_is_sharp"] = float(lm.is_sharp_signal)

        # ── Contextual features ───────────────────────────────────────────────
        f["home_advantage"] = self.home_advantage_factor
        f["injury_home"] = float(self.injury_flag_home)
        f["injury_away"] = float(self.injury_flag_away)
        f["market_confidence"] = self.market_confidence
        f["hours_to_kickoff"] = self.event.hours_to_kickoff

        self.features = f
        return f


# ─── Analyzer ─────────────────────────────────────────────────────────────────

class PreGameAnalyzer:
    """
    Enriches each Event with team stats, form, H2H, and line movement.

    If API_SPORTS_KEY is set: full analysis with real stats
    If not:                   market-only analysis (still useful for value detection)
    """

    def __init__(self, http: HTTPClient):
        self.http = http
        self.api_key = cfg.api.api_sports_key
        # Cache previously seen odds so we can detect line movement
        # Format: { event_id -> { outcome -> first_seen_odds } }
        self._odds_history: Dict[str, Dict[str, float]] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    async def analyse_events(self, events: List[Event],
                              injury_news: Optional[List[dict]] = None
                              ) -> List[MatchAnalysis]:
        """
        Main entry point. Takes a list of Event objects and returns
        a list of MatchAnalysis objects with all signals populated.
        """
        analyses = []
        injury_texts = [item["headline"].lower() for item in (injury_news or [])]

        for event in events:
            try:
                analysis = await self._analyse_one(event, injury_texts)
                analysis.to_feature_vector()   # pre-compute features
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Analysis failed for {event.home_team} vs {event.away_team}: {e}")

        logger.info(f"Analysed {len(analyses)}/{len(events)} events")
        return analyses

    # ── Private ───────────────────────────────────────────────────────────────

    async def _analyse_one(self, event: Event,
                            injury_texts: List[str]) -> MatchAnalysis:
        analysis = MatchAnalysis(event=event)

        # 1. Fetch team form and H2H if we have an API key
        if self.api_key and event.sport.startswith("soccer"):
            home_id, away_id = await self._lookup_team_ids(
                event.home_team, event.away_team)
            if home_id and away_id:
                analysis.home_form = await self._fetch_form(home_id, event.home_team)
                analysis.away_form = await self._fetch_form(away_id, event.away_team)
                analysis.h2h = await self._fetch_h2h(home_id, away_id, event)
        else:
            # No API key — fall back to market-implied form estimate
            analysis.home_form = self._estimate_form_from_odds(event, "home")
            analysis.away_form = self._estimate_form_from_odds(event, "away")

        # 2. Line movement — compare current odds to first-seen odds
        analysis.line_movements = self._compute_line_movement(event)

        # 3. Home advantage (sport-specific coefficients)
        analysis.home_advantage_factor = self._home_advantage(event.sport)

        # 4. Injury flags — check if team name appears near an injury keyword
        analysis.injury_flag_home = self._check_injury(event.home_team, injury_texts)
        analysis.injury_flag_away = self._check_injury(event.away_team, injury_texts)

        # 5. Market confidence — how many bookmakers are pricing this event
        h2h = event.markets.get("h2h", {})
        if h2h:
            max_books = max((o.num_books for o in h2h.values()), default=0)
            analysis.market_confidence = min(max_books / 10.0, 1.0)  # normalise to 0-1

        return analysis

    async def _lookup_team_ids(self, home: str, away: str):
        """Look up team IDs from api-sports by name. Returns (home_id, away_id)."""
        async def get_id(name: str) -> Optional[int]:
            url = f"{API_SPORTS_BASE}/teams"
            data = await self.http.get_json(
                url,
                params={"search": name},
                headers={"x-apisports-key": self.api_key},
            )
            if data and data.get("results", 0) > 0:
                return data["response"][0]["team"]["id"]
            return None

        home_id = await get_id(home)
        away_id = await get_id(away)
        return home_id, away_id

    async def _fetch_form(self, team_id: int, team_name: str,
                           last_n: int = 5) -> FormRecord:
        """
        Pull last N fixtures for a team and compute form stats.
        """
        url = f"{API_SPORTS_BASE}/fixtures"
        data = await self.http.get_json(
            url,
            params={"team": team_id, "last": last_n, "status": "FT"},
            headers={"x-apisports-key": self.api_key},
        )

        record = FormRecord(team=team_name)
        if not data or "response" not in data:
            return record

        fixtures = data["response"]
        record.played = len(fixtures)

        goals_for_total = 0.0
        goals_against_total = 0.0
        form_chars = []

        for fixture in fixtures:
            teams = fixture.get("teams", {})
            goals = fixture.get("goals", {})
            is_home = teams.get("home", {}).get("name") == team_name

            if is_home:
                gf = goals.get("home") or 0
                ga = goals.get("away") or 0
                won = teams.get("home", {}).get("winner")
            else:
                gf = goals.get("away") or 0
                ga = goals.get("home") or 0
                won = teams.get("away", {}).get("winner")

            goals_for_total += gf
            goals_against_total += ga

            if won is True:
                record.wins += 1
                form_chars.append("W")
            elif won is False:
                record.losses += 1
                form_chars.append("L")
            else:
                record.draws += 1
                form_chars.append("D")

        if record.played > 0:
            record.goals_for = round(goals_for_total / record.played, 2)
            record.goals_against = round(goals_against_total / record.played, 2)
            points = record.wins * 3 + record.draws
            record.points_per_game = round(points / record.played, 2)
            record.form_string = "".join(form_chars)

        return record

    async def _fetch_h2h(self, home_id: int, away_id: int,
                          event: Event) -> H2HRecord:
        """Fetch head-to-head history between two teams."""
        url = f"{API_SPORTS_BASE}/fixtures/headtohead"
        data = await self.http.get_json(
            url,
            params={"h2h": f"{home_id}-{away_id}", "last": 10},
            headers={"x-apisports-key": self.api_key},
        )

        record = H2HRecord()
        if not data or "response" not in data:
            return record

        fixtures = data["response"]
        record.total_played = len(fixtures)
        home_goals_total = 0
        away_goals_total = 0

        for fix in fixtures:
            teams = fix.get("teams", {})
            goals = fix.get("goals", {})
            home_name = teams.get("home", {}).get("name")
            home_winner = teams.get("home", {}).get("winner")
            away_winner = teams.get("away", {}).get("winner")

            home_goals = goals.get("home") or 0
            away_goals = goals.get("away") or 0

            # Re-orient so "home" always means the team playing at home in THIS match
            if home_name == event.home_team:
                home_goals_total += home_goals
                away_goals_total += away_goals
                if home_winner:
                    record.home_team_wins += 1
                elif away_winner:
                    record.away_team_wins += 1
                else:
                    record.draws += 1
                result = f"{event.home_team} {home_goals}-{away_goals} {event.away_team}"
            else:
                home_goals_total += away_goals
                away_goals_total += home_goals
                if away_winner:
                    record.home_team_wins += 1
                elif home_winner:
                    record.away_team_wins += 1
                else:
                    record.draws += 1
                result = f"{event.home_team} {away_goals}-{home_goals} {event.away_team}"

            record.last_5_results.append(result)

        if record.total_played > 0:
            record.home_team_avg_goals = round(home_goals_total / record.total_played, 2)
            record.away_team_avg_goals = round(away_goals_total / record.total_played, 2)
            record.last_5_results = record.last_5_results[-5:]  # keep only last 5

        return record

    def _estimate_form_from_odds(self, event: Event, side: str) -> FormRecord:
        """
        No API key fallback: estimate implied form from market odds.
        A lower implied probability = weaker team = worse estimated form.
        """
        probs = event.implied_probs
        if side == "home":
            team = event.home_team
            win_prob = probs.get("home_win", 0.33)
        else:
            team = event.away_team
            win_prob = probs.get("away_win", 0.33)

        # Rough approximation: win_prob -> PPG estimate
        # Average PPG in a 3-outcome sport ranges from 0 (always loses) to 3 (always wins)
        estimated_ppg = round(win_prob * 3, 2)
        return FormRecord(
            team=team,
            played=5,   # assumed
            points_per_game=estimated_ppg,
            form_string="?????",  # unknown without real data
        )

    def _compute_line_movement(self, event: Event) -> List[LineMovement]:
        """
        Compare current odds to the first time we saw this event.
        A significant shortening (price dropping) on an outcome without
        obvious news is a classic "sharp money" indicator.
        """
        movements = []
        h2h = event.markets.get("h2h", {})

        for outcome, odds_data in h2h.items():
            current = odds_data.best_odds
            if current <= 0:
                continue

            history = self._odds_history.setdefault(event.event_id, {})
            if outcome not in history:
                # First time seeing this — store as opening
                history[outcome] = current
                movements.append(LineMovement(
                    outcome=outcome,
                    opening_odds=current,
                    current_odds=current,
                    movement_pct=0.0,
                    direction="stable",
                ))
                continue

            opening = history[outcome]
            movement_pct = round((current - opening) / opening * 100, 2)

            if movement_pct < -3.0:
                direction = "shortened"    # price came in — money bet on this
            elif movement_pct > 3.0:
                direction = "drifted"      # price went out — money against this
            else:
                direction = "stable"

            # Sharp signal: significant shortening (>5%) before most public bets
            is_sharp = (movement_pct < -5.0 and event.hours_to_kickoff > 6)

            movements.append(LineMovement(
                outcome=outcome,
                opening_odds=opening,
                current_odds=current,
                movement_pct=movement_pct,
                direction=direction,
                is_sharp_signal=is_sharp,
            ))

        return movements

    def _home_advantage(self, sport: str) -> float:
        """
        Sport-specific home advantage factor (0.0 – 1.0).
        Based on empirical win rates from historical data.
        """
        advantages = {
            "soccer_epl": 0.46,
            "soccer_uefa_champs_league": 0.54,
            "soccer_germany_bundesliga": 0.44,
            "basketball_nba": 0.60,
            "americanfootball_nfl": 0.57,
            "mma_mixed_martial_arts": 0.0,      # no home advantage in MMA
            "boxing_boxing": 0.0,
            "tennis_atp_french_open": 0.0,      # neutral venue
        }
        return advantages.get(sport, 0.45)  # default to average ~45%

    def _check_injury(self, team_name: str, injury_texts: List[str]) -> bool:
        """
        Very simple: returns True if team name appears in any injury headline.
        A more sophisticated version would use NER to find player names.
        """
        team_lower = team_name.lower()
        # Split multi-word team names and check each meaningful word
        words = [w for w in team_lower.split() if len(w) > 3]
        for text in injury_texts:
            if any(word in text for word in words):
                return True
        return False
