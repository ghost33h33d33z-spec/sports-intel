"""
models/value_detector.py

The AI value betting engine. Identifies bets where our estimated true
probability is HIGHER than what the bookmaker's odds imply.

How value betting works:
  - Bookmaker odds of 2.00 imply a 50% win probability
  - If WE think the real probability is 56%, then betting at 2.00 has +EV
  - Expected Value (EV) = (our_prob * decimal_odds) - 1
  - Positive EV = value bet

What this module does:
  1. Takes MatchAnalysis features as input
  2. Uses a trained Random Forest to estimate true win probability
  3. Compares our estimate vs implied market probability
  4. Flags bets where edge >= configured minimum
  5. Saves results and retrains when enough data accumulates

The model starts with a heuristic bootstrap (no historical data needed)
and improves over time as results come in.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyzers.pregame_analyzer import MatchAnalysis
from config import cfg

logger = logging.getLogger(__name__)

# ─── Output data model ────────────────────────────────────────────────────────

@dataclass
class ValueBet:
    """A single identified value bet opportunity."""
    event_id: str
    sport: str
    home_team: str
    away_team: str
    outcome: str                # "home_win", "away_win", "draw"
    best_odds: float
    market_implied_prob: float  # what the bookmaker thinks
    our_prob: float             # what our model thinks
    edge: float                 # our_prob - market_implied_prob
    expected_value: float       # (our_prob * odds) - 1  — positive is profitable
    confidence: float           # model confidence (0.0 – 1.0)
    sharpest_book: str
    hours_to_kickoff: float
    sharp_money_signal: bool    # is there line movement suggesting sharp bettors
    injury_concern: bool        # is a key player flagged as injured
    kick_off_time: str
    features_used: Dict[str, float] = field(default_factory=dict)

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        ev_str = f"+{self.expected_value:.1%}" if self.expected_value > 0 else f"{self.expected_value:.1%}"
        return (
            f"{self.home_team} vs {self.away_team} | "
            f"{self.outcome.replace('_', ' ').title()} @ {self.best_odds:.2f} | "
            f"Edge: {self.edge:.1%} | EV: {ev_str} | "
            f"Conf: {self.confidence:.0%} | KO: {self.hours_to_kickoff:.1f}h"
        )


# ─── Feature engineering ──────────────────────────────────────────────────────

# These are the exact features (in order) that the model expects.
# Adding new features = retrain. Order MUST stay consistent.
FEATURE_NAMES = [
    "implied_prob_home",
    "implied_prob_away",
    "implied_prob_draw",
    "overround",
    "home_ppg",
    "away_ppg",
    "ppg_diff",
    "home_win_rate",
    "away_win_rate",
    "h2h_home_win_rate",
    "h2h_away_win_rate",
    "h2h_draw_rate",
    "h2h_total_played",
    "h2h_avg_goals_home",
    "h2h_avg_goals_away",
    "home_advantage",
    "injury_home",
    "injury_away",
    "market_confidence",
    "hours_to_kickoff",
    "home_win_movement_pct",
    "away_win_movement_pct",
    "draw_movement_pct",
    "home_win_is_sharp",
    "away_win_is_sharp",
]


def extract_feature_vector(analysis: MatchAnalysis) -> np.ndarray:
    """
    Pull the features we need from a MatchAnalysis into a fixed-length array.
    Missing features default to 0.0 — this keeps everything stable.
    """
    f = analysis.features   # pre-computed dict from MatchAnalysis.to_feature_vector()
    return np.array([f.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)


# ─── Value Detector ───────────────────────────────────────────────────────────

class ValueDetector:
    """
    Wraps the ML model and value calculation logic.

    On first run with no saved model: uses a heuristic-based estimator.
    After MIN_SAMPLES results are collected: trains a proper ML model.

    The model predicts: "Is the home team more likely to win than the
    market implies?" (same logic applied separately for away/draw).
    """

    def __init__(self):
        self.model_path = Path(cfg.model.model_path)
        self.scaler_path = Path(cfg.model.scaler_path)
        self.result_log = Path("data/bet_results.jsonl")
        self.result_log.parent.mkdir(parents=True, exist_ok=True)

        self.pipeline: Optional[Pipeline] = None
        self._load_or_bootstrap()

    # ── Public ────────────────────────────────────────────────────────────────

    def find_value_bets(self, analyses: List[MatchAnalysis]) -> List[ValueBet]:
        """
        Main entry point. Given a list of MatchAnalysis objects,
        return all bets that meet our value threshold.
        """
        value_bets: List[ValueBet] = []

        for analysis in analyses:
            for bet in self._evaluate_event(analysis):
                value_bets.append(bet)

        # Sort by expected value descending
        value_bets.sort(key=lambda b: b.expected_value, reverse=True)

        logger.info(f"Found {len(value_bets)} value bets from {len(analyses)} events")
        return value_bets

    def log_result(self, event_id: str, outcome: str, won: bool) -> None:
        """
        Record actual result so we can retrain.
        Call this after a match completes.
        """
        record = {
            "event_id": event_id,
            "outcome": outcome,
            "won": won,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.result_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug(f"Logged result: {event_id} {outcome} {'WIN' if won else 'LOSS'}")

    def maybe_retrain(self, all_analyses: List[MatchAnalysis]) -> bool:
        """
        Retrain the model if we have enough labelled examples.
        Returns True if retrain happened.
        """
        labelled = self._load_labelled_results()
        if len(labelled) < cfg.model.min_samples_to_train:
            logger.info(f"Need {cfg.model.min_samples_to_train} samples to retrain, "
                        f"have {len(labelled)}")
            return False

        # Build X, y from analysis cache + result log
        # (simplified — in production you'd join on event_id)
        logger.info(f"Retraining on {len(labelled)} labelled samples...")
        X, y = self._build_training_data(all_analyses, labelled)
        if len(X) < cfg.model.min_samples_to_train:
            return False

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        return True

    # ── Private ───────────────────────────────────────────────────────────────

    def _evaluate_event(self, analysis: MatchAnalysis) -> List[ValueBet]:
        """
        For each outcome in an event, check if there's a value bet.
        Returns only those that clear the minimum edge and confidence thresholds.
        """
        bets = []
        h2h = analysis.event.markets.get("h2h", {})
        probs = analysis.event.implied_probs

        outcomes_to_check = ["home_win", "away_win"]
        # Only check draw for soccer
        if analysis.event.sport.startswith("soccer"):
            outcomes_to_check.append("draw")

        for outcome in outcomes_to_check:
            if outcome not in h2h:
                continue
            odds_data = h2h[outcome]
            best_odds = odds_data.best_odds
            if best_odds < cfg.betting.min_odds or best_odds > cfg.betting.max_odds:
                continue

            market_prob = probs.get(outcome, 0.0)
            if market_prob <= 0:
                continue

            # Get our probability estimate
            our_prob, confidence = self._predict_prob(analysis, outcome)
            if our_prob <= 0:
                continue

            edge = our_prob - market_prob
            ev = (our_prob * best_odds) - 1.0

            # Check thresholds
            if edge < cfg.betting.min_edge:
                continue
            if confidence < cfg.betting.min_confidence:
                continue

            # Check for sharp money / injury signals
            sharp_signal = any(
                lm.is_sharp_signal and lm.outcome == outcome
                for lm in analysis.line_movements
            )
            injury_concern = (
                (outcome == "home_win" and analysis.injury_flag_home) or
                (outcome == "away_win" and analysis.injury_flag_away)
            )

            bets.append(ValueBet(
                event_id=analysis.event.event_id,
                sport=analysis.event.sport,
                home_team=analysis.event.home_team,
                away_team=analysis.event.away_team,
                outcome=outcome,
                best_odds=best_odds,
                market_implied_prob=market_prob,
                our_prob=round(our_prob, 4),
                edge=round(edge, 4),
                expected_value=round(ev, 4),
                confidence=round(confidence, 4),
                sharpest_book=analysis.event.sharpest_book,
                hours_to_kickoff=analysis.event.hours_to_kickoff,
                sharp_money_signal=sharp_signal,
                injury_concern=injury_concern,
                kick_off_time=analysis.event.commence_time.strftime("%a %d %b %H:%M UTC"),
                features_used=analysis.features,
            ))

        return bets

    def _predict_prob(self, analysis: MatchAnalysis,
                       outcome: str) -> Tuple[float, float]:
        """
        Predict true probability for a specific outcome.
        Returns (probability, confidence).

        Uses trained model if available, else heuristic fallback.
        """
        features = extract_feature_vector(analysis)

        if self.pipeline is not None:
            try:
                proba = self.pipeline.predict_proba([features])[0]
                # Binary classifier: class 1 = value bet exists
                prob = float(proba[1])
                # Confidence = distance from 0.5 (how decisive the model is)
                confidence = abs(prob - 0.5) * 2
                return prob, confidence
            except Exception as e:
                logger.debug(f"Model predict failed: {e}, using heuristic")

        # ── Heuristic fallback ────────────────────────────────────────────────
        return self._heuristic_prob(analysis, outcome)

    def _heuristic_prob(self, analysis: MatchAnalysis,
                         outcome: str) -> Tuple[float, float]:
        """
        Rule-based probability estimator — no training data needed.
        Adjusts market implied probability using form, H2H, and line movement.

        Returns (adjusted_probability, confidence_score).
        """
        base = analysis.event.implied_probs.get(outcome, 0.33)
        adjustment = 0.0
        signals_used = 0

        # Form signal: if our team is in better form, nudge probability up
        if analysis.home_form and analysis.away_form:
            if outcome == "home_win":
                ppg_edge = analysis.home_form.points_per_game - analysis.away_form.points_per_game
                adjustment += ppg_edge * 0.03   # 3% per PPG difference
                signals_used += 1
            elif outcome == "away_win":
                ppg_edge = analysis.away_form.points_per_game - analysis.home_form.points_per_game
                adjustment += ppg_edge * 0.03
                signals_used += 1

        # H2H signal: historical dominance shifts probability
        if analysis.h2h and analysis.h2h.total_played >= 3:
            h2h = analysis.h2h
            total = h2h.total_played
            if outcome == "home_win":
                h2h_rate = h2h.home_team_wins / total
                adjustment += (h2h_rate - base) * 0.2
                signals_used += 1
            elif outcome == "away_win":
                h2h_rate = h2h.away_team_wins / total
                adjustment += (h2h_rate - base) * 0.2
                signals_used += 1

        # Line movement signal: sharp money shortened this price
        for lm in analysis.line_movements:
            if lm.outcome == outcome and lm.is_sharp_signal:
                adjustment += 0.04   # sharp shortening = +4% prob boost
                signals_used += 1

        # Injury penalty: injured team is less likely to win
        if outcome == "home_win" and analysis.injury_flag_home:
            adjustment -= 0.05
        elif outcome == "away_win" and analysis.injury_flag_away:
            adjustment -= 0.05

        # Clamp the final probability to [0.02, 0.98]
        adjusted_prob = max(0.02, min(0.98, base + adjustment))

        # Confidence: how many independent signals pointed the same way
        confidence = min(0.5 + signals_used * 0.1, 0.90)

        return adjusted_prob, confidence

    def _load_or_bootstrap(self) -> None:
        """Load saved model, or leave as None (heuristic mode)."""
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        logger.info("No saved model found — using heuristic mode until enough data collected")
        self.pipeline = None

    def _build_pipeline(self) -> Pipeline:
        """Build a fresh sklearn pipeline: scaler + gradient boosting."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ])

    def _load_labelled_results(self) -> List[dict]:
        """Load bet outcomes logged by log_result()."""
        if not self.result_log.exists():
            return []
        results = []
        with open(self.result_log) as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        return results

    def _build_training_data(
        self, analyses: List[MatchAnalysis], results: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Join analyses with result log to build (X, y) training data.
        y=1 means "the bet would have been profitable".
        """
        result_map = {r["event_id"]: r for r in results}
        X_rows, y_rows = [], []

        for analysis in analyses:
            result = result_map.get(analysis.event.event_id)
            if not result:
                continue
            features = extract_feature_vector(analysis)
            won = 1 if result.get("won") else 0
            X_rows.append(features)
            y_rows.append(won)

        if not X_rows:
            return np.array([]), np.array([])
        return np.array(X_rows), np.array(y_rows)
