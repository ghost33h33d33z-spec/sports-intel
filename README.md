# Sports Betting Intel Bot

Pre-game value bet detector. Monitors odds across multiple sports,
analyses team form + H2H + line movement, and alerts you via Telegram
when a bet has positive expected value.

---

## What It Does

```
Every 15 min:
  ┌─────────────────────────────────────────────────────┐
  │ 1. Fetch odds    TheOddsAPI → normalise all markets  │
  │ 2. Scrape        Injury news, team headlines          │
  │ 3. Analyse       Form, H2H, line movement per event  │
  │ 4. Detect        AI compares our prob vs market prob  │
  │ 5. Alert         Telegram: edge, EV, confidence bar  │
  └─────────────────────────────────────────────────────┘
```

A **value bet** is flagged when:
- Our estimated probability > market implied probability by ≥4%
- AI confidence ≥ 62%
- Odds between 1.30 and 8.00
- Kickoff within 24 hours

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

On Termux:
```bash
pkg install python
pip install -r requirements.txt
```

### 2. Get API Keys

| Service | Cost | What it provides |
|---|---|---|
| [TheOddsAPI](https://the-odds-api.com) | Free (500 req/mo) | Odds from 40+ bookmakers |
| [API-Sports](https://api-sports.io) | Free (100 req/day) | Team form, H2H, injuries |
| Telegram bot | Free | Alerts on your phone |

Create your Telegram bot:
1. Message `@BotFather` → `/newbot`
2. Get your chat ID via `@userinfobot`

### 3. Configure

```bash
cp .env.example .env
# edit .env with your keys
source .env
```

Or export directly:
```bash
export THE_ODDS_API_KEY="your_key"
export API_SPORTS_KEY="your_key"
export TG_TOKEN="123456:ABC..."
export TG_CHAT="987654321"
```

### 4. Run

```bash
python main.py
```

Background (keeps running after you close terminal):
```bash
nohup python main.py > logs/output.log 2>&1 &
tail -f logs/output.log    # watch it
pkill -f "python main.py"  # stop it
```

---

## Project Structure

```
sports_intel/
├── main.py                      ← Entry point, scan loop
├── config.py                    ← All settings (edit here)
├── requirements.txt
├── .env.example
│
├── core/
│   └── http_client.py           ← Browser-impersonating HTTP with retries
│
├── scrapers/
│   ├── odds_aggregator.py       ← TheOddsAPI → normalised Event objects
│   └── web_scraper.py           ← Injury news scraper
│
├── analyzers/
│   └── pregame_analyzer.py      ← Form, H2H, line movement analysis
│
├── models/
│   └── value_detector.py        ← AI value bet detection + EV calc
│
├── alerts/
│   └── telegram_alerts.py       ← Rich Telegram message formatting
│
├── data/
│   └── bet_results.jsonl        ← Result log (auto-created, used for retraining)
│
└── logs/
    └── intel.log                ← Rotating log file
```

---

## Tuning the Bot

All thresholds are in `config.py` → `BettingConfig`:

| Setting | Default | What it does |
|---|---|---|
| `min_edge` | 4% | Minimum prob edge over market |
| `min_confidence` | 62% | AI model confidence required |
| `min_odds` | 1.30 | Ignore heavy favourites below this |
| `max_odds` | 8.00 | Ignore longshots above this |
| `pregame_hours_before` | 24 | Only look at games within this window |
| `scan_interval` | 900s | How often to scan (seconds) |

---

## How The AI Works

**Phase 1 (no training data yet):** Heuristic mode.

Adjusts the market implied probability using:
- Form signal: PPG difference between teams (+3% per PPG)
- H2H signal: historical win rate between these two teams
- Line movement: sharp shortening (+4% if odds shortened >5% with no news)
- Injury penalty: -5% if key player flagged injured

**Phase 2 (after 100+ labelled results):** Gradient Boosting model.

The 25 features fed to the model:
```
implied_prob_home/away/draw   — what the market thinks
overround                     — bookmaker margin
home/away_ppg                 — points per game (form)
ppg_diff                      — form differential
home/away_win_rate            — recent win %
h2h_*                         — head-to-head stats
home_advantage                — sport-specific home factor
injury_home/away              — injury flag
market_confidence             — how many books priced this
hours_to_kickoff              — time until game
*_movement_pct                — line movement %
*_is_sharp                    — sharp money signal
```

Results are logged to `data/bet_results.jsonl`. The model retrains
automatically every 50 new results once 100 are available.

---

## Limitations & Disclaimers

- **This is for research/education.** Betting carries financial risk.
- The AI starts in heuristic mode with no historical edge — it needs
  your local results to improve over time.
- TheOddsAPI free tier is 500 requests/month. Each scan uses 1 request
  per sport. 8 sports × 4 scans/hour × 24h = 768 req/day — reduce
  `SPORTS_TO_MONITOR` or increase `scan_interval` if you hit limits.
- API-Sports free tier is 100 req/day. The form/H2H lookups use this.
  Without it the system falls back to market-implied estimates only.
