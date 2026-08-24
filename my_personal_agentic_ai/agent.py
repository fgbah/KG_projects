import anthropic
import json
from datetime import datetime
from memory import (
    load_memory, get_recent_trades, get_todays_trades,
    get_current_strategy, save_strategy_adjustment,
    save_reflection, get_winning_patterns
)
from regime import get_market_regime, get_regime_adjustments
from config import CLAUDE_API_KEY

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


# ✅ LOCKDOWN: prevent auto-adjustments from corrupting the strategy.
# Set False once N+ valid (non-churning) trades have been logged.
ENABLE_AUTO_STRATEGY_ADJUSTMENTS = False
MIN_TRADES_FOR_AUTO_ADJUST       = 50

# ✅ SAMPLE QUALITY: reject panic-level reflections until enough clean data
MIN_TRADES_FOR_REFLECTION    = 30
CHURN_WINDOW_MINUTES         = 60
CHURN_MAX_REENTRIES_PER_DAY  = 2


# ─────────────────────────────────────────────────────────────
# CONFIG SYNC — read ACTUAL live values from config.py
# Prevents Claude from hallucinating numbers
# ─────────────────────────────────────────────────────────────
def _get_live_strategy_from_config() -> dict:
    """
    Read the strategy values directly from config.py.
    These are the values the bot ACTUALLY uses for trading decisions
    (not whatever drift has built up in agent_memory.json).
    """
    try:
        from config import (
            PENNY_RSI_THRESHOLD, PENNY_VOLUME_SPIKE,
            PENNY_STOP_LOSS_PCT, PENNY_TAKE_PROFIT_PCT,
            PENNY_MAX_TRADE_SIZE, PENNY_MIN_NEWS_SCORE,
            RSI_BUY_THRESHOLD, VOLUME_SPIKE,
            STOP_LOSS_PCT, TAKE_PROFIT_PCT,
            MAX_TRADE_SIZE, MIN_NEWS_SCORE,
        )
        return {
            "penny": {
                "rsi_threshold":   float(PENNY_RSI_THRESHOLD),
                "volume_spike":    float(PENNY_VOLUME_SPIKE),
                "stop_loss_pct":   float(PENNY_STOP_LOSS_PCT),
                "take_profit_pct": float(PENNY_TAKE_PROFIT_PCT),
                "max_trade_size":  float(PENNY_MAX_TRADE_SIZE),
                "min_news_score":  float(PENNY_MIN_NEWS_SCORE),
            },
            "regular": {
                "rsi_threshold":   float(RSI_BUY_THRESHOLD),
                "volume_spike":    float(VOLUME_SPIKE),
                "stop_loss_pct":   float(STOP_LOSS_PCT),
                "take_profit_pct": float(TAKE_PROFIT_PCT),
                "max_trade_size":  float(MAX_TRADE_SIZE),
                "min_news_score":  float(MIN_NEWS_SCORE),
            },
        }
    except Exception as e:
        print(f"⚠️  Could not read live config: {e} — using memory fallback")
        return get_current_strategy()


# ─────────────────────────────────────────────────────────────
# CHURN FILTER — drop rapid re-entries from analysis input
# A trade counts as churn if same ticker traded >N times in one day,
# or if a BUY happens within CHURN_WINDOW of a prior SELL for that ticker.
# ─────────────────────────────────────────────────────────────
def _filter_churning_trades(trades: list) -> tuple[list, dict]:
    """
    Returns (clean_trades, stats) — strips rapid re-entries to give
    Claude only meaningful trade signals.

    Original trades remain in agent_memory.json (we don't mutate disk).
    """
    from datetime import datetime as _dt

    by_ticker_date = {}
    last_action_time = {}

    clean = []
    churning_removed = 0
    churning_tickers = set()

    for t in trades:
        sym  = t.get("symbol", "")
        date = t.get("date", "")
        key  = (sym, date)

        by_ticker_date[key] = by_ticker_date.get(key, 0) + 1

        # Rule 1: > N trades same ticker same day = churning
        if by_ticker_date[key] > CHURN_MAX_REENTRIES_PER_DAY:
            churning_removed += 1
            churning_tickers.add(sym)
            continue

        # Rule 2: BUY within CHURN_WINDOW of last action = churning
        try:
            tstr = t.get("timestamp", "")
            tdt  = _dt.strptime(tstr, "%Y-%m-%d %H:%M:%S")
            if (t.get("action") == "BUY"
                    and sym in last_action_time
                    and (tdt - last_action_time[sym]).total_seconds() < CHURN_WINDOW_MINUTES * 60):
                churning_removed += 1
                churning_tickers.add(sym)
                continue
            last_action_time[sym] = tdt
        except Exception:
            pass

        clean.append(t)

    stats = {
        "original_count": len(trades),
        "clean_count":    len(clean),
        "removed":        churning_removed,
        "churning_tickers": sorted(churning_tickers),
    }
    return clean, stats


def _sample_size_ok(trade_count: int) -> bool:
    """Return True only if we have enough clean trades for reflection."""
    return trade_count >= MIN_TRADES_FOR_REFLECTION


# ─────────────────────────────────────────────────────────────
# HYBRID LEARNING — Tier 2 suggestions + Tier 3 auto-unlock
# ─────────────────────────────────────────────────────────────
SUGGESTIONS_FILE           = "suggestions.json"
TIER3_UNLOCK_CLEAN_TRADES  = 50


def _count_clean_trades() -> int:
    """
    Count trades in agent_memory.json that survive the churn filter.
    Used to decide when to unlock Tier 3 auto-learning.
    """
    try:
        mem  = load_memory()
        raw  = mem.get("trades", [])
        clean, _ = _filter_churning_trades(raw)
        return len(clean)
    except Exception:
        return 0


def _write_suggestion(source: str,
                      trigger_symbol: str,
                      current_strategy: dict,
                      proposed_strategy: dict,
                      reasoning: str,
                      reflection: str,
                      clean_trade_count: int):
    """
    ✅ TIER 2: append Claude's suggestion to suggestions.json for manual review.
    User approves or ignores each suggestion outside the bot.
    """
    import json, os
    from datetime import datetime as _dt

    entry = {
        "timestamp":         _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":            source,
        "trigger_symbol":    trigger_symbol,
        "clean_trade_count": clean_trade_count,
        "reflection":        reflection,
        "reasoning":         reasoning,
        "current_strategy":  current_strategy,
        "proposed_strategy": proposed_strategy,
        "status":            "pending",
    }

    try:
        if os.path.exists(SUGGESTIONS_FILE):
            with open(SUGGESTIONS_FILE, "r") as f:
                items = json.load(f)
        else:
            items = []

        items.append(entry)
        items = items[-100:]

        with open(SUGGESTIONS_FILE, "w") as f:
            json.dump(items, f, indent=2)

        print(f"📝 Suggestion logged to {SUGGESTIONS_FILE} — review with:")
        print(f"   python3 -c \"import json; [print(s) for s in json.load(open('{SUGGESTIONS_FILE}')) if s['status']=='pending']\"")
    except Exception as e:
        print(f"⚠️  Could not write suggestion: {e}")


def _check_tier3_unlock():
    """
    ✅ Auto-flip ENABLE_AUTO_STRATEGY_ADJUSTMENTS to True once
    enough clean trades exist. Prints a loud banner on unlock.
    """
    global ENABLE_AUTO_STRATEGY_ADJUSTMENTS
    if ENABLE_AUTO_STRATEGY_ADJUSTMENTS:
        return
    clean = _count_clean_trades()
    if clean >= TIER3_UNLOCK_CLEAN_TRADES:
        ENABLE_AUTO_STRATEGY_ADJUSTMENTS = True
        print("\n" + "="*60)
        print(f"🔓 TIER 3 UNLOCKED — {clean} clean trades logged")
        print("   Auto-strategy adjustment now ACTIVE")
        print("   The bot will now apply Claude's recommendations")
        print("   automatically. Set ENABLE_AUTO_STRATEGY_ADJUSTMENTS")
        print("   back to False if you want to re-lock.")
        print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────
# 1. AFTER-TRADE REFLECTION
# ─────────────────────────────────────────────────────────────
def reflect_after_trade(trade: dict):
    """
    Claude reviews the just-completed trade.
    ✅ Uses LIVE config values (not stale memory).
    ✅ Filters churning trades from analysis input.
    ✅ Skips reflection entirely if sample too small.
    """
    print(f"\n🧠 Agent reflecting on {trade['symbol']} trade...")

    memory           = load_memory()
    recent_raw       = get_recent_trades(days=7)
    current_strategy = _get_live_strategy_from_config()
    stats            = memory["stats"]

    recent_trades, churn_stats = _filter_churning_trades(recent_raw)
    if churn_stats["removed"] > 0:
        print(f"🧹 Filtered {churn_stats['removed']} churning trades "
              f"({churn_stats['churning_tickers']})")

    if not _sample_size_ok(len(recent_trades)):
        print(f"📊 Sample too small — skipping reflection "
              f"({len(recent_trades)} clean trades, need {MIN_TRADES_FOR_REFLECTION}+)")
        return current_strategy

    trade_summary = "\n".join([
        f"- {t['date']} {t['action']} {t['symbol']} "
        f"({'PENNY' if t['is_penny'] else 'REGULAR'}) "
        f"@ ${t['price']:.4f} | P&L: {t['profit_pct']:+.2f}% | "
        f"RSI: {t['rsi']:.0f} | Vol: {t['volume_ratio']:.1f}x | "
        f"News: {t['news_score']:+.2f} | Reason: {t['reason']}"
        for t in recent_trades[-20:]
    ])

    total_trades     = stats.get("total_trades", 0)
    win_rate_display = (
        f"{round(stats.get('total_wins', 0) / max(total_trades, 1) * 100, 1)}%"
        if total_trades >= 5
        else f"N/A — only {total_trades} trades (need 5+ for reliable stats)"
    )

    prompt = f"""
You are an autonomous trading agent that just completed a trade.
Review this trade and decide if your strategy needs adjustment.

JUST COMPLETED TRADE:
Symbol:      {trade['symbol']}
Type:        {'PENNY' if trade['is_penny'] else 'REGULAR'}
Action:      {trade['action']}
Price:       ${trade['price']:.4f}
P&L:         {trade['profit_pct']:+.2f}%
Reason:      {trade['reason']}
RSI at buy:  {trade.get('rsi', 50):.0f}
Volume:      {trade.get('volume_ratio', 1.0):.1f}x average
News score:  {trade.get('news_score', 0.0):+.2f}

OVERALL PERFORMANCE:
Total trades: {total_trades}
Win rate:     {win_rate_display}  ← IGNORE if fewer than 5 trades
Total P&L:    ${stats.get('total_pnl', 0):+.2f}
Best trade:   +{stats.get('best_trade', 0)}%
Worst trade:  {stats.get('worst_trade', 0)}%

LAST 20 TRADES:
{trade_summary if trade_summary else 'No recent trades yet'}

CURRENT STRATEGY (LIVE FROM config.py — these ARE the actual live values, do NOT hallucinate different numbers):
{json.dumps(current_strategy, indent=2)}

Respond ONLY with this JSON:
{{
  "reflection": "<2-3 sentence analysis of this trade>",
  "adjust_strategy": true or false,
  "new_strategy": {{
    "penny": {{
      "rsi_threshold": <number>,
      "volume_spike": <number>,
      "stop_loss_pct": <number>,
      "take_profit_pct": <number>,
      "max_trade_size": <number>,
      "min_news_score": <number>
    }},
    "regular": {{
      "rsi_threshold": <number>,
      "volume_spike": <number>,
      "stop_loss_pct": <number>,
      "take_profit_pct": <number>,
      "max_trade_size": <number>,
      "min_news_score": <number>
    }}
  }},
  "adjustment_reasoning": "<what you changed and why, or 'No changes needed'>"
}}

Rules:
- Only adjust if clear pattern across multiple trades
- Make small incremental changes
- If fewer than 5 trades — do NOT adjust strategy yet
"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 1000,
            messages   = [{"role": "user", "content": prompt}]
        )
        raw  = response.content[0].text.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        print(f"🧠 Reflection: {data['reflection']}")
        save_reflection(data["reflection"], trigger="after_trade")

        _check_tier3_unlock()

        if (data["adjust_strategy"]
                and total_trades >= MIN_TRADES_FOR_AUTO_ADJUST
                and ENABLE_AUTO_STRATEGY_ADJUSTMENTS):
            save_strategy_adjustment(
                current_strategy.copy(),
                data["new_strategy"],
                data["adjustment_reasoning"]
            )
            print(f"⚙️  [TIER 3] Strategy adjusted: {data['adjustment_reasoning'][:100]}...")
            return data["new_strategy"]
        elif data["adjust_strategy"]:
            _write_suggestion(
                source            = "after_trade",
                trigger_symbol    = trade.get("symbol", ""),
                current_strategy  = current_strategy,
                proposed_strategy = data.get("new_strategy", {}),
                reasoning         = data["adjustment_reasoning"],
                reflection        = data.get("reflection", ""),
                clean_trade_count = _count_clean_trades(),
            )
            print(f"🔒 [TIER 2] Suggestion queued for manual review")
            print(f"   Reason: {data['adjustment_reasoning'][:100]}...")

        return current_strategy

    except Exception as e:
        print(f"⚠️  Reflection failed: {e}")
        return current_strategy


# ─────────────────────────────────────────────────────────────
# 2. DAILY END-OF-DAY REFLECTION
# ─────────────────────────────────────────────────────────────
def daily_reflection():
    """
    Deep end-of-day review.
    ✅ Uses LIVE config values (not stale memory).
    ✅ Filters churning trades from analysis input.
    ✅ Skips deep analysis entirely if sample too small.
    """
    print(f"\n🧠 Running daily reflection...")

    memory           = load_memory()
    todays_raw       = get_todays_trades()
    recent_raw       = get_recent_trades(days=30)
    current_strategy = _get_live_strategy_from_config()
    stats            = memory["stats"]

    todays_trades, today_churn = _filter_churning_trades(todays_raw)
    recent_trades, recent_churn = _filter_churning_trades(recent_raw)

    if today_churn["removed"] > 0:
        print(f"🧹 Today: filtered {today_churn['removed']} churning trades "
              f"({today_churn['churning_tickers']})")
    if recent_churn["removed"] > 0:
        print(f"🧹 30-day: filtered {recent_churn['removed']} churning trades")

    if not _sample_size_ok(len(recent_trades)):
        print(f"📊 Sample too small — skipping daily reflection "
              f"({len(recent_trades)} clean trades over 30 days, "
              f"need {MIN_TRADES_FOR_REFLECTION}+)")
        print(f"   Continuing to log trades — reflection will resume when enough data exists")
        return current_strategy

    if not todays_trades:
        print("📭 No trades today — skipping daily reflection")
        return current_strategy

    sells      = [t for t in todays_trades if t["action"] == "SELL"]
    today_wins = [t for t in sells if t["profit_pct"] >= 0]
    today_pnl  = sum(t.get("profit_dollars", 0) for t in sells)

    today_summary = "\n".join([
        f"- {t['time']} {t['action']} {t['symbol']} "
        f"P&L: {t.get('profit_pct', 0):+.2f}% | Reason: {t.get('reason', 'N/A')}"
        for t in todays_trades
    ])

    penny_30d   = [t for t in recent_trades
                   if t["action"] == "SELL" and t.get("is_penny")]
    regular_30d = [t for t in recent_trades
                   if t["action"] == "SELL" and not t.get("is_penny")]

    penny_wr   = round(len([t for t in penny_30d   if t["profit_pct"] >= 0])
                       / max(len(penny_30d), 1) * 100, 1)
    regular_wr = round(len([t for t in regular_30d if t["profit_pct"] >= 0])
                       / max(len(regular_30d), 1) * 100, 1)

    past_refs = "\n".join([
        f"- [{r['date']}] {r['reflection']}"
        for r in memory.get("reflections", [])[-5:]
    ])

    total_trades     = stats.get("total_trades", 0)
    win_rate_display = (
        f"{round(stats.get('total_wins', 0) / max(total_trades, 1) * 100, 1)}%"
        if total_trades >= 5
        else f"N/A — only {total_trades} trades"
    )

    prompt = f"""
You are an autonomous trading agent doing your end-of-day review.

TODAY ({datetime.now().strftime('%Y-%m-%d')}):
Trades: {len(todays_trades)} | Sells: {len(sells)} | Wins: {len(today_wins)}/{len(sells)} | P&L: ${today_pnl:+.2f}

TODAY'S TRADES:
{today_summary}

LAST 30 DAYS:
Penny:   {len(penny_30d)} trades | {penny_wr}% win rate
Regular: {len(regular_30d)} trades | {regular_wr}% win rate
All-time win rate: {win_rate_display}  ← IGNORE if fewer than 5 trades
All-time P&L: ${stats.get('total_pnl', 0):+.2f}

RECENT REFLECTIONS:
{past_refs if past_refs else 'None yet'}

CURRENT STRATEGY (LIVE FROM config.py — these ARE the actual live values, do NOT hallucinate different numbers):
{json.dumps(current_strategy, indent=2)}

Respond ONLY with this JSON:
{{
  "daily_reflection": "<3-5 sentences: what worked, what didn't, key patterns>",
  "market_observations": "<1-2 sentences about market conditions today>",
  "adjust_strategy": true or false,
  "new_strategy": {{
    "penny": {{
      "rsi_threshold": <number>,
      "volume_spike": <number>,
      "stop_loss_pct": <number>,
      "take_profit_pct": <number>,
      "max_trade_size": <number>,
      "min_news_score": <number>
    }},
    "regular": {{
      "rsi_threshold": <number>,
      "volume_spike": <number>,
      "stop_loss_pct": <number>,
      "take_profit_pct": <number>,
      "max_trade_size": <number>,
      "min_news_score": <number>
    }}
  }},
  "adjustment_reasoning": "<detailed explanation of every change>",
  "focus_for_tomorrow": "<1-2 sentences: what to watch tomorrow>"
}}
"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 1500,
            messages   = [{"role": "user", "content": prompt}]
        )
        raw  = response.content[0].text.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        print(f"\n{'='*60}")
        print(f"🧠 DAILY REFLECTION")
        print(f"{'='*60}")
        print(f"📊 {data['daily_reflection']}")
        print(f"📈 {data['market_observations']}")
        print(f"🔮 Tomorrow: {data['focus_for_tomorrow']}")

        pattern_insights = analyze_patterns()
        reflection_blob  = (data["daily_reflection"] + " | " +
                            data["focus_for_tomorrow"])
        if pattern_insights:
            reflection_blob += (" | Patterns: " +
                                pattern_insights.get("key_insight", ""))

        save_reflection(reflection_blob, trigger="daily")

        _check_tier3_unlock()

        if data["adjust_strategy"] and ENABLE_AUTO_STRATEGY_ADJUSTMENTS:
            save_strategy_adjustment(
                current_strategy.copy(),
                data["new_strategy"],
                data["adjustment_reasoning"]
            )
            print(f"\n⚙️  [TIER 3] Strategy updated: {data['adjustment_reasoning'][:100]}...")
            return data["new_strategy"]
        elif data["adjust_strategy"]:
            _write_suggestion(
                source            = "daily",
                trigger_symbol    = "END_OF_DAY",
                current_strategy  = current_strategy,
                proposed_strategy = data.get("new_strategy", {}),
                reasoning         = data["adjustment_reasoning"],
                reflection        = data.get("reflection", ""),
                clean_trade_count = _count_clean_trades(),
            )
            print(f"\n🔒 [TIER 2] Daily suggestion queued for manual review")
            print(f"   Reason: {data['adjustment_reasoning'][:150]}...")

        print(f"\n✅ Strategy unchanged for tomorrow")
        return current_strategy

    except Exception as e:
        print(f"⚠️  Daily reflection failed: {e}")
        return current_strategy


# ─────────────────────────────────────────────────────────────
# 3. GET LIVE STRATEGY
# ─────────────────────────────────────────────────────────────
def get_live_strategy() -> dict:
    """Returns current strategy adjusted for market regime."""
    base_strategy = get_current_strategy()
    regime_data   = get_market_regime()
    regime_name   = regime_data.get("regime", "NEUTRAL")
    adjustments   = get_regime_adjustments(regime_name)

    if regime_name == "NEUTRAL":
        return base_strategy

    adjusted = {
        "penny":   base_strategy["penny"].copy(),
        "regular": base_strategy["regular"].copy(),
    }

    rsi_adj  = adjustments.get("rsi_adjustment",  0)
    news_adj = adjustments.get("news_adjustment",  0.0)
    stop_adj = adjustments.get("stop_adjustment",  0.0)

    for cls in ["penny", "regular"]:
        adjusted[cls]["rsi_threshold"]  += rsi_adj
        adjusted[cls]["min_news_score"] += news_adj
        adjusted[cls]["stop_loss_pct"]   = max(
            0.01,
            adjusted[cls]["stop_loss_pct"] + stop_adj
        )

    return adjusted


# ─────────────────────────────────────────────────────────────
# 4. CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────────
def score_trade_confidence(symbol: str, indicators: dict,
                           sentiment: dict,
                           is_penny: bool = True) -> dict:
    """
    Claude rates a potential trade 1–10 before entry.

    ✅ Cost optimization: skip Claude call for obviously weak signals.
    If the local technicals + sentiment are all far from buyable,
    return a neutral score (5.0) without paying for Claude.
    """
    rsi          = float(indicators.get("rsi", 50.0))
    volume_ratio = float(indicators.get("volume_ratio", 1.0))
    momentum     = float(indicators.get("momentum", 0.0))
    news_score   = float(sentiment.get("score", 0.0) if sentiment else 0.0)

    weak_rsi    = rsi          < 45
    weak_volume = volume_ratio < 1.0
    weak_news   = news_score   < 0.3
    weak_mom    = momentum     < 0.1

    if weak_rsi and weak_volume and weak_news and weak_mom:
        return {
            "score":                 5.0,
            "recommendation":        "SKIP",
            "reasoning":             "Weak signals across the board — skipped Claude scoring to save cost",
            "reasoning_short":       "Pre-filter: weak signals",
            "risks":                 ["Multiple weak indicators"],
            "should_trade":          False,
            "trade_size_multiplier": 0.0,
            "regime":                "unknown",
            "pre_filtered":          True,
        }

    memory   = load_memory()
    patterns = get_winning_patterns(min_trades=5)
    regime   = get_market_regime()
    stats    = memory.get("stats", {})

    total_trades     = stats.get("total_trades", 0)
    win_rate_display = (
        f"{round(stats.get('total_wins', 0) / max(total_trades, 1) * 100, 1)}%"
        if total_trades >= 5
        else f"N/A — only {total_trades} trades so far (ignore performance stats)"
    )

    pattern_context = ""
    if patterns.get("enough_data"):
        w = patterns["winning"]
        l = patterns["losing"]
        pattern_context = f"""
WINNING TRADE PATTERNS ({w['count']} wins):
  Avg RSI: {w['avg_rsi']} | Avg Volume: {w['avg_volume_ratio']}x
  Avg News: {w['avg_news_score']} | Avg Profit: +{w['avg_profit']}%
  Best hours: {patterns['best_hours']}

LOSING TRADE PATTERNS ({l['count']} losses):
  Avg RSI: {l['avg_rsi']} | Avg Volume: {l['avg_volume_ratio']}x
  Avg News: {l['avg_news_score']} | Avg Loss: {l['avg_loss']}%
"""
    else:
        pattern_context = (f"Insufficient trade history for pattern analysis "
                           f"({patterns.get('message', '')}). "
                           f"Score based on technical and news signals only.")

    prompt = f"""
You are an autonomous trading agent scoring a potential trade.
Rate this trade 1–10 based on technical signals and news only.

PROPOSED TRADE:
Symbol:    {symbol}
Type:      {'PENNY' if is_penny else 'REGULAR'} stock
Price:     ${indicators.get('price', 0):.4f}
RSI:       {indicators.get('rsi', 0):.1f}
Volume:    {indicators.get('volume_ratio', 0):.2f}x average
Momentum:  {indicators.get('momentum', 0):.2f}%
News:      {sentiment.get('score', 0):+.2f} — {sentiment.get('reasoning', 'N/A')}

MARKET:
Regime:    {regime['regime']}
SPY:       {regime.get('spy_change', 0):+.2f}%

PERFORMANCE:
Win rate:  {win_rate_display}
← IGNORE win rate if sample < 5 trades — judge on signals alone
Total P&L: ${stats.get('total_pnl', 0):+.2f}

{pattern_context}

Score 1–10. Respond ONLY with JSON:
{{
  "score": <float 1.0-10.0>,
  "reasoning": "<2 sentences: why this score based on signals>",
  "key_strengths": ["<strength1>", "<strength2>"],
  "key_risks": ["<risk1>", "<risk2>"],
  "recommendation": "<STRONG BUY | BUY | WEAK BUY | SKIP>"
}}

Scoring guide:
9-10 = Perfect setup
7-8  = Strong signals
5-6  = Mixed but tradeable
3-4  = Weak signals
1-2  = Skip

IMPORTANT: A score of 4+ with any positive signal should be WEAK BUY not SKIP.
Only use SKIP for genuinely bad setups (score < 4).
"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 500,
            messages   = [{"role": "user", "content": prompt}]
        )
        raw  = response.content[0].text.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        score          = float(data["score"])
        reasoning      = data["reasoning"]
        recommendation = data["recommendation"].upper()

        regime_adj = get_regime_adjustments(regime["regime"])

        if score >= 8.5:
            size_mult = 1.5 * regime_adj["size_multiplier"]
        elif score >= 7.0:
            size_mult = 1.2 * regime_adj["size_multiplier"]
        elif score >= 5.5:
            size_mult = 1.0 * regime_adj["size_multiplier"]
        elif score >= 4.0:
            size_mult = 0.7 * regime_adj["size_multiplier"]
        else:
            size_mult = 0.0

        size_mult = round(min(size_mult, 2.0), 2)

        should_trade = score >= 4.0

        print(f"[{symbol}] 🎯 Confidence: {score:.1f}/10 | "
              f"{recommendation} | Size: {size_mult}x | {reasoning}")

        if data.get("key_risks"):
            print(f"[{symbol}]    Risks: {data['key_risks']}")

        from memory import save_confidence_score
        save_confidence_score(symbol, score, reasoning)

        return {
            "score":                score,
            "reasoning":            reasoning,
            "recommendation":       recommendation,
            "key_strengths":        data.get("key_strengths", []),
            "key_risks":            data.get("key_risks", []),
            "trade_size_multiplier":size_mult,
            "should_trade":         should_trade,
            "regime":               regime["regime"],
        }

    except Exception as e:
        print(f"[{symbol}] ⚠️  Confidence scoring failed: {e}")
        return {
            "score":                5.0,
            "reasoning":            "Scoring unavailable — using defaults",
            "recommendation":       "BUY",
            "trade_size_multiplier":1.0,
            "should_trade":         True,
            "regime":               regime.get("regime", "NEUTRAL"),
        }


# ─────────────────────────────────────────────────────────────
# 5. PATTERN ANALYZER
# ─────────────────────────────────────────────────────────────
def analyze_patterns():
    """Claude analyzes trade patterns after N+ trades."""
    patterns = get_winning_patterns(min_trades=10)

    if not patterns.get("enough_data"):
        print(f"📊 Pattern analysis: {patterns.get('message')}")
        return None

    print(f"\n📊 Running pattern analysis...")
    w = patterns["winning"]
    l = patterns["losing"]

    prompt = f"""
You are a trading performance analyst.
Give 3 specific actionable recommendations based on these patterns.

WINNING TRADES ({w['count']} trades, avg +{w['avg_profit']}%):
  RSI: avg {w['avg_rsi']} | Volume: avg {w['avg_volume_ratio']}x
  News: avg {w['avg_news_score']} | Avg Profit: +{w['avg_profit']}%
  Best hours: {patterns['best_hours']}

LOSING TRADES ({l['count']} trades, avg {l['avg_loss']}%):
  RSI: avg {l['avg_rsi']} | Volume: avg {l['avg_volume_ratio']}x
  News: avg {l['avg_news_score']} | Avg Loss: {l['avg_loss']}%

EXIT REASONS: {patterns['exit_reasons']}
Win rate: {patterns['win_rate']}%

Respond ONLY with JSON:
{{
  "key_insight": "<most important pattern>",
  "recommendations": ["<change 1>", "<change 2>", "<change 3>"],
  "warning": "<anything concerning>"
}}
"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 600,
            messages   = [{"role": "user", "content": prompt}]
        )
        raw  = response.content[0].text.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        print(f"\n💡 Key insight: {data['key_insight']}")
        for r in data.get("recommendations", []):
            print(f"   → {r}")
        if data.get("warning"):
            print(f"⚠️  Warning: {data['warning']}")

        return data

    except Exception as e:
        print(f"⚠️  Pattern analysis failed: {e}")
        return None
