# Personal Agentic AI Trading Agent

A sophisticated autonomous trading agent powered by Claude AI that provides intelligent trade analysis, strategy optimization, and market reflection.

## Overview

This agent implements a **three-tier learning system** for autonomous trading strategy management:

- **Tier 1**: Trade logging and data collection
- **Tier 2**: Manual review suggestions system (default mode)
- **Tier 3**: Automatic strategy adjustments (unlocks after 50+ clean trades)

## Key Features

### 🧠 Intelligent Reflection System
- **After-Trade Reflection**: Claude analyzes each completed trade and suggests strategy improvements
- **Daily End-of-Day Review**: Comprehensive analysis of daily performance with market observations
- **Pattern Analysis**: Identifies winning patterns and trading behaviors across historical data

### 🛡️ Data Quality Safeguards
- **Churn Filter**: Automatically detects and removes rapid re-entries and panic trades
- **Sample Size Validation**: Skips reflection until sufficient clean trade data exists
- **Config Sync**: Reads live strategy values directly from configuration (prevents hallucination)

### 🎯 Confidence Scoring
- Pre-trade assessment system (1-10 scale)
- Technical signal validation before Claude processing
- Cost optimization via pre-filtering weak signals
- Regime-aware position sizing

### 📊 Hybrid Learning Architecture
- **Tier 2 (Manual)**: Suggestions written to `suggestions.json` for human review
- **Tier 3 (Automatic)**: Auto-unlock after accumulating enough validated trades
- Clear audit trail of all suggested adjustments

## Configuration

### Core Parameters

| Parameter | Purpose |
|-----------|---------|
| `ENABLE_AUTO_STRATEGY_ADJUSTMENTS` | Toggle auto-mode (default: False) |
| `MIN_TRADES_FOR_AUTO_ADJUST` | Trades required before auto-adjustments (default: 50) |
| `MIN_TRADES_FOR_REFLECTION` | Minimum clean trades for reflection analysis (default: 30) |
| `CHURN_WINDOW_MINUTES` | Rapid re-entry detection window (default: 60 min) |
| `CHURN_MAX_REENTRIES_PER_DAY` | Max same-day re-entries before flagging as churn (default: 2) |
| `TIER3_UNLOCK_CLEAN_TRADES` | Threshold for automatic Tier 3 unlock (default: 50) |

## Strategy Parameters

The agent manages two strategy profiles:

### Penny Stock Profile
- RSI threshold for entry
- Volume spike requirement
- Stop-loss and take-profit percentages
- Maximum trade size
- Minimum news sentiment score

### Regular Stock Profile
- Same parameters optimized for regular equities

## Usage

### Core Functions

```python
# After each completed trade
reflect_after_trade(trade: dict)

# End-of-day analysis
daily_reflection()

# Score a potential entry
score_trade_confidence(symbol, indicators, sentiment, is_penny)

# Get current strategy with market regime adjustments
get_live_strategy() -> dict

# Analyze historical patterns
analyze_patterns()
```

### Input Data Structure (Trade)

```json
{
  "symbol": "SYMBOL",
  "date": "YYYY-MM-DD",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "action": "BUY or SELL",
  "price": 0.0000,
  "profit_pct": 0.00,
  "profit_dollars": 0.00,
  "is_penny": true/false,
  "reason": "Entry reason",
  "rsi": 50,
  "volume_ratio": 1.5,
  "news_score": 0.75
}
```

## Data Files

- **agent_memory.json**: Full trade history and memory state
- **suggestions.json**: Pending/approved/rejected strategy adjustments for manual review
- **config.py**: Live strategy parameters (source of truth)

## Tier Progression

### Tier 1 → Tier 2
- Default starting mode
- All Claude suggestions logged to `suggestions.json`
- User manually reviews and approves changes

### Tier 2 → Tier 3
- Automatic unlock when `_count_clean_trades()` ≥ 50
- Console banner notification upon unlock
- Can be manually re-locked by setting `ENABLE_AUTO_STRATEGY_ADJUSTMENTS = False`

## Safety Features

✅ **Validation Guardrails**
- Stop-loss enforcement: 0.01 ≤ stop_loss_pct ≤ 0.15
- Take-profit bounds: 0.03 ≤ take_profit_pct ≤ 0.50
- Max position: max_trade_size ≤ 1000
- Small incremental adjustments only

✅ **Decision Thresholds**
- Minimum 5 trades before strategy adjustments
- Minimum 30 clean trades before reflection analysis
- Churn detection prevents cascade failures

✅ **Cost Optimization**
- Pre-filters weak signals to avoid unnecessary Claude calls
- Scores skipped for obvious non-candidates
- Regime-based position sizing

## Example Workflow

1. **Trade Completed** → `reflect_after_trade()` called
2. **Analysis** → Claude reviews with live config + filtered history
3. **Suggestion** → If adjustment warranted, logged to `suggestions.json`
4. **Review** → User can inspect: 
   ```python
   python3 -c "import json; [print(s) for s in json.load(open('suggestions.json')) if s['status']=='pending']"
   ```
5. **After 50+ Trades** → Tier 3 unlocks, auto-adjustments activate
6. **Daily Recap** → `daily_reflection()` provides market insights

## Integration Points

- **Claude API**: Sonnet 4.5 model for analysis
- **Memory Module**: Persistent trade and strategy storage
- **Regime Module**: Market regime detection and adjustments
- **Config Module**: Live strategy parameters

## Notes

- All numbers are sanitized for privacy; use with actual trading parameters
- Suitable for both penny stocks and regular equities
- Market regime adjustments applied automatically to base strategy
- Pattern analysis requires minimum trade history for reliability

---

**Status**: Production-ready with manual approval gate (Tier 2)

**Next**: Unlock Tier 3 by logging 50+ validated trades without churn
