# Bitcoin_Trading_Agent
# PROJECT: wntnnTpcTZ8XX1TR

Background:


We are a fintech company focused on cryptocurrency trading building a smart bitcoin trading system designed to operate with minimal human supervision and continuously adapt to changing market conditions. The agent must dynamically manage budget allocation, shift between strategies, and make autonomous trading decisions while running 24/7.


This project will give you experience with real-time algorithmic trading, feature engineering in a volatile domain, LLM-assisted decision-making, and deploying robust, cloud-based AI systems that bridge automation with finance.



Project Objectives:


Accept a configurable budget (e.g., $1K or $100K)


Use Dollar-Cost Averaging (DCA) to accumulate more bitcoin when prices drop, distributing buys over time or price levels


Implement an ATR-based stop-loss strategy to manage short-term trades and avoid excessive loss exposure


Switch between different strategies (e.g., day trading, swing trading, value investing)


Adapt continuously to market conditions, ideally with the help of a lightweight LLM


Run 24/7 and deploy in a cloud environment


Send Telegram notifications for each trade made


Send a weekly email report every Monday at 9:00AM via Gmail




Key Strategy Concepts:


Dollar-Cost Averaging (DCA): A long-term investment strategy where you buy small amounts at regular intervals or when the price drops by a defined percentage, helping reduce the impact of market volatility.


Average True Range (ATR): A technical indicator that measures market volatility. In this project, ATR will be used to define dynamic stop-loss thresholds that adapt to current conditions and reduce false triggers.



Hybrid Strategy Expectation:


DCA is your base layer: Buy when price drops by a set percentage or at time-based intervals.


ATR-based stop-loss is used for active trades: For non-DCA entries (e.g., swing trades), use Stop = Entry Price - k × ATR, where k is configurable.


Optional: Use ATR to trigger opportunistic DCA: Add additional buys when price drops sharply relative to recent volatility.


Global Portfolio-Level Safeguard: You may implement a system-wide stop condition (e.g., pause all activity if total portfolio value drops 25%).


All thresholds, flags, and strategy logic must be parameterized and configurable by the user.



Configuration Management:


Application Configs (e.g., DCA %, ATR multiplier, strategy mode, data fetching interval, toggles): Store in a Google Sheet, read hourly, with a local cached fallback as JSON file.


Sensitive Credentials (API keys, Coinbase secrets, Gmail credentials, etc.): Store in a secure local .env file and never expose in the sheet or public code.




Trade Monitoring & Reporting:


Telegram Bot: Notify on each trade (buy/sell) with key details


Gmail: Weekly summary every Monday at 9:00 AM




Data Sources:


Investing.com: Bitcoin Historical Data (scraping needed)


CoinMarketCap API


Binance API


Yahoo Finance via yfinance


Coinbase Advanced Trade API




Feature Engineering:


Residents are encouraged to:

Use standard technical indicators (RSI, MACD, SMA, EMA)


Explore Bitcoin-specific features (block size, hash rate, mempool size)


Engineer custom features


Use an LLM module to generate, adapt, or select features dynamically based on context




Resources:

Investopedia: Technical Indicators


TradingView Scripts


CryptoQuant for on-chain BTC metrics




Security & Configuration Best Practices:


Pull application config from a Google Sheet (hourly)


Cache locally for fallback scenarios


Limit access using a secure service account


Store sensitive credentials only in a local JSON file


Load via secure method (e.g., dotenv or secrets manager)


Never commit secrets to source control


Log all trades and exceptions for traceability




Evaluation Criteria:


Portfolio Growth (BTC/USD): Performance over backtesting and simulation


Strategy Integration: Effective DCA + ATR hybrid logic


LLM Usage: Justified and creative application of LLMs


System Simplicity: Clear, well-structured, minimal architecture


Monitoring & Alerts: Functional email and Telegram notifications


Deployment: Dockerized, runnable locally, and deployable to DigitalOcean or AWS




Bonus Goals:


Streamlit/Flask/FastAPI dashboard for live portfolio tracking


Risk metrics (Sharpe ratio, Max Drawdown)


Plug-and-play modular strategy design


Edge-case simulation: e.g., "What if BTC drops 30% tomorrow?"





Use Case Scenario: How the System Works


Configuration:

Budget: $10,000

DCA Trigger: Buy $500 if BTC drops 3%

ATR Stop-Loss Multiplier: 1.5

Trading Mode: Hybrid (DCA + opportunistic)

Report Day: Monday at 9:00AM



Step-by-Step Flow:



Startup:

Loads credentials from config.json

Pulls strategy flags from Google Sheet (DCA %, ATR multiplier, toggles)




Market Monitoring:

Every 30 min, fetches BTC price and historical candles

Computes ATR(14)




DCA Trigger:

BTC drops by 3.2% since last buy

Agent buys $500 worth of BTC

Telegram alert sent:

"BTC dropped 3.2%. DCA Buy: $500 at $61,200. Portfolio: 0.32 BTC"




Opportunistic Swing Trade (LLM suggestion):

LLM detects volume breakout

BTC at $62,500 | ATR = $1,000 → Stop-loss set at $61,000

Trade executed and tracked

Telegram alert sent




Stop-Loss Triggered:

Price drops to $60,900 → Stop-loss hit

Trade closed automatically

Telegram alert:

"Stop-loss hit. Sold at $60,900. Loss: -2.56%"




Weekly Report Sent (Monday 9:00 AM):

Gmail summary includes:

Portfolio: 0.46 BTC, $28,600

P/L this week: +2.7%, +$328

Trades: 7 (2 DCA, 2 swing, 3 LLM-suggested)

Fees: $9.26

Strategy insight: LLM suggests tighter stops next week






Final Deliverables:


Make sure the code is well commented and clean


Inside the repository provide a report explaining the setup and how the system works


A short video walkthrough about the project highlighting the strengths of your solution and how it works, which is publicly accessible on YouTube


1 week evaluation log of your system tested with real data

Here is a **complete, professional README** you can use directly in your GitHub repository (you can copy-paste this as your `README.md`):

---

# Autonomous Bitcoin Trading Bot

## Overview

This project is a fully automated Bitcoin trading system designed to operate 24/7 with minimal human supervision. It dynamically manages capital allocation, adapts trading strategies based on market conditions, and executes trades in real time. The system integrates Dollar-Cost Averaging (DCA), ATR-based risk management, and optional LLM-assisted strategy decisions to create a flexible, data-driven trading agent suitable for high-volatility crypto markets.

---

## Key Features

* ✅ Automated trading and strategy execution
* ✅ Hybrid strategy: DCA + opportunistic swing trading
* ✅ ATR-based dynamic stop-loss system
* ✅ Fully configurable via Google Sheets
* ✅ Secure credential management via `.env`
* ✅ Telegram trade notifications
* ✅ Weekly Gmail performance reports
* ✅ Cloud-ready deployment (AWS / DigitalOcean / Docker)
* ✅ LLM-assisted market interpretation (optional)
* ✅ Backtesting and simulation support
* ✅ Modular architecture for strategy expansion
* ✅ Live portfolio dashboard (optional)

---

## Strategy Design

### 1. Dollar-Cost Averaging (DCA)

The bot purchases Bitcoin when price drops by a configured percentage or at fixed time intervals, helping reduce market timing risk.

### 2. ATR-Based Stop-Loss

For active (non-DCA) trades, stop-loss levels dynamically adjust using:

```
Stop = Entry Price – (k × ATR)
```

This allows risk controls to adapt to volatility.

### 3. Hybrid Strategy Engine

The system can:

* Switch between strategies (DCA, swing, day-trading)
* React to market conditions dynamically
* Pause trading during heavy drawdowns
* Execute LLM-inspired signals when enabled

---

## System Architecture

```
config/
├── config.json (cached config from Google Sheets)
env/
├── .env       (API keys and secrets)
bot_core.py    (strategy logic)
app.py         (API server & bot runner)
dashboard/     (Flask / Streamlit UI)
logs/          (trade history & errors)
```

---

## Configuration Management

| Item               | Source        |
| ------------------ | ------------- |
| Trading Parameters | Google Sheets |
| Secrets & API Keys | `.env`        |
| Local Fallback     | `config.json` |

Configs refresh hourly and automatically fail over to the local cache if unavailable.

---

## Trade Monitoring

### Telegram Alerts (Real-time)

Each trade sends:

* Trade type (BUY / SELL)
* BTC price
* Amount
* Portfolio balance
* Profit or loss

### Weekly Email Report (Every Monday 9:00AM)

Summary includes:

* Portfolio value
* Profit / Loss
* Number of trades
* Fees
* Strategy insights

---

## Supported Market Data Sources

* Binance API

---

## Feature Engineering

The system computes:

* RSI, MACD, EMA, SMA
* ATR for volatility control
* Price momentum and trend features
* Optional Bitcoin on-chain metrics
* Optional LLM-driven feature selection

---

## Deployment

Supports:

* ✅ Docker
* ✅ AWS
* ✅ Local execution


---

## Evaluation Metrics

* Portfolio growth
* Sharpe ratio
* Maximum drawdown
* Trade accuracy
* Profit factor
* Strategy stability

---

## Security

* API keys are NEVER stored in source code
* `.env`and csv files is excluded from GitHub

---

### Configuration:

```
Budget: $10,000
DCA Trigger: 3%
ATR Multiplier: 1.5
Mode: Hybrid
Weekly Report: Monday 9:00AM
```

### Sample Flow:

1. Bot buys or sells by model/LLM inputs
2. Swing trade opened with ATR stop
3. Stop-loss hit → Auto exit
4. Telegram alert sent
5. Weekly performance email delivered

---

## Bonus Implementations

* 📊 Dashboard UI
* 📉 Drawdown control
* 🧠 LLM-assisted optimization
* 🔄 Strategy plug-ins
* 🧪 Simulated market stress testing

---

## Final Deliverables

* ✅ Clean and commented code
* ✅ System documentation


---


📫 Contact

📧 Medium: Ernest Braimoh https://medium.com/@akindream/building-a-fully-autonomous-bitcoin-trading-bot-with-ai-and-real-time-risk-management-0b1e450235af

🔗 LinkedIn: Ernest Braimoh https://www.linkedin.com/posts/ernest-braimoh_building-a-fully-autonomous-bitcoin-trading-activity-7400511515298050048-SEwf?utm_source=share&utm_medium=member_desktop&rcm=ACoAACJ5f84BSF16YQBlNnzy86sMhIc99PdU8l0

🔗 Youtube: Ernest Braimoh
