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


# Project Summary
This project implements an autonomous Bitcoin trading system designed to run 24/7 with minimal human intervention. 
It combines Dollar-Cost Averaging (DCA) with ATR-based risk management and optional LLM-driven strategy selection to dynamically adapt to 
changing market conditions. The system supports configurable budgets, multiple trading strategies, and real-time market data from major 
crypto source, Binance. Configuration is managed via csv file with secure credential handling, while trades are monitored through Telegram 
alerts and weekly Gmail reports. The solution is built for cloud deployment, emphasizing automation, security and performance evaluation 
through backtesting and live simulation. Trading with profit was achieved while taking into cognizance trading charges.
