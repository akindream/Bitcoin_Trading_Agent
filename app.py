# =========================
# 1. Imports and Session State Setup
# =========================
import os
import time
import pytz
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, date, time as dtime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

import logging
# Suppress Streamlit "missing ScriptRunContext" noisy warning (harmless in many setups).
# This targets the specific Streamlit logger that emits the message.
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)


# Your custom functions and classes
from llm_integrations import get_llm_suggestion
from config import TRADES_LOG_FILE

from notifications import send_telegram_message, send_email
from strategy_backup import (
    BinanceClient_simulated,
    get_binance_price_data,
    get_live_btc_price_binance,
    calculate_atr_binance,
    backtest_strategy, train_price_prediction_model,
    log_trade_to_csv, create_advanced_features,
    get_price_prompt_data_cached, clean_data,
    check_and_send_weekly_report, get_or_create_eventloop,
    check_and_predict_live_signal, execute_ml_trade_at_hour
)


# =========================
# 2. Page & Sidebar UI Elements
# =========================
st.set_page_config(page_title="BTCUSDT Live Trading Dashboard", layout="wide")

# =========================
# Load env & initialize client
# =========================
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Trading mode selector in sidebar
st.sidebar.subheader("Trading Mode")
mode = st.sidebar.radio("Select Mode", ["Simulated", "Live"], index=0, key="mode_radio")
st.session_state.trading_mode = mode
st.sidebar.write(f"🟢 Current Mode: {mode}")

# Call the new function before initializing the client
get_or_create_eventloop()

@st.cache_resource
def get_binance_simulated():
    """Initializes and returns the Binance Client."""
    if not API_KEY or not API_SECRET:
        st.error("Binance API Key or Secret not found in environment variables.")
        return None
    try:
        if st.session_state.trading_mode == "Simulated":
            client = BinanceClient_simulated(API_KEY, API_SECRET)
            return client
    except Exception as e:
        st.error(f"Failed to initialize Binance Client: {e}")
        return None

client_sim = get_binance_simulated()
client_live = None


# Fail fast if keys are missing
if not API_KEY or not API_SECRET:
    st.error("❌ Binance API keys not found. Add BINANCE_API_KEY and BINANCE_API_SECRET to your .env and restart.")
    st.stop()

# =========================
# Session state bootstrapping
# =========================
# All session state initializations are now here
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=["time", "action", "price", "qty", "usdt", "btc"])
if 'auto_trading_on' not in st.session_state:
    st.session_state.auto_trading_on = False
if 'last_auto_ts' not in st.session_state:
    st.session_state.last_auto_ts = 0.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'usd': 100.0, 'btc': 0.0, 'open_position': False, 'last_buy_price': None, 'entry_price': 0.0, 'trades': []}
if 'llm_suggestion' not in st.session_state:
    st.session_state.llm_suggestion = "Generating AI suggestion..."
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_btc_price' not in st.session_state:
    st.session_state.current_btc_price = None
if 'atr_value' not in st.session_state:
    st.session_state.atr_value = None
if 'price_model' not in st.session_state:
    st.session_state.price_model = None
if 'features_used' not in st.session_state:
    st.session_state.features_used = []
if 'btc_df_features' not in st.session_state:
    st.session_state.btc_df_features = pd.DataFrame()
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = None
if 'last_live_prediction_display' not in st.session_state:
    st.session_state.last_live_prediction_display = None
if 'last_predicted_candle_time' not in st.session_state:
    st.session_state.last_predicted_candle_time = None
if 'pending_ml_trade_action' not in st.session_state:
    st.session_state.pending_ml_trade_action = None
if 'last_executed_ml_trade_hour' not in st.session_state:
    st.session_state.last_executed_ml_trade_hour = None
if 'dca_amount_usd' not in st.session_state:
    st.session_state.dca_amount_usd = 100.0
if 'last_weekly_report_sent_date' not in st.session_state:
    st.session_state.last_weekly_report_sent_date = None
# --- Weekly Retraining ---
if 'last_retrain_time' not in st.session_state:
    st.session_state.last_retrain_time = datetime.now(pytz.utc)
if 'transaction_cost_rate' not in st.session_state:
    st.session_state.transaction_cost_rate = 0.001
if 'trade_type' not in st.session_state:
    st.session_state.trade_type = None


# App constants
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DEFAULT_LOOKBACK = "30 day"


threshold = 0.5
max_trade_usdt = 100.0
auto_every_time = 60
transaction_cost_rate = 0.001


# Auto-refresh for live price updates
st_autorefresh(interval=300_000, limit=None, key="btc_auto_refresh")
check_and_send_weekly_report()

# =========================
# Main UI
# =========================
st.markdown(
    "<h1 style='text-align: center;'>Automated 🟢 BTC/USDT Live Trading Bot Dashboard</h1>",
    unsafe_allow_html=True
)

if st.session_state.sim_running:
    st.markdown("""
        <div style="animation: flash 1s infinite; background-color: #28a745; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold; width: 100%;">🟢 BOT IS RUNNING</div>
        <style>@keyframes flash {0% {opacity: 1;} 50% {opacity: 0.4;} 100% {opacity: 1;}}</style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="background-color: #dc3545; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold; width: 100%;">🔴 BOT STOPPED</div>
    """, unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("Bot Controls")
dca_drop_percent = st.sidebar.slider("DCA Drop %", 0.1, 10.0, 3.0, 0.1)
dca_amount_usd = st.sidebar.number_input("DCA Amount", 100, 10000,100, 50)
atr_multiplier = st.sidebar.slider("ATR Multiplier", 0.5, 5.0, 1.5, 0.1)
take_profit_pct = st.sidebar.slider("Take Profit %", 0.1, 20.0, 5.0, 0.1)

if st.sidebar.button("Reset Live Portfolio"):
    st.session_state.portfolio['usd'] = dca_amount_usd 
    st.session_state.portfolio['btc'] = 0.0
    st.session_state.portfolio['open_position'] = False
    st.session_state.portfolio['entry_price'] = 0.0
    st.session_state.sim_running = False
    st.session_state.last_prediction_time = None
    st.session_state.last_weekly_report_sent_date = None
    st.rerun()

live_cols = st.columns(4)
current_live_price = get_live_btc_price_binance(client_sim)
if st.session_state.trading_mode == "Live" and client_live:
    usdt_bal = client_live.get_account_balance("USDT") or 0.0
    btc_bal = client_live.get_account_balance("BTC") or 0.0
else:
    usdt_bal= st.session_state.portfolio['usd']
    btc_bal = st.session_state.portfolio['btc']

with live_cols[0]:
    st.metric("Live Price", f"${current_live_price:.2f}" if current_live_price else "—")
with live_cols[1]:
    st.metric("USDT", f"{usdt_bal:,.2f}")
with live_cols[2]:
    st.metric("BTC", f"{btc_bal:,.6f}")
with live_cols[3]:
    st.metric("Auto‑Trading", "ON" if st.session_state.sim_running else "OFF")
st.divider()

# =========================
# 1. Data Acquisition & Pre-processing
# =========================
st.header("1. Data Acquisition & Pre-processing")
st.subheader("Backtest Data Range Selection")

today = datetime.now().date()
default_start_date = today - timedelta(days=7)

col_date, col_time = st.columns(2)
with col_date:
    backtest_start_date = st.date_input("Start Date for Backtest Data", default_start_date)
with col_time:
    backtest_start_time = st.time_input("Start Time for Backtest Data", datetime.min.time())

backtest_start_datetime = datetime.combine(backtest_start_date, backtest_start_time)
backtest_start_datetime_utc = pytz.utc.localize(backtest_start_datetime)
backtest_period_str = backtest_start_datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
st.info(f"Backtest data will start from: **{backtest_period_str}**")

if client_sim:
    with st.spinner("Fetching and processing historical data from exchanges..."):
        exchange_tickers_config = [("BINANCE", "BTCUSDT")]
        historical_data_multi_exchange = get_binance_price_data(
            binance_client=client_sim,
            symbol="BTCUSDT", # Use the ticker from your list
            interval="1h",
            start_str=backtest_period_str
        )

        btc_df_raw_binance = historical_data_multi_exchange

        if not btc_df_raw_binance.empty:
            st.subheader("Raw BTC/USDT Data from Binance")
            st.dataframe(btc_df_raw_binance.head(2))
            st.dataframe(btc_df_raw_binance.tail(2))

            original_close_prices = btc_df_raw_binance['Close'].copy()
            processed_df = btc_df_raw_binance.copy()
            cleaned_normalized_df = clean_data(processed_df)
            st.session_state.btc_df_features = create_advanced_features(cleaned_normalized_df).copy()

            if 'Close' in st.session_state.btc_df_features.columns:
                aligned_original_close = original_close_prices.reindex(st.session_state.btc_df_features.index)
                st.session_state.btc_df_features.loc[:,'Close'] = aligned_original_close
                st.session_state.btc_df_features.dropna(subset=['Close'], inplace=True)
            else:
                st.error("The 'Close' column is missing from the feature DataFrame before rectification!")

            if not st.session_state.btc_df_features.empty:
                st.subheader("Cleaned, Normalized & Engineered Features")
                st.dataframe(st.session_state.btc_df_features.head(2))
                st.dataframe(st.session_state.btc_df_features.tail(2))
                st.session_state.btc_df_features['target_movement'] = (st.session_state.btc_df_features['Close'].shift(-1) > st.session_state.btc_df_features['Close']).astype(int)
                st.session_state.btc_df_features.dropna(subset=['target_movement'], inplace=True)
                st.dataframe(st.session_state.btc_df_features.head(2)) # Corrected line
                st.dataframe(st.session_state.btc_df_features.tail(2))
                st.info("Target variable 'target_movement' added: 1 if next close > current close, 0 otherwise.")
                
else:           
    st.error("Binance client not initialized. Cannot fetch historical data.")
    st.session_state.btc_df_features = pd.DataFrame()

# =========================
# 2. Model Training & Backtesting
# =========================
st.header("2. Model Training & Backtesting")

if st.session_state.get('btc_df_features') is None or st.session_state.btc_df_features.empty:
    st.warning("Please acquire data in Step 1 to train the model.")
else:
    if st.button("Train Price Prediction Model"):
        with st.spinner("Training Random Forest model..."):
            price_model, features_used, classification_report_text  = train_price_prediction_model(
                st.session_state.btc_df_features
            )
            st.session_state.price_model = price_model
            st.session_state.features_used = features_used
            if price_model:
                st.success("Model trained successfully!")

                # Display the classification report here
                st.subheader("Model Classification Report")
                st.text(classification_report_text)
            else:
                st.error("Model training failed.")

    if st.session_state.get('price_model'):
        st.success("Price prediction model is loaded and ready.")
        st.subheader("Backtest Trading Strategy")
        if st.button("Run Backtest"):
            if st.session_state.btc_df_features.empty:
                st.warning("No feature data available for backtesting.")
            else:
                with st.spinner("Running backtest..."):
                    trades_df, final_value, portfolio_df = backtest_strategy(
                        historical_data_with_features=st.session_state.btc_df_features,
                        model=st.session_state.price_model,
                        features_used=st.session_state.features_used,
                        initial_cash=st.session_state.get('dca_amount_usd', 100.0),
                    )
                    if not trades_df.empty:
                        st.write("### Backtest Results")
    
                        # Define the formatting for each column
                        format_dict = {
                            'price': '{:.2f}' .format,
                            'btc_change': '{:.8f}' .format,
                            'cash_change': '{:.4f}' .format,
                            'current_cash': '{:.4f}'.format,
                            'current_btc': '{:.8f}'.format,
                            'Profit/Loss': '{:.4f}'.format,
                            'Portfolio_value': '{:.4f}'.format
                        }
    
                        # Apply the formatting and display the DataFrame
                        st.dataframe(trades_df)

                    st.session_state.backtest_trades_df = trades_df
                    st.session_state.backtest_final_value = final_value
                    st.session_state.backtest_portfolio_df = portfolio_df
                    st.session_state.backtest_initial_cash = st.session_state.get('dca_amount_usd', 100.0)
                    st.success("Backtest completed!")
                    

# =========================
# 3. Backtest Results & Visualizations
# =========================
st.header("3. Backtest Results & Visualizations")

if 'backtest_portfolio_df' in st.session_state and not st.session_state.backtest_portfolio_df.empty:
    portfolio_results_df = st.session_state.backtest_portfolio_df
    trades_results_df = st.session_state.backtest_trades_df
    final_portfolio_value = st.session_state.backtest_final_value
    initial_cash_backtest = st.session_state.backtest_initial_cash

    total_profit_loss = final_portfolio_value - initial_cash_backtest
    total_trades = len(trades_results_df) if not trades_results_df.empty else 0
    win_rate = 0.0
    if 'Profit/Loss' in trades_results_df.columns:
        winning_trades = trades_results_df[trades_results_df['Profit/Loss'] > 0]
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

    st.write(f"**Backtest Initial Cash:** ${initial_cash_backtest:,.2f}")
    st.write(f"**Final Portfolio Value:** ${final_portfolio_value:,.2f}")
    st.write(f"**Total Profit/Loss:** ${total_profit_loss:,.2f}")
    st.write(f"**Total Trades:** {total_trades}")
    st.write(f"**Win Rate:** {win_rate:.2f}%")

    st.subheader("Backtest Trade History")
    st.dataframe(trades_results_df, width='stretch')

    st.subheader("Backtest Trade-Level Details: Portfolio Value at Trades")
    if not trades_results_df.empty and 'time' in trades_results_df.columns and 'Portfolio_value' in trades_results_df.columns:
        fig_portfolio_at_trade = go.Figure()
        fig_portfolio_at_trade.add_trace(go.Scatter(x=trades_results_df['time'], y=trades_results_df['Portfolio_value'], mode='lines', name='Portfolio Value at Trade'))
        fig_portfolio_at_trade.update_layout(title='Portfolio Value at Trade Events', xaxis_title='Time', yaxis_title='Portfolio Value (USD)')
        st.plotly_chart(fig_portfolio_at_trade, width='stretch')

    st.subheader("Backtest Trade-Level Details: BTC Price at Trades")
    if not trades_results_df.empty and 'time' in trades_results_df.columns and 'price' in trades_results_df.columns:
        fig_btc_price_at_trade = go.Figure()
        fig_btc_price_at_trade.add_trace(go.Scatter(x=trades_results_df['time'], y=trades_results_df['price'], mode='lines', name='BTC Price at Trade', line=dict(color='orange')))
        fig_btc_price_at_trade.update_layout(title='BTC Price at Trade Events', xaxis_title='Time', yaxis_title='BTC Price ($)')
        st.plotly_chart(fig_btc_price_at_trade, width='stretch')
else:
    st.info("Run a backtest in Step 2 to see results.")
st.divider()


# =========================
# 6. Live Trading Panel
# =========================
st.header("Live Trading Panel")

prediction_display_placeholder = st.empty()

# --- Start/Stop Live Trading Buttons ---
col_live_control1, col_live_control2 = st.sidebar.columns(2)
with col_live_control1:
    if st.sidebar.button("Start Live Trading", disabled=st.session_state.sim_running):
        if not st.session_state.price_model or not st.session_state.features_used:
            st.error("Please train the ML model in Step 2 first.")
        else:
            st.session_state.sim_running = True
            st.session_state.start_time = datetime.now(pytz.utc)  # Ensure timezone awareness
            st.session_state.last_prediction_time = datetime.now(pytz.utc) # Initialize for the first run
            st.session_state.last_predicted_candle_time = None
            st.session_state.pending_ml_trade_action = None
            st.session_state.last_executed_ml_trade_hour = None
            st.session_state.last_retrain_time = datetime.now(pytz.utc)
            st.success("✅ Bot started. Checking ML signal immediately...")
            st.rerun()

with col_live_control2:
    if st.sidebar.button("Stop Live Trading", disabled=not st.session_state.sim_running):
        # Close open position if any
        current_live_price = get_live_btc_price_binance(client_sim)
        p = st.session_state.portfolio
        current_session_pnl = 0.0
        final_portfolio_value = p['usd']

        if p['open_position'] and current_live_price is not None:
            btc_sold = p['btc']
            revenue = (btc_sold * current_live_price) * (1 - st.session_state.transaction_cost_rate)
            realized_profit_loss = revenue - st.session_state.previous_amount_used
            
            trade = {
                'type': 'SELL (Manual Close)',
                'price': current_live_price,
                'btc_change': -btc_sold, # Log as a negative value for clarity
                'time': datetime.now(),
                'Profit/Loss': realized_profit_loss,
                'cash_change': revenue,
                'current_cash': p['usd'] + revenue,
                'current_btc': 0.0,
                'Portfolio_value': p['usd'] + revenue
            }

            p['trades'].append(trade)
            log_trade_to_csv(trade)
            p['usd'] += revenue
            p['btc'] = 0.0
            p['open_position'] = False
            p['last_buy_price'] = None
            p['entry_price'] = None
            p['previous_amount_used'] = 0.0
            final_portfolio_value = p['usd']
            st.success(f"⛔ Bot stopping. Position closed. Realized P/L: ${realized_profit_loss:,.2f}")
        else:
            st.info("⛔ Bot stopping. No open position to close.")

        # Save session result
        if 'results' not in st.session_state:
            st.session_state.results = []
        st.session_state.results.append({
            'start': st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.start_time else 'N/A',
            'stop': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pnl': current_session_pnl,
            'total_portfolio_value': final_portfolio_value
        })
        st.session_state.sim_running = False
        st.session_state.start_time = None
        st.session_state.last_prediction_time = None
        st.session_state.last_predicted_candle_time = None
        st.session_state.pending_ml_trade_action = None
        st.session_state.last_executed_ml_trade_hour = None
        st.rerun()

# --- Live Trading Loop ---
if st.session_state.sim_running:
    current_time_utc = datetime.now(pytz.utc)
    prediction_display_placeholder.write(f"[{current_time_utc.strftime('%H:%M:%S UTC')}] Bot is active. Monitoring signals...")
    
    # Retrain logic
    if (current_time_utc - st.session_state.last_retrain_time).days >= 7:
        st.success("🔄 Retraining ML model with last 30 days of data...")
        historical_df = get_binance_price_data(client_sim, "BTCUSDT", "1h", "30 days ago UTC")
        historical_df_clean = clean_data(historical_df)
        features_df = create_advanced_features(historical_df_clean)
        if not features_df.empty:
            model, features_used, report = train_price_prediction_model(features_df)
            st.session_state.price_model = model
            st.session_state.features_used = features_used
            st.session_state.last_retrain_time = datetime.now(pytz.utc)
            st.success("✅ Weekly retrain completed.")
            st.info(report)

    # --- Check ML Prediction ---
    pending_trade = check_and_predict_live_signal(
        model=st.session_state.price_model,
        features_used=st.session_state.features_used,
        binance_client_instance=client_sim,
        prediction_display_placeholder=prediction_display_placeholder
    )
    
    # --- Execute Trade if Pending ---
    execute_ml_trade_at_hour(client_sim, prediction_display_placeholder, pending_trade)

    # --- Update Live Portfolio Status ---
    current_live_price = get_live_btc_price_binance(client_sim)
    st.session_state.current_btc_price = current_live_price
    current_atr = calculate_atr_binance(client_sim)
    st.session_state.atr_value = current_atr
    display_price = f"${current_live_price:,.2f}" if current_live_price else "Fetching..."
    display_atr = f"{current_atr:.2f}" if current_atr else "Calculating..."
    portfolio_value = st.session_state.portfolio['usd'] + (st.session_state.portfolio['btc'] * current_live_price if current_live_price else 0)

    st.markdown(f"""
    - **Current USD Balance:** ${st.session_state.portfolio['usd']:,.2f}
    - **Current BTC Holdings:** {st.session_state.portfolio['btc']:,.6f} BTC
    - **Current BTC Price:** {display_price}
    - **Estimated Total Portfolio Value:** ${portfolio_value:,.2f}
    - **Calculated ATR Value:** {display_atr}
    """)

    if st.session_state.portfolio['open_position']:
        st.info(f"**Open Position Details:** Entry Price: ${st.session_state.portfolio['entry_price']:.2f}")
        if st.session_state.portfolio['entry_price'] and current_live_price:
            unrealized_pl = (current_live_price - st.session_state.portfolio['entry_price']) * st.session_state.portfolio['btc']
            st.write(f"Unrealized P/L: ${unrealized_pl:,.2f}")
else:
    st.session_state.last_live_prediction_display = None
    st.session_state.last_predicted_candle_time = None
    st.session_state.pending_ml_trade_action = None
    st.session_state.last_executed_ml_trade_hour = -1
    st.session_state.last_prediction_time = datetime.min.replace(tzinfo=pytz.utc)
    st.session_state.notification_sent_for_prediction_time = None
    prediction_display_placeholder.info("Start the bot to receive live trading signals. 🚀")

st.divider()


# =========================
# 7. Live Trading Metrics & Charts
# =========================

st.subheader("Live Trading Metrics & Charts")
live_trades_df = pd.DataFrame() # Initialize an empty dataframe

if os.path.exists(TRADES_LOG_FILE):
    live_trades_df = pd.read_csv(TRADES_LOG_FILE)
    if not live_trades_df.empty:
        # Clean and format the dataframe
        live_trades_df.columns = live_trades_df.columns.str.strip()
        
        # Ensure 'time' column is converted to datetime objects
        if 'time' in live_trades_df.columns:
            live_trades_df['time'] = pd.to_datetime(live_trades_df['time'])
            
        # Ensure numeric columns are in the correct format
        numeric_cols = ['price', 'btc_change', 'cash_change', 'current_cash', 'current_btc', 'Profit/Loss', 'Portfolio_value']
        
        for col in numeric_cols:
            if col in live_trades_df.columns:
                live_trades_df[col] = pd.to_numeric(live_trades_df[col], errors='coerce')
        
        # Sort by time in ascending order to ensure correct plotting
        live_trades_df = live_trades_df.sort_values(by='time', ascending=True)

    else:
        st.info("No live trades logged yet.")
else:
    st.info("No live trades log file found.")


if not live_trades_df.empty and 'Portfolio_value' in live_trades_df.columns:
    # Calculate key metrics, mirroring the backtest section
    initial_cash_live = st.session_state.dca_amount_usd
    final_portfolio_value = live_trades_df['Portfolio_value'].iloc[-1]
    total_profit_loss = final_portfolio_value - initial_cash_live
    total_trades = len(live_trades_df)
    
    winning_trades = 0
    win_rate = 0.0
    if 'Profit/Loss' in live_trades_df.columns:
        winning_trades = len(live_trades_df[live_trades_df['Profit/Loss'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    st.write(f"**Live Initial Cash (at first trade):** ${initial_cash_live:,.2f}")
    st.write(f"**Final Portfolio Value:** ${final_portfolio_value:,.2f}")
    st.write(f"**Total Profit/Loss:** ${total_profit_loss:,.2f}")
    st.write(f"**Total Trades:** {total_trades}")
    st.write(f"**Win Rate:** {win_rate:.2f}%")

    st.subheader("Live Trade History")
    st.dataframe(live_trades_df, width='stretch')

    st.subheader("Live Trading Details: Portfolio Value at Trades")
    if 'time' in live_trades_df.columns and 'Portfolio_value' in live_trades_df.columns:
        fig_portfolio_at_trade = go.Figure()
        fig_portfolio_at_trade.add_trace(go.Scatter(x=live_trades_df['time'], y=live_trades_df['Portfolio_value'], mode='lines', name='Portfolio Value at Trade'))
        fig_portfolio_at_trade.update_layout(title='Portfolio Value at Live Trade Events', xaxis_title='Time', yaxis_title='Portfolio Value (USD)')
        st.plotly_chart(fig_portfolio_at_trade, width='stretch')

    st.subheader("Live Trading Details: BTC Price at Trades")
    if 'time' in live_trades_df.columns and 'price' in live_trades_df.columns:
        fig_btc_price_at_trade = go.Figure()
        fig_btc_price_at_trade.add_trace(go.Scatter(x=live_trades_df['time'], y=live_trades_df['price'], mode='lines', name='BTC Price at Trade', line=dict(color='orange')))
        fig_btc_price_at_trade.update_layout(title='BTC Price at Live Trade Events', xaxis_title='Time', yaxis_title='BTC Price ($)')
        st.plotly_chart(fig_btc_price_at_trade, width='stretch')

# Note: The "Historical Session Summaries" section remains unchanged.
# It can be placed directly after this block.

st.subheader("Historical Session Summaries")
if st.session_state.results:
    results_df = pd.DataFrame(st.session_state.results)
    results_df['Cumulative Session P/L'] = results_df['pnl'].cumsum()
    results_df = results_df.rename(columns={'pnl': 'Session P/L', 'total_portfolio_value': 'Ending Portfolio Value'})
    st.dataframe(results_df)
    csv_results = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Session Results CSV", data=csv_results, file_name="trade_sessions_results.csv", mime="text/csv")
    st.subheader("Performance Charts")
    col1, col2 = st.columns(2)
    with col1:
        fig_cum_pnl, ax_cum_pnl = plt.subplots(figsize=(8, 4))
        ax_cum_pnl.plot(results_df['stop'], results_df['Cumulative Session P/L'], marker='o', linestyle='-', color='blue')
        ax_cum_pnl.set_title("Cumulative Session Profit/Loss Over Time")
        ax_cum_pnl.set_xlabel("Session Stop Time")
        ax_cum_pnl.set_ylabel("Cumulative P/L ($)")
        st.pyplot(fig_cum_pnl)
    with col2:
        fig_total_value, ax_total_value = plt.subplots(figsize=(8, 4))
        ax_total_value.plot(results_df['stop'], results_df['Ending Portfolio Value'], marker='s', linestyle='-', color='orange')
        ax_total_value.set_title("Total Portfolio Value Over Time (End of Session)")
        ax_total_value.set_xlabel("Session Stop Time")
        ax_total_value.set_ylabel("Total Value ($)")
        st.pyplot(fig_total_value)
else:
    st.info("No completed trade sessions to summarize yet. Stop the bot to record a session.")

# =========================
# 8. Manual Controls
# =========================
st.sidebar.header("Manual Notifications")
manual_msg = st.sidebar.text_area("Enter notification message")
if st.sidebar.button("Send Telegram Notification"):
    if manual_msg.strip():
        send_telegram_message(manual_msg.strip())
        st.sidebar.success("Telegram message sent!")
    else:
        st.sidebar.error("Please enter a message.")
if st.sidebar.button("Send Email Notification"):
    if manual_msg.strip():
        send_email("Manual Notification", manual_msg.strip())
        st.sidebar.success("Email sent!")
    else:
        st.sidebar.error("Please enter a message.")
st.divider()

# =========================
# 9. LLM Suggestion Section
# =========================
with st.expander("AI Trading Suggestion"):
    if st.button("Get AI Suggestion"):
        if client_sim and OPENAI_API_KEY:
            with st.spinner("Thinking..."):
                price_data_for_llm = get_price_prompt_data_cached(client_sim)
                if "No data available" in price_data_for_llm or "client not initialized" in price_data_for_llm:
                    st.error(f"Could not get historical price data: {price_data_for_llm}")
                    st.session_state.llm_suggestion = "Could not generate AI suggestion due to data error."
                else:
                    st.session_state.llm_suggestion = get_llm_suggestion(price_data_for_llm, st.session_state.portfolio)
            st.info(st.session_state.llm_suggestion)
        elif not OPENAI_API_KEY:
            st.warning("OpenAI API key not found. Cannot get AI suggestion.")
        else:
            st.error("Binance client not initialized. Cannot get AI suggestion.")

time.sleep(120)
st.rerun()
