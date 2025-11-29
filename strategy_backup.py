#===================================================================== --- Dependencies --- =========================================================================

import pandas as pd
from datetime import datetime, timedelta, timezone
from binance.client import Client as BinanceRESTClient
from binance.enums import * # Import all enums for order types, etc.
import ta # Import the ta library
from sklearn.preprocessing import MinMaxScaler # Needed for clean_and_normalize_data
from sklearn.linear_model import LogisticRegression # Needed for train_price_prediction_model
import os
import csv
import ta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import pytz
from datetime import datetime, timedelta
from notifications import send_telegram_message, send_email
import logging
import math
from sklearn.metrics import classification_report
if "transaction_cost_rate" not in st.session_state:
    st.session_state.transaction_cost_rate = 0.001

logger = logging.getLogger(__name__)

TRADES_LOG_FILE = "trades_log.csv"
CSV_FIELDNAMES = ["time", "type", "price", "btc_change", "cash_change", "current_cash", "current_btc", "Profit/Loss", "Portfolio_value"]

# Dummy functions for notifications and logging, replace with actual implementations
# In a real application, you'd integrate with Telegram bot API, email services, and a robust logging system.

#=============================================================== --- Binance Client Class --- =========================================================================
import asyncio

def get_or_create_eventloop():
    """Creates a new event loop if one doesn't exist."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
        
class BinanceClient_simulated:
    def __init__(self, api_key, api_secret):
        if not api_key or not api_secret:
            raise ValueError("Binance API Key or Secret is missing for BinanceClient initialization.")
        self.rest_client = BinanceRESTClient(api_key, api_secret)
        print("DEBUG (strategy.py): BinanceRESTClient instance created.")

    def get_symbol_ticker(self, symbol):
        """Fetches the current symbol price."""
        try:
            return self.rest_client.get_symbol_ticker(symbol=symbol)
        except Exception as e:
            print(f"ERROR (BinanceClient): Error getting symbol ticker for {symbol}: {e}")
            return None

    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """Fetches historical klines."""
        try:
            return self.rest_client.get_historical_klines(symbol, interval, start_str, end_str)
        except Exception as e:
            print(f"ERROR (BinanceClient): Error getting historical klines for {symbol}: {e}")
            return None

    def get_account_balance(self, asset):
        """Fetches the free balance for a specific asset from Binance."""
        try:
            balance = self.rest_client.get_asset_balance(asset=asset)
            free_balance = float(balance['free'])
            print(f"DEBUG (BinanceClient): Fetched {asset} balance: {free_balance}")
            return free_balance
        except Exception as e:
            print(f"ERROR (BinanceClient): Error getting balance for {asset}: {e}")
            return None

            
    def get_exchange_info(self, symbol):
        """
        Fetches exchange information for a symbol, useful for getting minNotional, stepSize, etc.
        """
        try:
            info = self.rest_client.get_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    return s
            return None
        except Exception as e:
            print(f"ERROR (BinanceClient): Error fetching exchange info for {symbol}: {e}")
            return None

    def get_min_notional(self, symbol):
        """Get minimum notional value for a symbol."""
        info = self.get_exchange_info(symbol)
        if info:
            for f in info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    return float(f['minNotional'])
        return 0.0

    def get_step_size(self, symbol):
        """Get the step size (precision) for a symbol's quantity."""
        info = self.get_exchange_info(symbol)
        if info:
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return float(f['stepSize'])
        return 0.0

    def adjust_quantity_to_precision(self, symbol, quantity):
        """Adjusts quantity to the symbol's step size."""
        step_size = self.get_step_size(symbol)
        if step_size > 0:
            # Calculate number of decimal places for step_size
            step_decimals = max(0, int(-math.log10(step_size)) if step_size != 0 else 0)
            adjusted_quantity = math.floor(quantity / step_size) * step_size
            return round(adjusted_quantity, step_decimals)
        return quantity


#================================================================ --- Data Acquisition Functions --- ======================================================================
def get_historical_klines_binance(binance_client_instance, symbol, interval, start_str):
    try:
        klines = binance_client_instance.get_historical_klines(symbol, interval, start_str)
        if not klines: # Check if klines list is empty
            print(f"DEBUG (strategy.py): No klines returned for {symbol} from Binance for period {start_str}.")
            return pd.DataFrame()
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
            'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
            'Taker_buy_quote_asset_volume', 'Ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        print(f"DEBUG (strategy.py): Successfully fetched {len(df)} klines from Binance for {symbol}.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"ERROR (strategy.py): Error fetching historical klines from Binance: {e}")
        return pd.DataFrame()

def get_live_btc_price_binance(binance_client_instance):
    """Fetches the current live BTCUSDT price from Binance."""
    if not binance_client_instance:
        print("WARNING (strategy.py): Binance client instance not provided for live price fetch.")
        return None
    try:
        ticker = binance_client_instance.get_symbol_ticker(symbol='BTCUSDT')
        if ticker:
            price = float(ticker['price'])
            print(f"DEBUG (strategy.py): Fetched live BTC price from Binance: {price}")
            return price
        else:
            print("ERROR (strategy.py): No ticker data returned from Binance for BTCUSDT.")
            return None
    except Exception as e:
        print(f"ERROR (strategy.py): Error fetching live BTC price from Binance: {e}")
        return None


def calculate_atr_binance(binance_client_instance, symbol='BTCUSDT', interval='1h', period='30 days ago UTC'):
    df = get_historical_klines_binance(binance_client_instance, symbol, interval, period)
    if df.empty:
        print("WARNING (strategy.py): No data to calculate ATR for Binance.")
        return None
    try:
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift())
        df['low_close'] = abs(df['Low'] - df['Close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr = df['true_range'].rolling(window=14).mean().iloc[-1]
        print(f"DEBUG (strategy.py): Calculated ATR for {symbol}: {atr}")
        return atr
    except Exception as e:
        print(f"ERROR (strategy.py): Error calculating ATR for Binance: {e}")
        return None


def get_binance_price_data(binance_client, symbol, interval, start_str):
    """
    Fetches historical price data from Binance.
    Args:
        binance_client: Initialized Binance client instance.
        symbol (str): Trading pair, e.g., "BTCUSDT".
        interval (str): Candlestick interval, e.g., "1h".
        start_str (str): Specific start time string, e.g., "30 days ago UTC".
    Returns:
        pd.DataFrame: A DataFrame of historical data or an empty DataFrame if no data is found.
    """
    if not binance_client:
        print(f"WARNING: Binance client not provided for {symbol} data fetch.")
        return pd.DataFrame()
    
    df = get_historical_klines_binance(binance_client, symbol, interval, start_str)
    if df.empty:
        print(f"DEBUG: No historical data fetched for {symbol}.")
    return df

def clean_data(df):
    """Fills missing values."""
    if df.empty:
        return pd.DataFrame()
    return df.ffill().bfill().dropna()

def create_advanced_features(df, atr_window=14):
    """Generates advanced features including technical indicators."""
    if df.empty or not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        print("Missing required columns for feature engineering.")
        return pd.DataFrame()
        
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=atr_window
    )
    df['ATR'] = atr_indicator.average_true_range()
    df['Price_Change_ATR_Ratio'] = (df['Close'].diff() / df['ATR']).fillna(0)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    
    return df.dropna()

# --- ML Model Training ---
def train_price_prediction_model(features_df, target_col='target_movement', test_size=0.2):
    """Trains a classification model to predict price movement (up/down)."""
    if features_df.empty or target_col not in features_df.columns:
        print(f"DataFrame empty or '{target_col}' not found for training.")
        return None, None, None

    cols_to_exclude = ['Open', 'High', 'Low', 'Close', 'Volume', target_col]
    feature_cols = [col for col in features_df.columns if col not in cols_to_exclude]

    X = features_df[feature_cols].copy()
    y = features_df[target_col].copy()

    X = X.fillna(X.mean(numeric_only=True))
    y = y.fillna(y.mode()[0])

    if X.empty or y.empty:
        print("Features or target became empty after NaN handling.")
        return None, None, None
    
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        print("Not enough data for train/test split after NaN handling.")
        return None, None, None
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Generate the report and store it in a variable
    report_text = classification_report(y_test, y_pred)
    
    # Return the model, features, and the report text
    return model, feature_cols, report_text

def backtest_strategy(historical_data_with_features, model, features_used, initial_cash, transaction_cost_rate=0.001):
    """
    Simulates trades over historical data using a trained model.
    """
    if historical_data_with_features.empty or not model:
        logger.warning("No data or model provided for backtesting.")
        return pd.DataFrame(), initial_cash, pd.DataFrame()

    cash = initial_cash
    btc_holdings = 0.0
    previous_amount_used = 0.0
    trades_log = []
    portfolio_history = []

    for i in range(len(historical_data_with_features)):
        current_data = historical_data_with_features.iloc[i]
        current_price = current_data['Close']
        features_for_prediction = current_data[features_used].fillna(historical_data_with_features[features_used].mean(numeric_only=True)).to_frame().T
        
        current_portfolio_value = cash + (btc_holdings * current_price)
        portfolio_history.append({'timestamp': historical_data_with_features.index[i], 'value': current_portfolio_value})

        if not features_for_prediction.empty:
            predicted_action = model.predict(features_for_prediction)[0]
            realized_profit_loss = 0.0

            if predicted_action == 1:  # BUY signal
                # ADDED CHECK: Only buy if there are no current BTC holdings
                if btc_holdings == 0 and cash > 0 and current_price > 0:
                    buy_amount_btc = (cash / current_price) * (1 - transaction_cost_rate)
                    btc_holdings += buy_amount_btc
                    previous_amount_used=cash
                    cash = 0.0
                    
                    trades_log.append({
                        'time': historical_data_with_features.index[i],
                        'type': 'BUY',
                        'price': current_price,
                        'btc_change': buy_amount_btc,
                        'cash_change': -previous_amount_used,
                        'current_cash':cash,
                        'current_btc': btc_holdings,
                        'Profit/Loss': "0.0000",
                        'Portfolio_value': cash + (btc_holdings * current_price)
                    })
                # No change to this part
                else:
                    logger.info(f"Skipping buy signal at {historical_data_with_features.index[i]} due to existing holdings or insufficient funds.")

            elif predicted_action == 0:  # SELL signal
                if btc_holdings > 0 and current_price > 0:
                    revenue = (btc_holdings * current_price) * (1 - transaction_cost_rate)
                    realized_profit_loss = (revenue - previous_amount_used)
                    
                    cash += revenue
                    btc_holdings = 0.0
                    
                    trades_log.append({
                        'time': historical_data_with_features.index[i],
                        'type': 'SELL',
                        'price': current_price,
                        'btc_change': "0.00000000",
                        'cash_change': revenue,
                        'current_cash': cash,
                        'current_btc': "0.00000000",
                        'Profit/Loss': realized_profit_loss,
                        'Portfolio_value': cash
                    })
                else:
                    logger.info(f"Skipping but signal at {historical_data_with_features.index[i]} due to existing Hold position")
    final_portfolio_value = cash + (btc_holdings * historical_data_with_features['Close'].iloc[-1] if not historical_data_with_features.empty else initial_cash)
    logger.info(f"Backtest finished. Final portfolio value: ${final_portfolio_value:,.2f}")
    
    trades_df = pd.DataFrame(trades_log)
    portfolio_df = pd.DataFrame(portfolio_history).set_index('timestamp')

    if not trades_df.empty:
        trades_df['time'] = pd.to_datetime(trades_df['time'])
        numeric_cols = ['price', 'btc_change', 'cash_change', 'current_cash', 'current_btc', 'Profit/Loss', 'Portfolio_value']
        
        for col in numeric_cols:
            trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')

    return trades_df, final_portfolio_value, portfolio_df

def check_and_predict_live_signal(model, features_used, binance_client_instance, prediction_display_placeholder):
    """
    Checks if it's time to make a new live prediction and queues a trade action.
    """
    pending_trade = None
    current_time_utc = datetime.now(pytz.utc)
    last_closed_candle_data_time = current_time_utc.replace(minute=0, second=0, microsecond=0)
    candle_being_predicted_start_time = last_closed_candle_data_time + timedelta(hours=1)
    prediction_trigger_time = last_closed_candle_data_time + timedelta(minutes=30)
    
    # Initialize session state variables if they do not exist
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = None

    should_make_new_prediction = (
        st.session_state.sim_running and
        model and
        current_time_utc >= prediction_trigger_time and
        current_time_utc < candle_being_predicted_start_time and
        st.session_state.last_prediction_time != last_closed_candle_data_time
    )

    if should_make_new_prediction:
        st.session_state.last_prediction_time = last_closed_candle_data_time
        st.session_state.last_predicted_candle_time = candle_being_predicted_start_time
        st.session_state.pending_ml_trade_action = None

        live_features_list = [f for f in features_used if f != 'target_movement']

        try:
            live_btc_df_raw = get_binance_price_data(
                binance_client=binance_client_instance,
                symbol="BTCUSDT",
                interval="1h",
                start_str="30 days ago UTC"
            )
            
            if not live_btc_df_raw.empty:
                cleaned_live_df = clean_data(live_btc_df_raw.copy())
                live_features_df = create_advanced_features(cleaned_live_df)
                
                if not live_features_df.empty and live_features_list:
                    latest_live_features = live_features_df[live_features_list].iloc[-1:].copy()
                    
                    if not latest_live_features.empty:
                        latest_live_features = latest_live_features.fillna(live_features_df[live_features_list].mean())
                        
                        live_prediction = model.predict(latest_live_features)[0]
                        current_price = get_live_btc_price_cached(binance_client_instance)

                        prediction_message = (
                            f"**Live Bot Signal @ {current_time_utc.strftime('%Y-%m-%d %H:%M UTC')}**\n\n"
                            f"Latest Price: **${current_price:,.2f}**\n"
                            f"Prediction for **{candle_being_predicted_start_time.strftime('%H:%M UTC')} candle**:\n"
                        )
                        
                        if live_prediction == 1:
                            prediction_message += "**BUY Signal!** (Predicted UP movement)"
                            if not st.session_state.portfolio['open_position']:
                                st.session_state.pending_ml_trade_action = 'BUY'
                            else:
                                st.info("ML Model predicted BUY, but already in an open position. No action queued.")
                                st.session_state.pending_ml_trade_action = None
                        else:
                            prediction_message += "**SELL/HOLD Signal.** (Predicted DOWN/Flat movement)"
                            if st.session_state.portfolio['open_position']:
                                st.session_state.pending_ml_trade_action = 'SELL'
                            else:
                                st.info("ML Model predicted SELL/HOLD, but no open position. No action queued.")
                                st.session_state.pending_ml_trade_action = None

                        st.session_state.last_live_prediction_display = prediction_message
                        prediction_display_placeholder.markdown(prediction_message)
                        send_telegram_message(prediction_message)
                    else:
                        st.session_state.last_live_prediction_display = "Warning: Could not prepare latest features for ML prediction."
                        st.session_state.pending_ml_trade_action = None
                else:
                    st.session_state.last_live_prediction_display = "Warning: No valid features remaining for prediction."
                    st.session_state.pending_ml_trade_action = None
            else:
                st.session_state.last_live_prediction_display = "Warning: No live historical data available for prediction."
                st.session_state.pending_ml_trade_action = None
        except Exception as e:
            st.error(f"Error in live prediction loop: {e}")
            st.session_state.last_live_prediction_display = f"Error in live prediction: {e}"
            st.session_state.pending_ml_trade_action = None
    else:
        st.session_state.last_live_prediction_display = f"[{current_time_utc.strftime('%Y-%m-%d %H:%M UTC')}] No new prediction triggered."
    
    pending_trade = st.session_state.pending_ml_trade_action
    return pending_trade
    
def execute_ml_trade_at_hour(BinanceClient_simulated, prediction_display_placeholder, pending_trade, transaction_cost_rate=0.001):
    current_time_utc = datetime.now(pytz.utc)
    current_hour_utc_time = current_time_utc.replace(minute=0, second=0, microsecond=0)

    # --- Debug display ---
    prediction_display_placeholder.write(f"[{current_time_utc.strftime('**%H:%M:%S UTC**')}] **Checking trade execution...**")
    st.write(f"**Pending ML Action:** {pending_trade}")
    st.write(f"**Last Predicted Candle Time:** {st.session_state.last_predicted_candle_time.strftime('%H:%M UTC') if st.session_state.last_predicted_candle_time else 'N/A'}")
    st.write(f"**Current Hour UTC:** {current_hour_utc_time.strftime('%H:%M UTC')}")
    st.write(f"**Last Executed ML Trade Hour:** {st.session_state.last_executed_ml_trade_hour}")
    
    # --- Key change: also allow execution if we're already past the scheduled time ---
    if (pending_trade is not None and
        st.session_state.last_predicted_candle_time is not None and
        current_time_utc >= st.session_state.last_predicted_candle_time and
        st.session_state.last_executed_ml_trade_hour != current_hour_utc_time.hour):

        trade_time_label = "immediately (late signal)" if current_time_utc > st.session_state.last_predicted_candle_time else f"for {current_hour_utc_time.strftime('%H:%M UTC')} candle"
        prediction_display_placeholder.write(f"[{current_time_utc.strftime('%H:%M:%S UTC')}] DEBUG: Triggering ML trade execution {trade_time_label}")

        current_price = get_live_btc_price_binance(BinanceClient_simulated)
        if not current_price:
            prediction_display_placeholder.error("Could not get live price for trade execution. Retrying next cycle.")
            return

        p = st.session_state.portfolio
        trade_executed = False

        if pending_trade == 'BUY':
            if not p['open_position'] and p['usd'] >= st.session_state.dca_amount_usd:
                # --- BUY logic here ---
                usd_to_spend = st.session_state.dca_amount_usd
                btc_to_buy = (usd_to_spend / current_price) * (1 - transaction_cost_rate)
                
                p['btc'] += btc_to_buy
                p['usd'] -= usd_to_spend
                p['entry_price'] = current_price
                p['open_position'] = True
                st.session_state.previous_amount_used = usd_to_spend # store for PnL calc
                
                trade_dict = {
                    'time': datetime.now(),
                    'type': 'BUY (ML)',
                    'price': current_price,
                    'btc_change': btc_to_buy,
                    'cash_change': -usd_to_spend,
                    'current_cash': p['usd'],
                    'current_btc': p['btc'],
                    'Profit/Loss': 0.0,
                    'Portfolio_value': p['usd'] + (p['btc'] * current_price)
                }
                
                p['trades'].append(trade_dict)
                log_trade_to_csv(trade_dict)
                
                prediction_display_placeholder.success(f"ML BUY Executed {trade_time_label} at ${current_price:,.2f}")
                trade_executed = True
            else:
                prediction_display_placeholder.info(f"ML BUY signal {trade_time_label} but conditions not met (already open or insufficient USD).")
        
        elif pending_trade == 'SELL':
            if p['open_position'] and p['btc'] > 0:
                # --- SELL logic here ---
                btc_sold = p['btc']
                revenue = (btc_sold * current_price) * (1 - transaction_cost_rate)
                previous_amount_used = st.session_state.previous_amount_used if 'previous_amount_used' in st.session_state else 0
                realized_profit_loss = revenue - previous_amount_used
                
                p['usd'] += revenue
                p['btc'] = 0.0
                p['open_position'] = False
                p['entry_price'] = None

                trade_dict = {
                    'time': datetime.now(),
                    'type': 'SELL (ML)',
                    'price': current_price,
                    'btc_change': -btc_sold,
                    'cash_change': revenue,
                    'current_cash': p['usd'],
                    'current_btc': p['btc'],
                    'Profit/Loss': realized_profit_loss,
                    'Portfolio_value': p['usd']
                }

                p['trades'].append(trade_dict)
                log_trade_to_csv(trade_dict)
                
                prediction_display_placeholder.success(f"✅ ML SELL Executed {trade_time_label} at ${current_price:,.2f}, Realized P/L: ${realized_profit_loss:,.2f}")
                trade_executed = True
            else:
                prediction_display_placeholder.info(f"ML SELL signal {trade_time_label} but no open position to sell.")

        if trade_executed:
            st.session_state.last_executed_ml_trade_hour = current_hour_utc_time.hour
            st.session_state.pending_ml_trade_action = None
            st.session_state.last_live_prediction_display = None
            st.rerun()
        else:
            st.session_state.pending_ml_trade_action = None
            st.session_state.last_live_prediction_display = None
            st.session_state.last_executed_ml_trade_hour = current_hour_utc_time.hour
    return pending_trade



# Caching functions for Streamlit performance
@st.cache_data(ttl=3600)
def get_price_prompt_data_cached(_binance_client):
    """Fetches 24 hours of 1-hour BTCUSDT data for the LLM prompt."""
    if not _binance_client:
        return "Binance client not initialized. Cannot get data for prompt."
    df = get_historical_klines_binance(_binance_client, "BTCUSDT", "1h", "24 hours ago UTC")
    if df.empty:
        return "No data available from Binance for prompt."
    df = df[['Open', 'High', 'Low', 'Close']]
    df.index = df.index.strftime('%Y-%m-%d %H:%M')
    return df.to_string()

@st.cache_data(ttl=60)
def get_live_btc_price_cached(_binance_client):
    """Fetches the current live BTCUSDT price from Binance with caching."""
    return get_live_btc_price_binance(_binance_client)

@st.cache_data(ttl=300)
def get_calculated_atr_cached(_binance_client, symbol: str = 'BTCUSDT', interval: str = '1h', period: str = '30 days'):
    """Calculates the Average True Range (ATR) from Binance historical data, with caching."""
    return calculate_atr_binance(_binance_client, symbol=symbol, interval=interval, period=period)

def log_trade_to_csv(trade):
    """Logs a trade to a CSV file."""
    file_exists = os.path.isfile(TRADES_LOG_FILE)
    with open(TRADES_LOG_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        if isinstance(trade['time'], datetime):
            trade['time'] = trade['time'].strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow(trade)
#=============================================================== --- Weekly Email Report Functions --- =======================================================================

def get_weekly_performance_summary(trade_log_df, portfolio_timeline_df, initial_cash):
    """
    Generates an HTML summary of the past week's trading performance.
    Assumes trade_log_df has a 'time' column and portfolio_timeline_df has datetime index and 'value' column.
    """
    if trade_log_df.empty or portfolio_timeline_df.empty:
        return "<h3>Weekly Crypto Bot Performance Update</h3><p>No trading data available for the past week.</p>"

    # Ensure timestamps are UTC for comparison
    now_utc = datetime.now(pytz.utc)
    one_week_ago_utc = now_utc - timedelta(days=7)

    # Convert 'time' column to datetime objects if it's not already
    if not pd.api.types.is_datetime64_any_dtype(trade_log_df['time']):
        trade_log_df['time'] = pd.to_datetime(trade_log_df['time'], utc=True)
    
    # Ensure portfolio_timeline_df index is datetime and UTC
    if not pd.api.types.is_datetime64_any_dtype(portfolio_timeline_df.index):
        portfolio_timeline_df.index = pd.to_datetime(portfolio_timeline_df.index, utc=True)

    weekly_trades = trade_log_df[trade_log_df['time'] >= one_week_ago_utc]
    weekly_timeline = portfolio_timeline_df[portfolio_timeline_df.index >= one_week_ago_utc]

    if weekly_trades.empty or weekly_timeline.empty:
        return "<h3>Weekly Crypto Bot Performance Update</h3><p>No significant trading activity or portfolio data in the last 7 days.</p>"

    start_value_week = weekly_timeline['value'].iloc[0] # Value at the beginning of the week
    end_value_week = weekly_timeline['value'].iloc[-1]   # Value at the end of the week

    weekly_profit_loss_usd = end_value_week - start_value_week
    weekly_profit_loss_percent = (weekly_profit_loss_usd / start_value_week * 100) if start_value_week != 0 else 0

    num_trades_week = len(weekly_trades)
    
    # Calculate total realized P/L from trades during the week
    realized_pl_week = weekly_trades['Profit/Loss'].sum()

    current_overall_portfolio_value = portfolio_timeline_df['value'].iloc[-1]
    current_btc_price = portfolio_timeline_df['price'].iloc[-1] if 'price' in portfolio_timeline_df.columns else "N/A"

    summary_content = f"""
    <h3>Weekly Crypto Bot Performance Update</h3>
    <p>Report Date: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    <hr>
    <h4>Summary for the Last 7 Days:</h4>
    <ul>
        <li><strong>Weekly Portfolio Change (USD):</strong> ${weekly_profit_loss_usd:,.2f}</li>
        <li><strong>Weekly Portfolio Change (%):</strong> {weekly_profit_loss_percent:,.2f}%</li>
        <li><strong>Realized P/L from Trades This Week:</strong> ${realized_pl_week:,.2f}</li>
        <li><strong>Number of Trades Executed:</strong> {num_trades_week}</li>
    </ul>
    <hr>
    <h4>Current Portfolio Status:</h4>
    <ul>
        <li><strong>Overall Portfolio Value:</strong> ${current_overall_portfolio_value:,.2f}</li>
        <li><strong>Current BTC Price:</strong> ${current_btc_price:,.2f} (from last recorded data)</li>
        <li><strong>Current USD Balance:</strong> ${st.session_state.portfolio['usd']:,.2f}</li>
        <li><strong>Current BTC Holdings:</strong> {st.session_state.portfolio['btc']:,.6f} BTC</li>
    </ul>
    <p>Review your dashboard for detailed insights.</p>
    <br>
    <p>Best regards,</p>
    <p>Your Crypto Trading Bot</p>
    """
    return summary_content

def check_and_send_weekly_report():
    current_time_utc = datetime.now(pytz.utc)
    current_date_utc = current_time_utc.date()

    # Define Monday 9 AM UTC
    monday_9am_utc = current_time_utc.replace(hour=9, minute=0, second=0, microsecond=0)

    # Check if it's Monday AND it's 9 AM UTC (or just after, if the autorefresh lands later)
    # AND a report hasn't been sent for THIS Monday yet.
    if (current_time_utc.weekday() == 0 and # Monday is 0
        current_time_utc >= monday_9am_utc and
        st.session_state.last_weekly_report_sent_date != current_date_utc):

        st.info(f"It's Monday {current_time_utc.strftime('%H:%M UTC')} and time to send the weekly report!")

        # Attempt to get data for the report from backtest results for now
        trade_log_for_report = pd.DataFrame()
        portfolio_timeline_for_report = pd.DataFrame()
        initial_cash_for_report = 0

        if 'backtest_results' in st.session_state and st.session_state.backtest_results and not st.session_state.backtest_results['timeline'].empty:
            trade_log_for_report = st.session_state.backtest_results['trades']
            portfolio_timeline_for_report = st.session_state.backtest_results['timeline']
            initial_cash_for_report = st.session_state.backtest_results['initial_cash']
            st.write("DEBUG: Using backtest results for weekly report.")
        else:
            # If live trading, you'd load live data here
            # Example:
            # try:
            #     trade_log_for_report = pd.read_csv(TRADES_LOG_FILE, parse_dates=['time'])
            #     # You'd also need a way to log your portfolio value over time for 'portfolio_timeline_for_report'
            #     st.write("DEBUG: Attempting to use live trades_log.csv for weekly report.")
            # except Exception as e:
            #     st.warning(f"Could not load live trade log for weekly report: {e}")
            st.warning("No backtest results or live trading data available for weekly report.")
            return # Exit if no data for the report

        if trade_log_for_report.empty or portfolio_timeline_for_report.empty:
            st.warning("Not enough data to generate a meaningful weekly report.")
            return

        with st.spinner("Generating and sending weekly email report..."):
            subject = f"Crypto Bot Weekly Performance Update - {current_date_utc.strftime('%Y-%m-%d')}"
            body = get_weekly_performance_summary(trade_log_for_report, portfolio_timeline_for_report, initial_cash_for_report)
            
            try:
                send_email(subject, body)
                st.session_state.last_weekly_report_sent_date = current_date_utc # Mark as sent for this Monday
                st.success("Weekly performance report email sent successfully!")
            except Exception as e:
                st.error(f"Failed to send weekly report email: {e}")
        st.rerun() # Rerun to update the UI immediately
