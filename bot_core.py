import time
import threading
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
import random
from firebase_admin import credentials, initialize_app, firestore
from google.cloud.firestore_v1 import client as FirestoreClient
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Persistence Configuration ---
# Firestore paths must be consistent with security rules
TRADES_COLLECTION_PATH = f"artifacts/{__app_id}/public/data/trade_logs"
STATE_DOC_PATH = f"artifacts/{__app_id}/public/data/bot_state/live_state"

# Global state for bot operations (will be mirrored to/from Firestore)
state: Dict[str, Any] = {
    'portfolio': {'usd': 10000.0, 'btc': 0.0, 'open_position': False, 'entry_price': 0.0, 'trades': []},
    'last_prediction_trigger_time': None,
    'last_execution_hour': None,
    'pending_ml_trade_action': None,
    'stop_loss_pct': 0.02,
    'last_price': 0.0
}
bot_running = False

# --- Firebase Functions (To be used by the Bot Daemon) ---

def initialize_firebase(config, auth_token):
    """Initializes Firebase and returns the Firestore client."""
    # We must use credentials method that works on Google Cloud/Firebase backend environments
    # Since we are using an SDK that expects standard credentials/auth flow for a service account,
    # we need to simulate the initialization context provided by the platform.
    # In a real deployed environment, the service account JSON would be used.
    
    try:
        # Use existing app if already initialized (common in Streamlit environment)
        app = initialize_app(name="bot_daemon")
    except ValueError:
        # If running locally, initialize normally
        if not config:
            logging.error("Firebase config is empty. Cannot initialize database.")
            return None, None
            
        # Mock initialization for platform context (if needed, this would use a service account)
        # For simplicity and to match the platform's execution context:
        # We rely on the platform to handle the underlying authentication context.
        # This implementation requires the `google-cloud-firestore` library to be available.
        app = initialize_app(name="bot_daemon")
        
    db = firestore.client(app)
    logging.info("Firebase initialized successfully. Firestore client ready.")
    return db, None # Auth is not strictly needed for the Cloud SDK client

def update_firestore_doc(db: FirestoreClient, doc_path: str, data: Dict):
    """Updates or creates a Firestore document."""
    try:
        doc_ref = db.document(doc_path)
        doc_ref.set(data, merge=True)
    except Exception as e:
        logging.error(f"Failed to update Firestore document {doc_path}: {e}")

def add_trade_log(db: FirestoreClient, trade_dict: Dict):
    """Adds a new trade entry to the log collection."""
    try:
        collection_ref = db.collection(TRADES_COLLECTION_PATH)
        # Use the transaction time as part of the document data for sorting
        trade_dict['time'] = datetime.now(pytz.utc)
        collection_ref.add(trade_dict)
    except Exception as e:
        logging.error(f"Failed to add trade log to Firestore: {e}")

def get_firestore_doc(db: FirestoreClient, doc_path: str) -> Dict[str, Any]:
    """Retrieves a single document from Firestore."""
    try:
        doc_ref = db.document(doc_path)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            # Convert Firestore Timestamp to datetime objects if present
            if 'last_prediction_trigger_time' in data and hasattr(data['last_prediction_trigger_time'], 'strftime'):
                data['last_prediction_trigger_time'] = data['last_prediction_trigger_time'].replace(tzinfo=pytz.utc)
            if 'last_execution_hour' in data and hasattr(data['last_execution_hour'], 'strftime'):
                data['last_execution_hour'] = data['last_execution_hour'].replace(tzinfo=pytz.utc)

            return data
        return {}
    except Exception as e:
        logging.error(f"Failed to retrieve Firestore document {doc_path}: {e}")
        return {}

def get_firestore_collection(db: FirestoreClient, collection_path: str, limit=10, sort_by='time', descending=True):
    """Retrieves documents from a collection."""
    try:
        query = db.collection(collection_path).order_by(sort_by, direction=firestore.Query.DESCENDING if descending else firestore.Query.ASCENDING).limit(limit)
        results = query.stream()
        return [doc.to_dict() for doc in results]
    except Exception as e:
        logging.error(f"Failed to retrieve Firestore collection {collection_path}: {e}")
        return []

# --- Helper Functions (Simulated) ---

# Mock the machine learning model object
class MockModel:
    def predict(self, features_df):
        # 0 = SELL/HOLD, 1 = BUY
        return np.array([random.choice([0, 1])])

# Mock the Binance client instance
class MockBinanceClient:
    def __init__(self):
        # Used to mock the client connection
        pass

# Mock external dependencies for data and price (replace with actual calls in production)
def get_live_btc_price_binance(client):
    """Mocks fetching the current BTC price."""
    # Use a realistic mock price
    current_time = datetime.now().minute
    # Price variation to simulate market movement
    base_price = 65000 + (current_time * 10) 
    return base_price + random.uniform(-100, 100)

def get_binance_price_data(client, symbol, interval, start_str):
    """Mocks fetching historical data for feature generation."""
    # Creates a mock dataframe with required columns
    dates = pd.date_range(end=datetime.now(pytz.utc), periods=100, freq=interval)
    data = {
        'Close': np.random.normal(loc=65000, scale=1000, size=100),
        'High': np.random.normal(loc=65100, scale=1000, size=100),
        'Low': np.random.normal(loc=64900, scale=1000, size=100),
        'Volume': np.random.uniform(100, 500, size=100)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Open Time'
    return df

def create_advanced_features(df):
    """Mocks feature creation, ensuring a dataframe of correct shape is returned."""
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = np.random.uniform(30, 70, size=len(df))
    df = df.dropna()
    return df

def clean_data(df):
    """Mocks data cleaning."""
    return df.copy()

# --- Execution Logic ---

def execute_ml_trade_at_hour(db: FirestoreClient, current_state: Dict, pending_trade: str, transaction_cost_rate=0.001):
    """
    Executes pending ML trade action at the top of the hour.
    Updates the state in Firestore.
    """
    p = current_state['portfolio']
    current_price = get_live_btc_price_binance(MockBinanceClient()) 
    if not current_price:
        logging.error("ERROR: Could not fetch live BTC price for execution.")
        return

    usd_to_spend = 0.0
    btc_change = 0.0
    cash_change = 0.0
    realized_pl = 0.0
    trade_type = None

    # BUY
    if pending_trade == 'BUY' and not p['open_position'] and p['usd'] > 0:
        usd_to_spend = p['usd']
        btc_bought = (usd_to_spend / current_price) * (1 - transaction_cost_rate)
        
        p['btc'] += btc_bought
        p['usd'] = 0.0
        p['entry_price'] = current_price
        p['open_position'] = True
        btc_change = btc_bought
        cash_change = -usd_to_spend
        trade_type = 'BUY (ML)'
        logging.info(f"EXECUTED BUY at ${current_price:,.2f}. BTC purchased: {btc_bought:.6f}")

    # SELL
    elif pending_trade == 'SELL' and p['open_position'] and p['btc'] > 0:
        btc_sold = p['btc']
        revenue = btc_sold * current_price * (1 - transaction_cost_rate)
        realized_pl = revenue - (p['entry_price'] * btc_sold)

        p['usd'] += revenue
        p['btc'] = 0.0
        p['open_position'] = False
        p['entry_price'] = None
        btc_change = -btc_sold
        cash_change = revenue
        trade_type = 'SELL (ML)'
        logging.info(f"EXECUTED SELL at ${current_price:,.2f}. P/L: ${realized_pl:,.2f}")

    if trade_type:
        p['total_value'] = p['usd'] + p['btc'] * current_price # Update value before logging
        trade_dict = {
            'time': datetime.now(pytz.utc),
            'type': trade_type,
            'price': current_price,
            'btc_change': btc_change,
            'cash_change': cash_change,
            'current_cash': p['usd'],
            'current_btc': p['btc'],
            'Profit/Loss': realized_pl,
            'Portfolio_value': p['total_value']
        }
        add_trade_log(db, trade_dict)
        
        # Clear the pending action and mark execution time
        current_state['pending_ml_trade_action'] = None
        current_state['last_execution_hour'] = datetime.now(pytz.utc).replace(minute=0, second=0, microsecond=0)
        
        # Save the updated state and portfolio back to Firestore
        update_firestore_doc(db, STATE_DOC_PATH, current_state)
    
    return current_state.get('pending_ml_trade_action')

# --- Prediction Logic ---

def check_and_predict_live_signal(model, features_used, client, current_state: Dict, db: FirestoreClient):
    """
    Checks if it's time (HH:30 to HH:55) to make a new live prediction.
    If a signal is generated, it queues the trade action in Firestore.
    """
    current_time_utc = datetime.now(pytz.utc)
    current_minute = current_time_utc.minute

    # Determine the start time of the half-hour segment that just closed (e.g., 10:30 if time is 10:45)
    if current_minute < 30:
        # If 10:15, last completed hour candle closed at 10:00. Trigger time for next prediction is 10:30
        last_trigger_time = current_time_utc.replace(minute=30, second=0, microsecond=0) - timedelta(hours=1)
    else:
        # If 10:45, the current trigger time is 10:30
        last_trigger_time = current_time_utc.replace(minute=30, second=0, microsecond=0)

    # Define the specific window for making a prediction (HH:30 to HH:55)
    prediction_window_start = last_trigger_time
    prediction_window_end = prediction_window_start + timedelta(minutes=25)

    # Update live price and save to state constantly (for dashboard display)
    live_price = get_live_btc_price_binance(client)
    current_state['last_price'] = live_price
    
    # Calculate portfolio value (for continuous display)
    p = current_state['portfolio']
    current_state['portfolio']['total_value'] = p['usd'] + p['btc'] * live_price
    update_firestore_doc(db, STATE_DOC_PATH, {'last_price': live_price, 'portfolio': current_state['portfolio']})


    should_make_new_prediction_now = (
        model and
        current_time_utc >= prediction_window_start and
        current_time_utc < prediction_window_end and
        current_state['last_prediction_trigger_time'] != prediction_window_start 
    )

    if should_make_new_prediction_now:
        current_state['last_prediction_trigger_time'] = prediction_window_start
        current_state['pending_ml_trade_action'] = None # Reset action for new prediction

        logging.info(f"Triggering new prediction at {current_time_utc.strftime('%Y-%m-%d %H:%M UTC')}")
        
        try:
            # Fetch 1h historical data
            df_raw = get_binance_price_data(client, 'BTCUSDT', '1h', '60 days ago UTC')
            
            if df_raw.empty:
                logging.warning("No live historical data available for prediction.")
                return 

            cleaned_df = clean_data(df_raw.copy())
            features_df = create_advanced_features(cleaned_df)
            
            if len(features_df) < 2:
                logging.warning("Not enough complete feature rows to make a prediction for the last closed candle.")
                return 
            
            # Select features for the LAST *COMPLETELY CLOSED* candle (second to last row)
            # This is correct for 1h data fetched at HH:30.
            latest_features_for_prediction = features_df[features_used].iloc[-2:-1].copy()
            latest_features_for_prediction = latest_features_for_prediction.fillna(features_df[features_used].mean())
            
            prediction = model.predict(latest_features_for_prediction)[0]
            
            if prediction == 1: # BUY signal
                if not p['open_position']:
                    current_state['pending_ml_trade_action'] = 'BUY'
                    logging.info("BUY action queued in Firestore.")
            else: # SELL/HOLD signal
                if p['open_position']:
                    current_state['pending_ml_trade_action'] = 'SELL'
                    logging.info("SELL action queued in Firestore.")
            
            # Persist the new state (including pending action and trigger time) to Firestore
            update_firestore_doc(db, STATE_DOC_PATH, current_state)

        except Exception as e:
            logging.error(f"Error during live prediction: {e}")
            current_state['pending_ml_trade_action'] = None
            update_firestore_doc(db, STATE_DOC_PATH, {'pending_ml_trade_action': None}) # Clear pending action on error

    # --- Stop-loss check (runs continuously) ---
    if p['open_position']:
        entry_price = p['entry_price']
        
        # Only set to SELL if not already pending a SELL
        if live_price <= entry_price * (1 - current_state['stop_loss_pct']) and current_state['pending_ml_trade_action'] != 'SELL':
            current_state['pending_ml_trade_action'] = 'SELL'
            logging.warning(f"STOP-LOSS triggered! Current Price: ${live_price:,.2f}. SELL action QUEUED.")
            update_firestore_doc(db, STATE_DOC_PATH, {'pending_ml_trade_action': 'SELL'})

# --- Main Bot Loop (This is the daemon process on AWS) ---

def bot_loop():
    global bot_running

    # Initialize Firebase for the daemon
    db, _ = initialize_firebase(json.loads(typeof __firebase_config !== 'undefined' ? __firebase_config : '{}'), 
                                typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : '')
    
    if not db:
        logging.critical("Bot failed to initialize database. Exiting loop.")
        return

    # Mock dependencies for the bot core
    model = MockModel()
    features_used = ['SMA_10', 'RSI'] # Must match features created in create_advanced_features
    client = MockBinanceClient()

    # Load initial state from Firestore or set defaults
    initial_state_data = get_firestore_doc(db, STATE_DOC_PATH)
    if initial_state_data:
        state.update(initial_state_data)
        logging.info("Bot state loaded from Firestore.")
    else:
        # Initialize state in Firestore if it doesn't exist
        state['last_prediction_trigger_time'] = datetime.now(pytz.utc).replace(year=2000) # Set a far past time
        state['last_execution_hour'] = datetime.now(pytz.utc).replace(year=2000)
        update_firestore_doc(db, STATE_DOC_PATH, state)
        logging.info("Initialized default bot state in Firestore.")


    while bot_running:
        current_time_utc = datetime.now(pytz.utc)
        
        # 1. Prediction/Stop-Loss Check (Continuous - updates state in Firestore)
        # We need to read the state first to ensure we use the very latest pending action
        current_state = get_firestore_doc(db, STATE_DOC_PATH) or state # Use loaded state or in-memory default
        check_and_predict_live_signal(model, features_used, client, current_state, db) 

        # 2. Execution Check (Runs ONLY at HH:00-HH:05)
        is_execution_window = (current_time_utc.minute >= 0 and current_time_utc.minute < 5)
        current_hour_key = current_time_utc.replace(minute=0, second=0, microsecond=0)
        
        # We must re-fetch state just before execution to get the latest pending action/stop-loss status
        current_state = get_firestore_doc(db, STATE_DOC_PATH) or state

        if is_execution_window and current_state.get('last_execution_hour') != current_hour_key:
            
            pending_trade = current_state.get('pending_ml_trade_action')
            
            if pending_trade:
                logging.info(f"⏰ HH:00 Execution Triggered. Fulfilling pending {pending_trade} trade.")
                
                # Execute the trade and update state/logs in Firestore
                execute_ml_trade_at_hour(db, current_state, pending_trade)
                
            else:
                logging.info("HH:00 Execution Triggered, but no pending ML trade action found.")
                # Ensure the execution time is still marked to prevent re-execution next minute
                update_firestore_doc(db, STATE_DOC_PATH, {'last_execution_hour': current_hour_key})

        time.sleep(10) # Reduced sleep time for better timing accuracy around HH:00 and HH:30

    logging.info("Bot loop terminated.")
