from openai import OpenAI
from typing import Dict, Any, Optional
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

def get_llm_suggestion(price_prompt: str, portfolio: dict):
    client = OpenAI(api_key=OPENAI_API_KEY)

    current_price = st.session_state.get("current_btc_price", None)

    if (
        current_price is not None
        and portfolio.get('entry_price') is not None
        and portfolio.get('btc', 0) > 0
    ):
        unrealized_pl = (current_price - portfolio['entry_price']) * portfolio['btc']
    else:
        unrealized_pl = 0.0

    prompt = f"""
    You are a crypto trading assistant.
    Here is the last 24 hours of BTC/USDT hourly data:
    {price_prompt}

    Current portfolio status:
    - Open position: {portfolio.get('open_position', False)}
    - BTC held: {portfolio.get('btc', 0):.6f}
    - USD balance: ${portfolio.get('usd', 0):,.2f}
    - Entry price: {portfolio.get('entry_price', 'N/A')}
    - Current price: {current_price if current_price is not None else 'N/A'}
    - Unrealized P/L: ${unrealized_pl:,.2f}

    Based on this data, what is your trading suggestion?
    Respond with "Buy", "Sell", "Hold" and provide a brief justification.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a crypto trading assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {str(e)}"
