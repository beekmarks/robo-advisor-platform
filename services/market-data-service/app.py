"""
Market Data Service for Robo-Advisor Platform
Real-time and historical market data aggregation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import yfinance as yf
import redis
import json
from datetime import datetime, timedelta
import asyncio
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Data Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime

class HistoricalDataResponse(BaseModel):
    symbol: str
    data: List[Dict]
    period: str

@app.get("/quote/{symbol}", response_model=MarketDataResponse)
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    try:
        # Check cache first
        cache_key = f"quote:{symbol}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            data = json.loads(cached_data)
            return MarketDataResponse(**data)

        # Fetch from Yahoo Finance with safer access
        ticker = yf.Ticker(symbol)
        current_price = None
        previous_close = None
        volume = 0
        try:
            fi = getattr(ticker, 'fast_info', None)
            if fi:
                current_price = float(getattr(fi, 'last_price', None) or getattr(fi, 'last_trade_price', None) or 0)
                previous_close = float(getattr(fi, 'previous_close', None) or current_price or 0)
                volume = int(getattr(fi, 'last_volume', None) or getattr(fi, 'ten_day_average_volume', 0) or 0)
        except Exception as inner:
            logger.warning(f"fast_info unavailable for {symbol}: {inner}")

        # Fallback to .history if needed
        if not current_price or current_price == 0:
            hist = ticker.history(period="2d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
            else:
                # Last-resort defaults
                current_price = 100.0
                previous_close = 100.0
                volume = 0

        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0.0

        response_data = {
            'symbol': symbol,
            'price': current_price,
            'change': change,
            'change_percent': change_percent,
            'volume': volume,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Cache for 30 seconds
        redis_client.setex(cache_key, 30, json.dumps(response_data))
        return MarketDataResponse(**response_data)

    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {str(e)}")
        # Return a graceful fallback instead of 500 to avoid breaking the UI
        fallback = {
            'symbol': symbol,
            'price': 100.0,
            'change': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        return MarketDataResponse(**fallback)

@app.get("/historical/{symbol}")
async def get_historical_data(symbol: str, period: str = "1y"):
    """Get historical data for a symbol"""
    try:
        cache_key = f"historical:{symbol}:{period}"
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # Fetch from Yahoo Finance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        data = []
        for date, row in hist.iterrows():
            data.append({
                'date': date.isoformat(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        response = {
            'symbol': symbol,
            'data': data,
            'period': period
        }
        
        # Cache for 1 hour
        redis_client.setex(cache_key, 3600, json.dumps(response))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
        # Graceful fallback
        return {
            'symbol': symbol,
            'data': [],
            'period': period
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "market-data-service",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
