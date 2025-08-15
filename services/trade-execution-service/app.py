"""
Trade Execution Service for Tech Safari 2K25 Robo-Advisor Platform
Handles order management, execution, and trade settlement with paper trading support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
import redis
import json
from enum import Enum
import asyncpg
import yfinance as yf
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trade Execution Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = "postgresql://user:password@postgres:5432/roboadvisor"
REDIS_URL = "redis://redis:6379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TradeRequest(BaseModel):
    user_id: str
    portfolio_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    reason: Optional[str] = None

class BulkTradeRequest(BaseModel):
    user_id: str
    portfolio_id: str
    trades: List[TradeRequest]
    execution_strategy: str = "simultaneous"  # or "sequential"

class OrderResponse(BaseModel):
    order_id: str
    status: OrderStatus
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    average_fill_price: Optional[float]
    estimated_cost: float
    fees: float
    created_at: datetime
    updated_at: datetime

class ExecutionReport(BaseModel):
    execution_id: str
    orders: List[OrderResponse]
    total_cost: float
    total_fees: float
    execution_time_ms: int
    success_rate: float
    failed_orders: List[Dict]

@dataclass
class MarketData:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime

class PaperTradingEngine:
    """Paper trading engine for realistic execution simulation"""
    
    def __init__(self):
        self.market_impact_model = MarketImpactModel()
        self.fee_calculator = FeeCalculator()
        
    async def execute_order(self, order: TradeRequest) -> OrderResponse:
        """Execute a paper trade with realistic simulation"""
        try:
            order_id = str(uuid.uuid4())
            
            # Get current market data
            market_data = await self._get_market_data(order.symbol)
            
            # Calculate execution price with market impact
            execution_price = await self._calculate_execution_price(order, market_data)
            
            # Simulate execution delay
            execution_delay = self._simulate_execution_delay(order)
            await asyncio.sleep(execution_delay / 1000)  # Convert to seconds
            
            # Calculate fees
            fees = self.fee_calculator.calculate_fees(order.quantity, execution_price)
            
            # Create order response
            response = OrderResponse(
                order_id=order_id,
                status=OrderStatus.FILLED,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                average_fill_price=execution_price,
                estimated_cost=order.quantity * execution_price + fees,
                fees=fees,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store execution record
            await self._store_execution_record(order, response)
            
            # Update portfolio holdings
            await self._update_portfolio_holdings(order, response)
            
            logger.info(f"Paper trade executed: {order.symbol} {order.side} {order.quantity} @ ${execution_price:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}")
            return OrderResponse(
                order_id=str(uuid.uuid4()),
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0,
                average_fill_price=None,
                estimated_cost=0,
                fees=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

    async def _get_market_data(self, symbol: str) -> MarketData:
        """Get real-time market data"""
        try:
            # Check cache first
            cache_key = f"market_data:{symbol}"
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return MarketData(**data)
            
            # Fetch from Yahoo Finance with safer access
            ticker = yf.Ticker(symbol)
            current_price = None
            volume = 1000000
            try:
                fi = getattr(ticker, 'fast_info', None)
                if fi:
                    current_price = float(getattr(fi, 'last_price', None) or getattr(fi, 'last_trade_price', None) or 100)
                    volume = int(getattr(fi, 'last_volume', None) or getattr(fi, 'ten_day_average_volume', 1000000))
            except Exception:
                current_price = None
            if not current_price or current_price <= 0:
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else volume
                else:
                    current_price = 100.0
            
            # Simulate bid-ask around current price
            bid = current_price * 0.999
            ask = current_price * 1.001
            
            market_data = MarketData(
                symbol=symbol,
                price=current_price,
                bid=bid,
                ask=ask,
                volume=volume,
                timestamp=datetime.utcnow()
            )
            
            # Cache for 30 seconds
            redis_client.setex(cache_key, 30, json.dumps({
                'symbol': symbol,
                'price': current_price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'timestamp': datetime.utcnow().isoformat()
            }))
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {str(e)}")
            # Return default data
            return MarketData(
                symbol=symbol,
                price=100.0,
                bid=99.9,
                ask=100.1,
                volume=1000000,
                timestamp=datetime.utcnow()
            )

    async def _calculate_execution_price(self, order: TradeRequest, market_data: MarketData) -> float:
        """Calculate realistic execution price with market impact"""
        base_price = market_data.ask if order.side == OrderSide.BUY else market_data.bid
        
        # Apply market impact
        market_impact = self.market_impact_model.calculate_impact(
            order.quantity * base_price,  # Order value
            market_data.volume * base_price,  # Daily volume value
            order.side
        )
        
        # Apply price improvement/degradation
        if order.order_type == OrderType.MARKET:
            execution_price = base_price + market_impact
        elif order.order_type == OrderType.LIMIT:
            # For limit orders, use limit price if favorable
            if order.side == OrderSide.BUY:
                execution_price = min(order.limit_price, base_price + market_impact)
            else:
                execution_price = max(order.limit_price, base_price + market_impact)
        else:
            execution_price = base_price + market_impact
        
        return max(execution_price, 0.01)  # Minimum price of $0.01

    def _simulate_execution_delay(self, order: TradeRequest) -> int:
        """Simulate realistic execution delay in milliseconds"""
        base_delay = 50  # 50ms base delay
        
        # Add randomness
        random_delay = np.random.normal(0, 20)
        
        # Market orders are faster
        if order.order_type == OrderType.MARKET:
            return max(base_delay + random_delay, 10)
        else:
            return max(base_delay * 2 + random_delay, 20)

    async def _store_execution_record(self, order: TradeRequest, response: OrderResponse):
        """Store execution record in database"""
        try:
            # In a real implementation, this would store to PostgreSQL
            execution_record = {
                'order_id': response.order_id,
                'user_id': order.user_id,
                'portfolio_id': order.portfolio_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'execution_price': response.average_fill_price,
                'fees': response.fees,
                'timestamp': response.created_at.isoformat(),
                'reason': order.reason
            }
            
            # Store in Redis for demo
            redis_client.lpush(f"executions:{order.user_id}", json.dumps(execution_record))
            redis_client.expire(f"executions:{order.user_id}", 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Failed to store execution record: {str(e)}")

    async def _update_portfolio_holdings(self, order: TradeRequest, response: OrderResponse):
        """Update portfolio holdings after execution"""
        try:
            holdings_key = f"holdings:{order.portfolio_id}"
            current_holdings = redis_client.hget(holdings_key, order.symbol)
            
            if current_holdings:
                current_shares = float(current_holdings)
            else:
                current_shares = 0.0
            
            # Update shares based on trade
            if order.side == OrderSide.BUY:
                new_shares = current_shares + order.quantity
            else:
                new_shares = current_shares - order.quantity
            
            # Store updated holdings
            if new_shares > 0:
                redis_client.hset(holdings_key, order.symbol, str(new_shares))
            else:
                redis_client.hdel(holdings_key, order.symbol)
            
            redis_client.expire(holdings_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Failed to update portfolio holdings: {str(e)}")

class MarketImpactModel:
    """Model for calculating market impact of trades"""
    
    def calculate_impact(self, order_value: float, daily_volume_value: float, side: OrderSide) -> float:
        """Calculate market impact in dollars per share"""
        
        # Participation rate (order size relative to daily volume)
        participation_rate = order_value / max(daily_volume_value, 1000000)  # Minimum $1M volume
        
        # Base impact (square root model)
        base_impact = 0.001 * np.sqrt(participation_rate)  # 0.1% impact for 1% participation
        
        # Direction adjustment (buying pushes price up, selling down)
        direction_multiplier = 1 if side == OrderSide.BUY else -1
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.1)
        
        return base_impact * direction_multiplier * random_factor

class FeeCalculator:
    """Calculate trading fees and commissions"""
    
    def __init__(self):
        self.commission_per_share = 0.0  # Commission-free trading
        self.sec_fee_rate = 0.0000278  # SEC fee rate
        self.finra_fee_rate = 0.000145  # FINRA fee rate
        self.min_fee = 0.01  # Minimum fee
    
    def calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate total fees for a trade"""
        trade_value = quantity * price
        
        # Commission (usually $0 for ETFs)
        commission = quantity * self.commission_per_share
        
        # Regulatory fees
        sec_fee = trade_value * self.sec_fee_rate
        finra_fee = quantity * self.finra_fee_rate
        
        total_fees = commission + sec_fee + finra_fee
        
        return max(total_fees, self.min_fee)

class TradeExecutionService:
    """Main trade execution service"""
    
    def __init__(self):
        self.paper_engine = PaperTradingEngine()
        
    async def execute_single_trade(self, trade_request: TradeRequest) -> OrderResponse:
        """Execute a single trade"""
        return await self.paper_engine.execute_order(trade_request)
    
    async def execute_bulk_trades(self, bulk_request: BulkTradeRequest) -> ExecutionReport:
        """Execute multiple trades with specified strategy"""
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        orders = []
        failed_orders = []
        total_cost = 0.0
        total_fees = 0.0
        
        try:
            if bulk_request.execution_strategy == "simultaneous":
                # Execute all trades simultaneously
                tasks = [
                    self.paper_engine.execute_order(trade) 
                    for trade in bulk_request.trades
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_orders.append({
                            'trade': bulk_request.trades[i].dict(),
                            'error': str(result)
                        })
                    else:
                        orders.append(result)
                        if result.status == OrderStatus.FILLED:
                            total_cost += result.estimated_cost
                            total_fees += result.fees
                        
            else:  # sequential execution
                for trade in bulk_request.trades:
                    try:
                        result = await self.paper_engine.execute_order(trade)
                        orders.append(result)
                        if result.status == OrderStatus.FILLED:
                            total_cost += result.estimated_cost
                            total_fees += result.fees
                    except Exception as e:
                        failed_orders.append({
                            'trade': trade.dict(),
                            'error': str(e)
                        })
            
            end_time = datetime.utcnow()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            success_rate = len([o for o in orders if o.status == OrderStatus.FILLED]) / len(bulk_request.trades)
            
            return ExecutionReport(
                execution_id=execution_id,
                orders=orders,
                total_cost=total_cost,
                total_fees=total_fees,
                execution_time_ms=execution_time_ms,
                success_rate=success_rate,
                failed_orders=failed_orders
            )
            
        except Exception as e:
            logger.error(f"Bulk execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Bulk execution failed: {str(e)}")

    async def get_execution_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get execution history for a user"""
        try:
            executions = redis_client.lrange(f"executions:{user_id}", 0, limit - 1)
            return [json.loads(execution) for execution in executions]
        except Exception as e:
            logger.error(f"Failed to get execution history: {str(e)}")
            return []

    async def get_portfolio_holdings(self, portfolio_id: str) -> Dict[str, float]:
        """Get current portfolio holdings"""
        try:
            holdings_key = f"holdings:{portfolio_id}"
            holdings = redis_client.hgetall(holdings_key)
            return {symbol: float(shares) for symbol, shares in holdings.items()}
        except Exception as e:
            logger.error(f"Failed to get portfolio holdings: {str(e)}")
            return {}

# Initialize service
trade_service = TradeExecutionService()

# API Endpoints
@app.post("/execute", response_model=OrderResponse)
async def execute_trade(trade_request: TradeRequest):
    """Execute a single trade"""
    return await trade_service.execute_single_trade(trade_request)

@app.post("/execute-bulk", response_model=ExecutionReport)
async def execute_bulk_trades(bulk_request: BulkTradeRequest):
    """Execute multiple trades"""
    return await trade_service.execute_bulk_trades(bulk_request)

@app.get("/history/{user_id}")
async def get_execution_history(user_id: str, limit: int = 100):
    """Get execution history for a user"""
    return await trade_service.get_execution_history(user_id, limit)

@app.get("/holdings/{portfolio_id}")
async def get_portfolio_holdings(portfolio_id: str):
    """Get current portfolio holdings"""
    return await trade_service.get_portfolio_holdings(portfolio_id)

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    engine = PaperTradingEngine()
    market_data = await engine._get_market_data(symbol)
    return {
        'symbol': market_data.symbol,
        'price': market_data.price,
        'bid': market_data.bid,
        'ask': market_data.ask,
        'volume': market_data.volume,
        'timestamp': market_data.timestamp.isoformat()
    }

@app.post("/simulate-rebalancing")
async def simulate_rebalancing(request: Dict):
    """Simulate portfolio rebalancing trades"""
    user_id = request.get('user_id')
    portfolio_id = request.get('portfolio_id')
    target_allocation = request.get('target_allocation', {})
    current_value = request.get('current_value', 100000)
    
    # Get current holdings
    current_holdings = await trade_service.get_portfolio_holdings(portfolio_id)
    
    # Generate rebalancing trades
    trades = []
    for symbol, target_weight in target_allocation.items():
        target_value = current_value * target_weight
        current_shares = current_holdings.get(symbol, 0)
        
        # Get current price
        engine = PaperTradingEngine()
        market_data = await engine._get_market_data(symbol)
        current_value_symbol = current_shares * market_data.price
        
        difference = target_value - current_value_symbol
        
        if abs(difference) > 50:  # Minimum trade threshold
            shares_to_trade = abs(difference) / market_data.price
            side = OrderSide.BUY if difference > 0 else OrderSide.SELL
            
            trade = TradeRequest(
                user_id=user_id,
                portfolio_id=portfolio_id,
                symbol=symbol,
                side=side,
                quantity=shares_to_trade,
                reason="rebalancing"
            )
            trades.append(trade)
    
    if trades:
        bulk_request = BulkTradeRequest(
            user_id=user_id,
            portfolio_id=portfolio_id,
            trades=trades,
            execution_strategy="simultaneous"
        )
        return await trade_service.execute_bulk_trades(bulk_request)
    else:
        return ExecutionReport(
            execution_id=str(uuid.uuid4()),
            orders=[],
            total_cost=0.0,
            total_fees=0.0,
            execution_time_ms=0,
            success_rate=1.0,
            failed_orders=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "trade-execution-service",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "paper_trading",
            "market_impact_modeling",
            "bulk_execution",
            "rebalancing_simulation",
            "execution_history"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)
