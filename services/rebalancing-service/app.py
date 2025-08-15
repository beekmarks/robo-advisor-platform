from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic import field_validator
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import os
import httpx

app = FastAPI(title="Rebalancing Service", version="0.1.0")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MARKET_DATA_URL = os.getenv("MARKET_DATA_URL", "http://market-data-service:8082")
TRADE_EXECUTION_URL = os.getenv("TRADE_EXECUTION_URL", "http://trade-execution-service:8086")
MIN_TRADE_USD = Decimal(os.getenv("MIN_TRADE_USD", "10"))

TWO_DP = Decimal("0.01")
FOUR_DP = Decimal("0.0001")


class Portfolio(BaseModel):
    user_id: str = Field(..., description="User identifier")
    holdings: Dict[str, Decimal] = Field(
        default_factory=dict, description="symbol -> shares"
    )
    target_allocation: Dict[str, Decimal] = Field(
        ..., description="symbol -> weight (0..1)"
    )
    last_rebalanced: datetime = Field(default_factory=datetime.utcnow)
    total_value: Optional[Decimal] = Field(
        default=None, description="Optional total portfolio value; computed if missing"
    )

    @field_validator("target_allocation")
    @classmethod
    def normalize_allocation(cls, v: Dict[str, Decimal]) -> Dict[str, Decimal]:
        if not v:
            raise ValueError("target_allocation cannot be empty")
        total = sum(Decimal(str(x)) for x in v.values())
        if total == 0:
            raise ValueError("target_allocation sum cannot be zero")
        if total > Decimal("1.5"):
            v = {k: (Decimal(str(val)) / Decimal("100")) for k, val in v.items()}
        total2 = sum(Decimal(str(x)) for x in v.values())
        if total2 <= Decimal("0.99") or total2 >= Decimal("1.01"):
            if total2 > 0:
                v = {k: (Decimal(str(val)) / total2) for k, val in v.items()}
        return {k: Decimal(str(val)).quantize(FOUR_DP) for k, val in v.items()}


class MarketConditions(BaseModel):
    vix: Optional[Decimal] = Field(default=None)
    sp500_trend: Optional[str] = Field(default=None, description="strong_uptrend|uptrend|neutral|downtrend|strong_downtrend")


class CheckRebalanceRequest(BaseModel):
    portfolio: Portfolio
    market_conditions: Optional[MarketConditions] = None


class TradeOrder(BaseModel):
    symbol: str
    action: str
    shares: Decimal
    order_type: str = Field(default="market")
    limit_price: Optional[Decimal] = None


class ExecuteRebalanceRequest(BaseModel):
    portfolio: Portfolio
    trigger_type: str = Field(..., description="threshold|calendar|strategic")
    market_conditions: Optional[MarketConditions] = None


class MarketDataClient:
    def __init__(self, base_url: str = MARKET_DATA_URL):
        self.base_url = base_url.rstrip("/")

    async def get_price(self, symbol: str) -> Optional[Decimal]:
        url = f"{self.base_url}/quote/{symbol}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = ["price", "regular_market_price", "regularMarketPrice", "last", "close"]
                    for key in candidates:
                        if key in data and data[key] is not None:
                            return Decimal(str(data[key]))
        except Exception:
            pass
        return None

    async def get_prices(self, symbols: List[str]) -> Dict[str, Optional[Decimal]]:
        results: Dict[str, Optional[Decimal]] = {}
        for s in symbols:
            results[s] = await self.get_price(s)
        return results


class TradeExecutionClient:
    def __init__(self, base_url: str = TRADE_EXECUTION_URL):
        self.base_url = base_url.rstrip("/")

    async def execute(self, order: TradeOrder, *, user_id: str, portfolio_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/execute"
        payload = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "symbol": order.symbol,
            "side": order.action,
            "quantity": float(order.shares),
            "order_type": order.order_type,
            "limit_price": float(order.limit_price) if order.limit_price else None,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()


from decimal import Decimal as D

class RebalancingEngine:
    def __init__(self, min_trade_usd: Decimal = Decimal("100")):
        self.threshold_normal = Decimal("0.05")
        self.threshold_high_vol = Decimal("0.10")
        self.min_trade_size = min_trade_usd
        self.market = MarketDataClient()

    @staticmethod
    def _q(v: Decimal, quant: Decimal = FOUR_DP) -> Decimal:
        return Decimal(v).quantize(quant, rounding=ROUND_HALF_UP)

    async def _compute_current_values(self, holdings: Dict[str, Decimal]) -> tuple[Dict[str, Decimal], Decimal, Dict[str, Decimal]]:
        symbols = list(holdings.keys())
        prices = await self.market.get_prices(symbols)
        values: Dict[str, Decimal] = {}
        missing: Dict[str, Decimal] = {}
        total = Decimal("0")
        for sym, shares in holdings.items():
            price = prices.get(sym)
            if price is None:
                missing[sym] = shares
                continue
            val = Decimal(str(shares)) * price
            val = val.quantize(TWO_DP, rounding=ROUND_HALF_UP)
            values[sym] = val
            total += val
        total = total.quantize(TWO_DP, rounding=ROUND_HALF_UP)
        return values, total, {k: holdings[k] for k in missing}

    @staticmethod
    def calculate_allocation(values: Dict[str, Decimal], total: Decimal) -> Dict[str, Decimal]:
        if total <= 0:
            return {k: Decimal("0") for k in values.keys()}
        alloc = {k: (v / total) for k, v in values.items()}
        return {k: Decimal(str(w)).quantize(FOUR_DP) for k, w in alloc.items()}

    def calculate_drift(self, current_allocation: Dict[str, Decimal], target_allocation: Dict[str, Decimal]) -> Decimal:
        all_symbols = set(current_allocation.keys()) | set(target_allocation.keys())
        total_drift = Decimal("0")
        for sym in all_symbols:
            cur = Decimal(str(current_allocation.get(sym, Decimal("0"))))
            tgt = Decimal(str(target_allocation.get(sym, Decimal("0"))))
            total_drift += abs(cur - tgt)
        return (total_drift / Decimal("2")).quantize(FOUR_DP)

    def _threshold(self, market_conditions: Optional[MarketConditions]) -> Decimal:
        if market_conditions and market_conditions.vix is not None:
            try:
                vix = Decimal(str(market_conditions.vix))
                if vix > Decimal("30"):
                    return self.threshold_high_vol
            except Exception:
                pass
        return self.threshold_normal

    async def should_rebalance(self, portfolio: Portfolio, market_conditions: Optional[MarketConditions]) -> Dict[str, Any]:
        values, total, missing = await self._compute_current_values(
            {k: Decimal(str(v)) for k, v in portfolio.holdings.items()}
        )
        current_alloc = self.calculate_allocation(values, total)
        drift = self.calculate_drift(current_alloc, portfolio.target_allocation)

        threshold = self._threshold(market_conditions)
        # Normalize timezone to avoid naive/aware subtraction issues
        lr = portfolio.last_rebalanced
        if getattr(lr, 'tzinfo', None) is not None and lr.tzinfo is not None:
            lr_naive = lr.replace(tzinfo=None)
        else:
            lr_naive = lr
        days_since = (datetime.utcnow() - lr_naive).days
        calendar_trigger = days_since >= 90

        return {
            "drift": float(drift),
            "drift_percentage": float((drift * Decimal("100")).quantize(FOUR_DP)),
            "threshold": float(threshold),
            "calendar_trigger": calendar_trigger,
            "should_rebalance": drift > threshold or calendar_trigger,
            "missing_prices": list(missing.keys()),
            "computed_total_value": float(total) if total else None,
            "current_allocation": {k: float(v) for k, v in current_alloc.items()},
        }

    async def generate_trades(self, portfolio: Portfolio) -> List[TradeOrder]:
        values, total, missing = await self._compute_current_values(
            {k: Decimal(str(v)) for k, v in portfolio.holdings.items()}
        )
        total_value = Decimal(str(portfolio.total_value)) if portfolio.total_value else total
        if total_value is None or total_value <= 0:
            return []

        all_symbols = set(values.keys()) | set(portfolio.target_allocation.keys())

        trades: List[TradeOrder] = []
        for sym in all_symbols:
            current_val = values.get(sym, Decimal("0"))
            tgt_weight = Decimal(str(portfolio.target_allocation.get(sym, Decimal("0"))))
            desired_val = (tgt_weight * total_value).quantize(TWO_DP)
            delta_val = (desired_val - current_val).quantize(TWO_DP)

            if abs(delta_val) < self.min_trade_size:
                continue

            price = await self.market.get_price(sym)
            if price is None or price <= 0:
                continue

            shares = (delta_val / price).copy_abs()
            shares = self._q(shares, FOUR_DP)
            if shares <= 0:
                continue

            action = "buy" if delta_val > 0 else "sell"
            trades.append(TradeOrder(symbol=sym, action=action, shares=shares))
        return trades

    async def execute_trades(self, orders: List[TradeOrder], *, user_id: str, portfolio_id: str) -> List[Dict[str, Any]]:
        client = TradeExecutionClient()
        results: List[Dict[str, Any]] = []
        for order in orders:
            try:
                res = await client.execute(order, user_id=user_id, portfolio_id=portfolio_id)
                results.append({"symbol": order.symbol, "result": res})
            except Exception as e:
                results.append({"symbol": order.symbol, "error": str(e)})
        return results

    async def strategic_rebalancing(self, portfolio: Portfolio, market_conditions: Optional[MarketConditions]) -> List[TradeOrder]:
        trades = await self.generate_trades(portfolio)
        return trades


engine = RebalancingEngine(min_trade_usd=MIN_TRADE_USD)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "rebalancing-service",
        "market_data_url": MARKET_DATA_URL,
        "trade_execution_url": TRADE_EXECUTION_URL,
        "min_trade_usd": float(engine.min_trade_size),
        "version": app.version,
    }


@app.post("/check-rebalance")
async def check_rebalance(payload: CheckRebalanceRequest) -> Dict[str, Any]:
    result = await engine.should_rebalance(payload.portfolio, payload.market_conditions)
    return result


@app.post("/generate-trades")
async def generate_trades(payload: CheckRebalanceRequest) -> Dict[str, Any]:
    trades = await engine.generate_trades(payload.portfolio)
    return {
        "orders": [t.dict() for t in trades],
        "count": len(trades),
    }


@app.post("/execute-rebalance")
async def execute_rebalance(payload: ExecuteRebalanceRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    orders = await engine.generate_trades(payload.portfolio)
    derived_portfolio_id = f"rebalance-{payload.portfolio.user_id}"
    results = await engine.execute_trades(orders, user_id=payload.portfolio.user_id, portfolio_id=derived_portfolio_id)
    return {
        "trigger_type": payload.trigger_type,
        "orders": [o.dict() for o in orders],
        "executions": results,
    }


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Rebalancing Service up"}
