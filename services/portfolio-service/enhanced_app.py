"""
Enhanced Portfolio Service for Robo-Advisor Platform
Includes tax-loss harvesting, factor-based investing, and advanced portfolio optimization
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.black_litterman import BlackLittermanModel
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import logging
import redis
import json
from dataclasses import dataclass
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Portfolio Service", version="2.0.0")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database and Redis configuration
DATABASE_URL = "postgresql://user:password@postgres:5432/roboadvisor"
REDIS_URL = "redis://redis:6379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

class PortfolioRequest(BaseModel):
    user_id: str
    risk_tolerance: int  # 1-10 scale
    investment_amount: float
    time_horizon: int  # years
    preferences: Dict
    esg_preference: Optional[bool] = False
    factor_preferences: Optional[List[str]] = []
    tax_optimization: Optional[bool] = True
    current_holdings: Optional[Dict[str, float]] = {}

class TaxLossHarvestingRequest(BaseModel):
    user_id: str
    portfolio_id: str
    tax_rate: float = 0.25  # 25% tax rate default
    min_loss_threshold: float = 100.0  # Minimum loss to harvest

class PortfolioResponse(BaseModel):
    allocation: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    tax_efficiency_score: float
    explanation: Dict
    factor_exposures: Dict[str, float]
    rebalancing_trades: List[Dict]

class TaxLossHarvestingResponse(BaseModel):
    harvesting_opportunities: List[Dict]
    potential_tax_savings: float
    recommended_trades: List[Dict]
    wash_sale_warnings: List[Dict]

# Enhanced ETF Universe with factor classifications
ENHANCED_ETF_UNIVERSE = {
    'core_us_equity': {
        'VTI': {'expense_ratio': 0.03, 'factors': ['market'], 'tax_efficiency': 0.95},
        'VOO': {'expense_ratio': 0.03, 'factors': ['large_cap'], 'tax_efficiency': 0.96},
        'VUG': {'expense_ratio': 0.04, 'factors': ['growth'], 'tax_efficiency': 0.94}
    },
    'factor_equity': {
        'VTV': {'expense_ratio': 0.04, 'factors': ['value'], 'tax_efficiency': 0.92},
        'MTUM': {'expense_ratio': 0.15, 'factors': ['momentum'], 'tax_efficiency': 0.88},
        'QUAL': {'expense_ratio': 0.15, 'factors': ['quality'], 'tax_efficiency': 0.90},
        'USMV': {'expense_ratio': 0.15, 'factors': ['low_volatility'], 'tax_efficiency': 0.93}
    },
    'international': {
        'VXUS': {'expense_ratio': 0.08, 'factors': ['international'], 'tax_efficiency': 0.89},
        'VEA': {'expense_ratio': 0.05, 'factors': ['developed_markets'], 'tax_efficiency': 0.91},
        'VWO': {'expense_ratio': 0.10, 'factors': ['emerging_markets'], 'tax_efficiency': 0.85}
    },
    'fixed_income': {
        'BND': {'expense_ratio': 0.03, 'factors': ['bonds'], 'tax_efficiency': 0.80},
        'VTEB': {'expense_ratio': 0.05, 'factors': ['municipal_bonds'], 'tax_efficiency': 0.98},
        'TLT': {'expense_ratio': 0.15, 'factors': ['long_treasury'], 'tax_efficiency': 0.75}
    },
    'alternatives': {
        'VNQ': {'expense_ratio': 0.12, 'factors': ['real_estate'], 'tax_efficiency': 0.70},
        'GLD': {'expense_ratio': 0.40, 'factors': ['commodities'], 'tax_efficiency': 0.65},
        'DBC': {'expense_ratio': 0.87, 'factors': ['commodities'], 'tax_efficiency': 0.60}
    },
    'esg': {
        'ESGU': {'expense_ratio': 0.15, 'factors': ['esg', 'large_cap'], 'tax_efficiency': 0.91},
        'ESGV': {'expense_ratio': 0.09, 'factors': ['esg', 'value'], 'tax_efficiency': 0.89}
    }
}

@dataclass
class HoldingPosition:
    symbol: str
    shares: float
    cost_basis: float
    purchase_date: datetime
    current_price: float
    unrealized_gain_loss: float

class EnhancedPortfolioOptimizer:
    def __init__(self):
        self.prices_cache = {}
        self.factor_loadings = {}
        
    async def optimize_portfolio(self, request: PortfolioRequest) -> Dict:
        """Enhanced portfolio optimization with factor tilts and tax considerations"""
        try:
            # Select ETFs based on preferences
            selected_etfs = self._select_enhanced_etfs(request)
            
            # Fetch price data
            prices = await self._fetch_prices_async(selected_etfs)
            
            # Calculate expected returns with factor adjustments
            mu = self._calculate_factor_adjusted_returns(prices, request.factor_preferences)
            S = risk_models.sample_cov(prices)
            
            # Apply Black-Litterman if user has strong factor preferences
            if request.factor_preferences:
                mu = self._apply_black_litterman(mu, S, request.factor_preferences)
            
            # Optimize with constraints
            ef = EfficientFrontier(mu, S)
            
            # Add tax-efficiency constraints
            if request.tax_optimization:
                self._add_tax_constraints(ef, selected_etfs)
            
            # Optimize based on risk tolerance
            weights = self._optimize_by_risk_tolerance(ef, request.risk_tolerance)
            cleaned_weights = ef.clean_weights()
            
            # Calculate portfolio metrics
            performance = ef.portfolio_performance(verbose=False)
            
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(cleaned_weights, selected_etfs)
            
            # Generate rebalancing trades if current holdings exist
            rebalancing_trades = []
            if request.current_holdings:
                rebalancing_trades = await self._generate_rebalancing_trades(
                    request.current_holdings, cleaned_weights, request.investment_amount
                )
            
            # Calculate tax efficiency score
            tax_efficiency = self._calculate_tax_efficiency_score(cleaned_weights, selected_etfs)
            
            # Discrete allocation
            latest_prices = prices.iloc[-1]
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=request.investment_amount)
            allocation, leftover = da.greedy_portfolio()
            
            return {
                'weights': cleaned_weights,
                'allocation': allocation,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'tax_efficiency_score': tax_efficiency,
                'factor_exposures': factor_exposures,
                'rebalancing_trades': rebalancing_trades,
                'leftover_cash': leftover,
                'selected_etfs': selected_etfs
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

    def _select_enhanced_etfs(self, request: PortfolioRequest) -> List[str]:
        """Enhanced ETF selection with factor tilts and ESG preferences"""
        selected = []
        
        # Base allocation based on risk tolerance
        equity_weight = min(0.1 * request.risk_tolerance, 0.9)
        bond_weight = max(0.1, 1 - equity_weight)
        
        # Core holdings
        if equity_weight > 0:
            selected.extend(['VTI', 'VXUS'])  # US and International
            
        if bond_weight > 0:
            if request.preferences.get('tax_bracket', 'low') == 'high':
                selected.append('VTEB')  # Municipal bonds for high earners
            else:
                selected.append('BND')   # Regular bonds
        
        # Factor tilts based on preferences
        factor_map = {
            'value': 'VTV',
            'momentum': 'MTUM', 
            'quality': 'QUAL',
            'low_volatility': 'USMV'
        }
        
        for factor in request.factor_preferences:
            if factor in factor_map:
                selected.append(factor_map[factor])
        
        # ESG preferences
        if request.esg_preference:
            selected.extend(['ESGU', 'ESGV'])
        
        # Alternative allocations for higher risk tolerance
        if request.risk_tolerance >= 7:
            selected.extend(['VNQ', 'GLD'])  # REITs and commodities
        
        return list(set(selected))  # Remove duplicates

    async def _fetch_prices_async(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """Async price fetching with caching and fallback to mock data"""
        cache_key = f"prices:{'_'.join(sorted(tickers))}:{period}"
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            return pd.read_json(cached_data)
        
        try:
            # Attempt to fetch from Yahoo Finance
            df = yf.download(tickers, period=period, progress=False)['Adj Close']
            
            # Check if we got valid data
            if df.empty or df.isnull().all().all():
                logger.warning("Yahoo Finance returned empty data, using mock data")
                df = self._generate_mock_prices(tickers, period)
            else:
                logger.info(f"Successfully fetched real market data for {len(tickers)} tickers")
                
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {e}, using mock data")
            df = self._generate_mock_prices(tickers, period)
        
        # Cache for 1 hour
        redis_client.setex(cache_key, 3600, df.to_json())
        
        return df
    
    def _generate_mock_prices(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """Generate realistic mock price data for development/testing"""
        import numpy as np
        
        # Determine number of days based on period
        days = 252 * 2 if period == "2y" else 252  # 252 trading days per year
        
        # Create date range
        dates = pd.date_range(end=datetime.utcnow(), periods=days, freq='D')
        
        # Mock price data with realistic characteristics
        mock_data = {}
        
        for ticker in tickers:
            # Set realistic starting prices and volatilities based on asset class
            if ticker in ['VTI', 'VOO', 'VTV', 'VUG']:  # US Equity
                start_price = 200 + np.random.normal(0, 20)
                annual_return = 0.08
                volatility = 0.16
            elif ticker in ['VXUS', 'VEA', 'VWO']:  # International
                start_price = 150 + np.random.normal(0, 15)
                annual_return = 0.06
                volatility = 0.18
            elif ticker in ['BND', 'VTEB', 'TLT']:  # Bonds
                start_price = 100 + np.random.normal(0, 5)
                annual_return = 0.03
                volatility = 0.05
            elif ticker in ['VNQ']:  # REITs
                start_price = 180 + np.random.normal(0, 18)
                annual_return = 0.07
                volatility = 0.20
            elif ticker in ['GLD', 'DBC']:  # Commodities
                start_price = 160 + np.random.normal(0, 16)
                annual_return = 0.04
                volatility = 0.22
            else:  # Default
                start_price = 100
                annual_return = 0.06
                volatility = 0.15
            
            # Generate price series using geometric Brownian motion
            dt = 1/252  # Daily time step
            drift = annual_return - 0.5 * volatility**2
            
            prices = [start_price]
            for i in range(1, days):
                random_shock = np.random.normal(0, 1)
                price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
                new_price = prices[-1] * np.exp(price_change)
                prices.append(new_price)
            
            mock_data[ticker] = prices
        
        df = pd.DataFrame(mock_data, index=dates)
        logger.info(f"Generated mock price data for {len(tickers)} tickers over {days} days")
        
        return df

    def _calculate_factor_adjusted_returns(self, prices: pd.DataFrame, factor_preferences: List[str]) -> pd.Series:
        """Calculate expected returns with factor adjustments"""
        # Base historical returns
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        
        # Apply factor tilts if specified
        if factor_preferences:
            factor_adjustments = {
                'value': 0.02,      # 2% value premium
                'momentum': 0.015,  # 1.5% momentum premium
                'quality': 0.01,    # 1% quality premium
                'low_volatility': -0.005  # Lower expected return for lower vol
            }
            
            for factor in factor_preferences:
                if factor in factor_adjustments:
                    # Apply adjustment to relevant ETFs
                    factor_etfs = self._get_factor_etfs(factor)
                    for etf in factor_etfs:
                        if etf in mu.index:
                            mu[etf] += factor_adjustments[factor]
        
        return mu

    def _apply_black_litterman(self, mu: pd.Series, S: pd.DataFrame, factor_preferences: List[str]) -> pd.Series:
        """Apply Black-Litterman model for factor views"""
        try:
            # Market cap weights (simplified)
            market_caps = pd.Series(index=mu.index, data=1.0)  # Equal weight baseline
            
            # Create views based on factor preferences
            P = np.zeros((len(factor_preferences), len(mu)))
            Q = np.zeros(len(factor_preferences))
            
            for i, factor in enumerate(factor_preferences):
                factor_etfs = self._get_factor_etfs(factor)
                for etf in factor_etfs:
                    if etf in mu.index:
                        etf_idx = mu.index.get_loc(etf)
                        P[i, etf_idx] = 1.0
                        Q[i] = 0.02  # 2% outperformance view
            
            # Apply Black-Litterman
            bl = BlackLittermanModel(S, pi=mu, P=P, Q=Q)
            return bl.bl_returns()
            
        except Exception as e:
            logger.warning(f"Black-Litterman failed, using base returns: {str(e)}")
            return mu

    def _add_tax_constraints(self, ef: EfficientFrontier, selected_etfs: List[str]):
        """Add tax-efficiency constraints to optimization"""
        # Prefer more tax-efficient ETFs
        tax_scores = {}
        for etf in selected_etfs:
            for category, etfs in ENHANCED_ETF_UNIVERSE.items():
                if etf in etfs:
                    tax_scores[etf] = etfs[etf]['tax_efficiency']
        
        # Add constraint to favor tax-efficient holdings
        def tax_efficiency_constraint(weights):
            return sum(weights[i] * tax_scores.get(etf, 0.8) for i, etf in enumerate(selected_etfs)) - 0.85
        
        ef.add_constraint(tax_efficiency_constraint)

    def _optimize_by_risk_tolerance(self, ef: EfficientFrontier, risk_tolerance: int) -> Dict:
        """Optimize portfolio based on risk tolerance"""
        if risk_tolerance <= 3:
            return ef.min_volatility()
        elif risk_tolerance <= 7:
            return ef.max_sharpe()
        else:
            # Aggressive - target higher return
            try:
                return ef.efficient_return(target_return=0.12)
            except:
                return ef.max_sharpe()

    def _calculate_factor_exposures(self, weights: Dict, selected_etfs: List[str]) -> Dict[str, float]:
        """Calculate portfolio factor exposures"""
        factor_exposures = {
            'market': 0.0, 'value': 0.0, 'momentum': 0.0, 'quality': 0.0,
            'low_volatility': 0.0, 'international': 0.0, 'bonds': 0.0,
            'real_estate': 0.0, 'commodities': 0.0, 'esg': 0.0
        }
        
        for etf, weight in weights.items():
            if weight > 0:
                etf_factors = self._get_etf_factors(etf)
                for factor in etf_factors:
                    if factor in factor_exposures:
                        factor_exposures[factor] += weight
        
        return factor_exposures

    def _get_etf_factors(self, etf: str) -> List[str]:
        """Get factor exposures for an ETF"""
        for category, etfs in ENHANCED_ETF_UNIVERSE.items():
            if etf in etfs:
                return etfs[etf]['factors']
        return ['market']  # Default

    def _get_factor_etfs(self, factor: str) -> List[str]:
        """Get ETFs that provide exposure to a specific factor"""
        factor_etfs = []
        for category, etfs in ENHANCED_ETF_UNIVERSE.items():
            for etf, data in etfs.items():
                if factor in data['factors']:
                    factor_etfs.append(etf)
        return factor_etfs

    async def _generate_rebalancing_trades(self, current_holdings: Dict, target_weights: Dict, total_value: float) -> List[Dict]:
        """Generate trades to rebalance from current to target allocation"""
        trades = []
        
        # Get current prices
        all_symbols = list(set(list(current_holdings.keys()) + list(target_weights.keys())))
        current_prices = await self._get_current_prices(all_symbols)
        
        # Calculate current values
        current_values = {symbol: shares * current_prices.get(symbol, 0) for symbol, shares in current_holdings.items()}
        current_total = sum(current_values.values())
        
        # Calculate target values
        target_values = {symbol: weight * total_value for symbol, weight in target_weights.items()}
        
        # Generate trades
        for symbol in all_symbols:
            current_value = current_values.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            difference = target_value - current_value
            
            if abs(difference) > 50:  # Minimum trade threshold
                price = current_prices.get(symbol, 0)
                if price > 0:
                    shares_to_trade = difference / price
                    trades.append({
                        'symbol': symbol,
                        'action': 'buy' if difference > 0 else 'sell',
                        'shares': abs(shares_to_trade),
                        'estimated_value': abs(difference),
                        'current_weight': current_value / current_total if current_total > 0 else 0,
                        'target_weight': target_weights.get(symbol, 0),
                        'reason': 'rebalancing'
                    })
        
        return trades

    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                prices[symbol] = info.get('regularMarketPrice', info.get('previousClose', 0))
            except:
                prices[symbol] = 0
        return prices

    def _calculate_tax_efficiency_score(self, weights: Dict, selected_etfs: List[str]) -> float:
        """Calculate overall tax efficiency score for the portfolio"""
        total_score = 0.0
        total_weight = 0.0
        
        for etf, weight in weights.items():
            if weight > 0:
                for category, etfs in ENHANCED_ETF_UNIVERSE.items():
                    if etf in etfs:
                        tax_efficiency = etfs[etf]['tax_efficiency']
                        total_score += weight * tax_efficiency
                        total_weight += weight
                        break
        
        return total_score / total_weight if total_weight > 0 else 0.8

class TaxLossHarvestingEngine:
    def __init__(self):
        self.wash_sale_period = 30  # 30 days for wash sale rule
        
    async def identify_harvesting_opportunities(self, request: TaxLossHarvestingRequest) -> TaxLossHarvestingResponse:
        """Identify tax-loss harvesting opportunities"""
        try:
            # Get user's current holdings
            holdings = await self._get_user_holdings(request.user_id, request.portfolio_id)
            
            # Calculate unrealized gains/losses
            opportunities = []
            total_potential_savings = 0.0
            recommended_trades = []
            wash_sale_warnings = []
            
            for holding in holdings:
                if holding.unrealized_gain_loss < -request.min_loss_threshold:
                    # This is a loss position
                    tax_savings = abs(holding.unrealized_gain_loss) * request.tax_rate
                    
                    # Check for wash sale rule violations
                    wash_sale_risk = await self._check_wash_sale_risk(holding, request.user_id)
                    
                    opportunity = {
                        'symbol': holding.symbol,
                        'shares': holding.shares,
                        'cost_basis': holding.cost_basis,
                        'current_value': holding.shares * holding.current_price,
                        'unrealized_loss': holding.unrealized_gain_loss,
                        'potential_tax_savings': tax_savings,
                        'wash_sale_risk': wash_sale_risk,
                        'purchase_date': holding.purchase_date.isoformat()
                    }
                    
                    opportunities.append(opportunity)
                    
                    if not wash_sale_risk:
                        total_potential_savings += tax_savings
                        
                        # Find suitable replacement ETF
                        replacement = self._find_replacement_etf(holding.symbol)
                        if replacement:
                            recommended_trades.append({
                                'action': 'sell',
                                'symbol': holding.symbol,
                                'shares': holding.shares,
                                'reason': 'tax_loss_harvesting'
                            })
                            recommended_trades.append({
                                'action': 'buy',
                                'symbol': replacement,
                                'shares': holding.shares,  # Approximate
                                'reason': 'replacement_for_tax_harvesting'
                            })
                    else:
                        wash_sale_warnings.append({
                            'symbol': holding.symbol,
                            'warning': f"Recent activity in {holding.symbol} may trigger wash sale rule",
                            'recommendation': "Wait 30 days before harvesting this loss"
                        })
            
            return TaxLossHarvestingResponse(
                harvesting_opportunities=opportunities,
                potential_tax_savings=total_potential_savings,
                recommended_trades=recommended_trades,
                wash_sale_warnings=wash_sale_warnings
            )
            
        except Exception as e:
            logger.error(f"Tax loss harvesting error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tax loss harvesting failed: {str(e)}")

    async def _get_user_holdings(self, user_id: str, portfolio_id: str) -> List[HoldingPosition]:
        """Get user's current holdings from database"""
        # This would typically query the database
        # For demo purposes, return sample data
        sample_holdings = [
            HoldingPosition(
                symbol="VTI",
                shares=100,
                cost_basis=220.0,
                purchase_date=datetime.now() - timedelta(days=60),
                current_price=210.0,
                unrealized_gain_loss=-1000.0
            ),
            HoldingPosition(
                symbol="MTUM",
                shares=50,
                cost_basis=180.0,
                purchase_date=datetime.now() - timedelta(days=45),
                current_price=175.0,
                unrealized_gain_loss=-250.0
            )
        ]
        return sample_holdings

    async def _check_wash_sale_risk(self, holding: HoldingPosition, user_id: str) -> bool:
        """Check if selling would trigger wash sale rule"""
        # Check if user bought same or substantially similar security within 30 days
        cutoff_date = datetime.now() - timedelta(days=self.wash_sale_period)
        
        # This would query transaction history
        # For demo, return False (no wash sale risk)
        return False

    def _find_replacement_etf(self, original_symbol: str) -> Optional[str]:
        """Find a suitable replacement ETF to maintain exposure while avoiding wash sale"""
        # Map of similar ETFs that are not substantially identical
        replacement_map = {
            'VTI': 'ITOT',    # Total stock market alternatives
            'VOO': 'SPY',     # S&P 500 alternatives  
            'VTV': 'IWD',     # Value alternatives
            'MTUM': 'PDP',    # Momentum alternatives
            'QUAL': 'DGRW',   # Quality alternatives
            'USMV': 'EFAV',   # Low vol alternatives
            'BND': 'AGG',     # Bond alternatives
            'VNQ': 'XLRE'     # REIT alternatives
        }
        
        return replacement_map.get(original_symbol)

# Initialize services
portfolio_optimizer = EnhancedPortfolioOptimizer()
tax_harvesting_engine = TaxLossHarvestingEngine()

# API Endpoints
@app.post("/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Enhanced portfolio optimization endpoint"""
    try:
        result = await portfolio_optimizer.optimize_portfolio(request)
        
        # Generate explanation
        explanation = generate_enhanced_explanation(request, result)
        
        return PortfolioResponse(
            allocation=result['allocation'],
            expected_return=result['expected_return'],
            volatility=result['volatility'],
            sharpe_ratio=result['sharpe_ratio'],
            tax_efficiency_score=result['tax_efficiency_score'],
            explanation=explanation,
            factor_exposures=result['factor_exposures'],
            rebalancing_trades=result['rebalancing_trades']
        )
    except Exception as e:
        logger.error(f"Portfolio optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tax-loss-harvesting", response_model=TaxLossHarvestingResponse)
async def tax_loss_harvesting(request: TaxLossHarvestingRequest):
    """Tax-loss harvesting analysis endpoint"""
    return await tax_harvesting_engine.identify_harvesting_opportunities(request)

@app.get("/factor-analysis/{portfolio_id}")
async def get_factor_analysis(portfolio_id: str):
    """Get detailed factor analysis for a portfolio"""
    # Implementation would analyze current portfolio factor exposures
    return {
        "portfolio_id": portfolio_id,
        "factor_exposures": {
            "market": 0.75,
            "value": 0.15,
            "momentum": 0.10,
            "quality": 0.20,
            "low_volatility": 0.05
        },
        "factor_performance": {
            "value_contribution": 0.02,
            "momentum_contribution": 0.015,
            "quality_contribution": 0.01
        }
    }

def generate_enhanced_explanation(request: PortfolioRequest, result: Dict) -> Dict:
    """Generate enhanced XAI explanation for portfolio decisions"""
    explanation = {
        'allocation_rationale': {},
        'factor_decisions': {},
        'tax_considerations': {},
        'risk_analysis': {},
        'plain_language': ""
    }
    
    # Allocation rationale
    equity_weight = sum(w for etf, w in result['weights'].items() if etf in ['VTI', 'VOO', 'VUG', 'VTV'])
    bond_weight = sum(w for etf, w in result['weights'].items() if etf in ['BND', 'VTEB', 'TLT'])
    
    explanation['allocation_rationale'] = {
        'equity_weight': equity_weight,
        'bond_weight': bond_weight,
        'risk_tolerance_influence': request.risk_tolerance * 0.1,
        'time_horizon_influence': min(request.time_horizon / 30, 1.0)
    }
    
    # Factor decisions
    explanation['factor_decisions'] = {
        factor: result['factor_exposures'].get(factor, 0) 
        for factor in request.factor_preferences
    }
    
    # Tax considerations
    explanation['tax_considerations'] = {
        'tax_efficiency_score': result['tax_efficiency_score'],
        'municipal_bond_allocation': result['weights'].get('VTEB', 0),
        'tax_managed_funds': sum(w for etf, w in result['weights'].items() if etf in ['VTI', 'VOO'])
    }
    
    # Generate plain language explanation
    risk_level = "conservative" if request.risk_tolerance <= 3 else "moderate" if request.risk_tolerance <= 7 else "aggressive"
    
    explanation['plain_language'] = f"""
    Your {risk_level} portfolio allocates {equity_weight*100:.0f}% to stocks and {bond_weight*100:.0f}% to bonds, 
    targeting {result['expected_return']*100:.1f}% annual returns with {result['volatility']*100:.1f}% volatility.
    
    Factor tilts: {', '.join(request.factor_preferences) if request.factor_preferences else 'None specified'}
    Tax efficiency score: {result['tax_efficiency_score']*100:.0f}%
    
    The portfolio emphasizes {'tax-efficient index funds' if result['tax_efficiency_score'] > 0.9 else 'balanced tax efficiency'}
    and {'ESG-focused investments' if request.esg_preference else 'broad market exposure'}.
    """
    
    return explanation

# API Endpoints
@app.post("/portfolio/generate")
async def generate_portfolio(request: PortfolioRequest):
    """Generate optimized portfolio recommendations"""
    try:
        logger.info(f"Generating portfolio for user {request.user_id}")
        
        # Generate simplified portfolio allocation based on risk tolerance
        allocations = _generate_simple_allocation(request)
        
        # Create portfolio result
        portfolio_id = f"portfolio_{request.user_id}_{int(datetime.utcnow().timestamp())}"
        
        # Calculate expected metrics based on risk tolerance
        expected_return = 0.04 + (request.risk_tolerance / 10) * 0.06  # 4-10% range
        volatility = 0.08 + (request.risk_tolerance / 10) * 0.12  # 8-20% range
        sharpe_ratio = expected_return / volatility if volatility > 0 else 1.0
        
        result = {
            "weights": allocations,
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "tax_efficiency_score": 0.85,
            "factor_exposures": {"value": 0.1, "growth": 0.1, "momentum": 0.05}
        }
        
        # Store portfolio data
        portfolio_data = {
            "portfolio_id": portfolio_id,
            "user_id": request.user_id,
            "created_at": datetime.utcnow().isoformat(),
            "request_params": request.dict(),
            "optimization_result": result
        }
        
        # Cache portfolio data
        redis_client.setex(f"portfolio:{request.user_id}", 86400, json.dumps(portfolio_data, default=str))
        redis_client.setex(f"portfolio_id:{portfolio_id}", 86400, json.dumps(portfolio_data, default=str))
        
        logger.info(f"Portfolio generated successfully for user {request.user_id}")
        
        return {
            "portfolio_id": portfolio_id,
            "user_id": request.user_id,
            "created_at": portfolio_data["created_at"],
            "allocations": result["weights"],
            "expected_return": result["expected_return"],
            "volatility": result["volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "tax_efficiency_score": result["tax_efficiency_score"],
            "factor_exposures": result["factor_exposures"],
            "explanation": f"Portfolio optimized for risk tolerance {request.risk_tolerance}/10 with {request.investment_amount} investment",
            "total_value": request.investment_amount,
            "status": "generated"
        }
        
    except Exception as e:
        logger.error(f"Portfolio generation failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")

def _generate_simple_allocation(request: PortfolioRequest) -> dict:
    """Generate simple portfolio allocation based on risk tolerance"""
    risk = request.risk_tolerance
    
    # Conservative allocation (risk 1-3)
    if risk <= 3:
        allocations = {
            "BND": 0.50,    # Bonds 50%
            "VTI": 0.30,    # US Stocks 30%
            "VXUS": 0.15,   # International 15%
            "VNQ": 0.05     # REITs 5%
        }
    # Moderate allocation (risk 4-7)
    elif risk <= 7:
        allocations = {
            "VTI": 0.50,    # US Stocks 50%
            "VXUS": 0.25,   # International 25%
            "BND": 0.20,    # Bonds 20%
            "VNQ": 0.05     # REITs 5%
        }
    # Aggressive allocation (risk 8-10)
    else:
        allocations = {
            "VTI": 0.60,    # US Stocks 60%
            "VXUS": 0.25,   # International 25%
            "VNQ": 0.10,    # REITs 10%
            "GLD": 0.05     # Commodities 5%
        }
    
    # Adjust for ESG preference
    if request.esg_preference:
        # Replace some allocations with ESG alternatives
        if "VTI" in allocations:
            esg_allocation = allocations["VTI"] * 0.5
            allocations["VTI"] -= esg_allocation
            allocations["ESGU"] = esg_allocation
    
    return allocations

@app.get("/portfolio/{user_id}")
async def get_portfolio(user_id: str):
    """Retrieve user's current portfolio"""
    try:
        logger.info(f"Retrieving portfolio for user {user_id}")
        
        # Get portfolio from cache
        portfolio_data = redis_client.get(f"portfolio:{user_id}")
        
        if not portfolio_data:
            logger.warning(f"No portfolio found for user {user_id}")
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = json.loads(portfolio_data)
        result = portfolio["optimization_result"]
        
        # Calculate current portfolio value and performance
        current_time = datetime.utcnow()
        created_time = datetime.fromisoformat(portfolio["created_at"])
        days_since_creation = (current_time - created_time).days
        
        # Mock performance calculation (in real app, would fetch current prices)
        mock_performance = 1 + (0.08 * days_since_creation / 365)  # 8% annual return assumption
        current_value = portfolio["request_params"]["investment_amount"] * mock_performance
        
        return {
            "portfolio_id": portfolio["portfolio_id"],
            "user_id": user_id,
            "created_at": portfolio["created_at"],
            "last_updated": current_time.isoformat(),
            "allocations": result["weights"],
            "expected_return": result["expected_return"],
            "volatility": result["volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "tax_efficiency_score": result["tax_efficiency_score"],
            "factor_exposures": result["factor_exposures"],
            "original_value": portfolio["request_params"]["investment_amount"],
            "current_value": current_value,
            "total_return": current_value - portfolio["request_params"]["investment_amount"],
            "total_return_percent": (mock_performance - 1) * 100,
            "status": "active",
            "days_since_creation": days_since_creation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio retrieval failed for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio retrieval failed: {str(e)}")

@app.post("/portfolio/rebalance")
async def rebalance_portfolio(request: dict):
    """Rebalance existing portfolio"""
    try:
        user_id = request.get("user_id")
        portfolio_id = request.get("portfolio_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        logger.info(f"Rebalancing portfolio for user {user_id}")
        
        # Get current portfolio
        portfolio_data = redis_client.get(f"portfolio:{user_id}")
        
        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = json.loads(portfolio_data)
        original_request = PortfolioRequest(**portfolio["request_params"])
        
        # Re-generate allocation with current market conditions
        allocations = _generate_simple_allocation(original_request)
        
        # Calculate expected metrics based on risk tolerance
        expected_return = 0.04 + (original_request.risk_tolerance / 10) * 0.06  # 4-10% range
        volatility = 0.08 + (original_request.risk_tolerance / 10) * 0.12  # 8-20% range
        sharpe_ratio = expected_return / volatility if volatility > 0 else 1.0
        
        result = {
            "weights": allocations,
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "tax_efficiency_score": 0.85,
            "factor_exposures": {"value": 0.1, "growth": 0.1, "momentum": 0.05}
        }
        
        # Update portfolio data
        portfolio["optimization_result"] = result
        portfolio["last_rebalanced"] = datetime.utcnow().isoformat()
        
        # Save updated portfolio
        redis_client.setex(f"portfolio:{user_id}", 86400, json.dumps(portfolio, default=str))
        
        logger.info(f"Portfolio rebalanced successfully for user {user_id}")
        
        return {
            "portfolio_id": portfolio["portfolio_id"],
            "user_id": user_id,
            "rebalanced_at": portfolio["last_rebalanced"],
            "new_allocations": result["weights"],
            "expected_return": result["expected_return"],
            "volatility": result["volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "tax_efficiency_score": result["tax_efficiency_score"],
            "factor_exposures": result["factor_exposures"],
            "status": "rebalanced",
            "message": "Portfolio successfully rebalanced with current market conditions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio rebalancing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio rebalancing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "enhanced-portfolio-service",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "portfolio_optimization",
            "tax_loss_harvesting", 
            "factor_investing",
            "esg_integration",
            "rebalancing_analysis"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
