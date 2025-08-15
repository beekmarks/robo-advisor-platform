from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal


class TaxLossHarvester:
    def __init__(self):
        self.wash_sale_period_days = 30
        self.min_loss_threshold = Decimal("100")
        self.etf_replacements = {
            'VTI': ['ITOT', 'SCHB'],
            'VOO': ['SPY', 'IVV'],
            'VEA': ['IEFA', 'SCHF'],
            'VWO': ['IEMG', 'SCHE'],
            'BND': ['AGG', 'SCHZ'],
        }

    def identify_harvesting_opportunities(self, holdings: List[Dict]) -> List[Dict]:
        opportunities: List[Dict] = []
        for h in holdings:
            unrealized_loss = Decimal(str(h.get('unrealized_loss', 0)))
            symbol = h.get('symbol')
            if unrealized_loss >= self.min_loss_threshold and symbol:
                if not self.check_wash_sale_rule(h):
                    replacement = self.get_replacement_etf(symbol)
                    opp = {
                        'symbol': symbol,
                        'shares': Decimal(str(h.get('shares', 0))),
                        'unrealized_loss': unrealized_loss,
                        'replacement': replacement,
                        'tax_benefit': (unrealized_loss * Decimal('0.37')),
                    }
                    opportunities.append(opp)
        return opportunities

    def check_wash_sale_rule(self, holding: Dict) -> bool:
        last_purchase: Optional[datetime] = holding.get('last_purchase_date')
        if isinstance(last_purchase, str):
            try:
                last_purchase = datetime.fromisoformat(last_purchase)
            except Exception:
                last_purchase = None
        if last_purchase:
            days = (datetime.utcnow() - last_purchase).days
            return days < self.wash_sale_period_days
        return False

    def get_replacement_etf(self, symbol: str) -> Optional[str]:
        repl = self.etf_replacements.get(symbol, [])
        return repl[0] if repl else None

    def execute_harvest(self, opportunity: Dict) -> Dict:
        symbol = opportunity['symbol']
        replacement = opportunity.get('replacement')
        shares = Decimal(str(opportunity.get('shares', 0)))
        trades = []
        if shares > 0:
            trades.append({'symbol': symbol, 'action': 'sell', 'shares': float(shares)})
            if replacement:
                trades.append({'symbol': replacement, 'action': 'buy', 'shares': float(shares)})
        return {
            'trades': trades,
            'estimated_tax_benefit': float(opportunity.get('tax_benefit', 0)),
            'harvested_loss': float(opportunity.get('unrealized_loss', 0))
        }
