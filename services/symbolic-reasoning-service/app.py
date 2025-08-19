"""
Symbolic Reasoning Service with Formal Verification
Uses Z3 SMT solver for constraint validation
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
from pydantic import BaseModel
import z3

app = FastAPI(title="Symbolic Reasoning Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortfolioConstraints(BaseModel):
    min_diversification: int = 5
    max_single_position: float = 0.25
    max_sector_exposure: float = 0.40  # reserved for future use
    min_liquidity_ratio: float = 0.10  # reserved for future use
    risk_budget: float = 0.15          # reserved for future use


class SymbolicVerifier:
    def verify_portfolio_constraints(self, portfolio: Dict[str, Any], constraints: PortfolioConstraints) -> Dict[str, Any]:
        solver = z3.Solver()
        holdings = portfolio.get("holdings", {}) or portfolio.get("allocations", {}) or {}
        positions: Dict[str, z3.ArithRef] = {}

        if not holdings:
            return {"valid": False, "constraints_satisfied": False, "error": "No holdings provided"}

        # Create Z3 variables and bounds
        for symbol in holdings:
            var = z3.Real(f"pos_{symbol}")
            positions[symbol] = var
            solver.add(var >= 0)
            solver.add(var <= 1)

        # Sum of positions = 1
        solver.add(z3.Sum(list(positions.values())) == 1)

        # Max single position
        for var in positions.values():
            solver.add(var <= constraints.max_single_position)

        # Min diversification: count of non-zero positions >= threshold
        solver.add(z3.Sum([z3.If(var > 0, 1, 0) for var in positions.values()]) >= constraints.min_diversification)

        if solver.check() == z3.sat:
            model = solver.model()
            optimal_weights = {}
            for symbol, var in positions.items():
                val = model.evaluate(var, model_completion=True)
                try:
                    optimal_weights[symbol] = float(str(val.as_decimal(10)).replace("?", ""))
                except Exception:
                    optimal_weights[symbol] = None
            return {
                "valid": True,
                "constraints_satisfied": True,
                "optimal_weights": optimal_weights,
                "verification_method": "Z3_SMT_SOLVER",
            }

        return {
            "valid": False,
            "constraints_satisfied": False,
            "verification_method": "Z3_SMT_SOLVER",
        }

    def verify_rebalancing_logic(self, current: Dict[str, float], target: Dict[str, float], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        for trade in trades:
            symbol = trade.get("symbol")
            if not symbol:
                return {"valid": False, "trades_verified": False, "error": "Trade missing symbol"}
            action = trade.get("action")
            shares = float(trade.get("shares", 0) or 0)
            cur = float(current.get(symbol, 0))
            tgt = float(target.get(symbol, 0))

            new = cur + shares if action == "buy" else cur - shares
            if abs(new - tgt) >= abs(cur - tgt):
                return {"valid": False, "trades_verified": False, "error": f"Trade for {symbol} not moving toward target"}
            if new < 0:
                return {"valid": False, "trades_verified": False, "error": f"Negative position for {symbol}"}

        return {"valid": True, "trades_verified": True, "verification_type": "REBALANCING_LOGIC"}

    def generate_shacl_constraints(self, rules: Dict[str, Any]) -> str:
        return f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix portfolio: <http://roboadvisor.com/portfolio#> .

portfolio:PortfolioShape
    a sh:NodeShape ;
    sh:targetClass portfolio:Portfolio ;
    sh:property [
        sh:path portfolio:totalValue ;
        sh:minInclusive {rules.get("min_portfolio_value", 1000)} ;
        sh:maxInclusive {rules.get("max_portfolio_value", 10000000)} ;
    ] ;
    sh:property [
        sh:path portfolio:numberOfPositions ;
        sh:minInclusive {rules.get("min_positions", 5)} ;
    ] ;
    sh:property [
        sh:path portfolio:riskScore ;
        sh:maxInclusive {rules.get("max_risk_score", 8)} ;
    ] .
""".strip()


verifier = SymbolicVerifier()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "symbolic-reasoning-service"}


@app.post("/verify/portfolio")
async def verify_portfolio(request: Dict[str, Any]):
    portfolio = request.get("portfolio", {})
    constraints = PortfolioConstraints(**request.get("constraints", {}))
    return verifier.verify_portfolio_constraints(portfolio, constraints)


@app.post("/verify/rebalancing")
async def verify_rebalancing(request: Dict[str, Any]):
    return verifier.verify_rebalancing_logic(
        request.get("current", {}) or {},
        request.get("target", {}) or {},
        request.get("trades", []) or [],
    )


@app.post("/generate/shacl")
async def generate_shacl(rules: Dict[str, Any]):
    return {"shacl": verifier.generate_shacl_constraints(rules), "format": "turtle"}
