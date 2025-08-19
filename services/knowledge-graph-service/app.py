"""
Knowledge Graph Service for Robo-Advisor
Manages financial knowledge and reasoning paths using Neo4j
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from datetime import datetime
from neo4j import GraphDatabase
import os
import asyncio
import logging

app = FastAPI(title="Knowledge Graph Service", version="1.0.0")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("kg-service")

# CORS for local dev (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class KnowledgeGraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._initialize_schema()

    def _initialize_schema(self):
        with self.driver.session() as session:
            session.run(
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (s:Security)
                REQUIRE s.symbol IS UNIQUE
                """
            )
            session.run(
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation)
                REQUIRE r.id IS UNIQUE
                """
            )
            session.run(
                """
                CREATE INDEX IF NOT EXISTS FOR (s:Security)
                ON (s.sector, s.risk_score)
                """
            )

    async def multi_hop_reasoning(self, query: str, max_hops: int = 3) -> Dict[str, Any]:
        # Simple demo: treat last token as a symbol
        symbol = query.split()[-1] if query else "AAPL"
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s1:Security {symbol: $symbol})-[:HAS_FACTOR]->(f:Factor)<-[:HAS_FACTOR]-(s2:Security)
                WHERE s1 <> s2
                WITH s1, s2, COUNT(f) as common_factors
                MATCH (s2)-[:CORRELATES_WITH]->(s3:Security)
                WHERE s3 <> s1
                RETURN s1.symbol as source,
                       s2.symbol as intermediate,
                       s3.symbol as target,
                       common_factors,
                       s3.expected_return as target_return
                ORDER BY common_factors DESC
                LIMIT 10
                """,
                symbol=symbol,
            )
            paths: List[Dict[str, Any]] = []
            for r in result:
                paths.append({
                    "reasoning_path": [r["source"], r["intermediate"], r["target"]],
                    "common_factors": r["common_factors"],
                    "target_return": r["target_return"],
                    "explanation": f"Found through {r['common_factors']} shared factors",
                })
            return {
                "query": query,
                "reasoning_type": "multi_hop_correlation",
                "paths": paths,
                "hops": 2,
            }

    async def compliance_check(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        allocations = portfolio.get("allocations", {}) or {}
        symbols = list(allocations.keys())
        with self.driver.session() as session:
            reg_result = session.run(
                """
                MATCH (r:Regulation {type: 'CONCENTRATION_LIMIT'})
                RETURN r.max_single_position as limit, r.id as reg_id
                """
            )
            rec = reg_result.single()
            if rec:
                lim = float(rec["limit"]) if rec["limit"] is not None else 1.0
                reg_id = rec["reg_id"]
                for sym, w in allocations.items():
                    try:
                        wf = float(w)
                    except Exception:
                        wf = 0.0
                    if wf > lim:
                        violations.append({
                            "regulation": reg_id,
                            "symbol": sym,
                            "current": wf,
                            "limit": lim,
                            "severity": "HIGH",
                        })

            if portfolio.get("esg_required"):
                esg_result = session.run(
                    """
                    MATCH (s:Security)
                    WHERE s.symbol IN $symbols AND s.esg_score < 70
                    RETURN s.symbol as symbol, s.esg_score as score
                    """,
                    symbols=symbols,
                )
                for r in esg_result:
                    warnings.append({
                        "type": "ESG_WARNING",
                        "symbol": r["symbol"],
                        "esg_score": r["score"],
                        "threshold": 70,
                    })

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "checked_at": datetime.utcnow().isoformat(),
        }


# Lazy, retrying initialization so container starts before Neo4j is ready
kg_manager: KnowledgeGraphManager | None = None


async def get_manager_with_retry(max_retries: int = None, delay_seconds: float = None) -> KnowledgeGraphManager:
    global kg_manager
    if kg_manager is not None:
        return kg_manager

    retries = int(os.getenv("NEO4J_MAX_RETRIES", str(max_retries or 60)))
    delay = float(os.getenv("NEO4J_RETRY_DELAY", str(delay_seconds or 2.0)))
    last_err: Exception | None = None
    for i in range(retries):
        try:
            kg = KnowledgeGraphManager()
            kg_manager = kg
            logger.info("Connected to Neo4j and initialized schema")
            return kg_manager
        except Exception as e:
            last_err = e
            logger.warning(f"Neo4j not ready yet (attempt {i+1}/{retries}): {e}")
            await asyncio.sleep(delay)
    # If we reach here, give up (callers can handle)
    raise HTTPException(status_code=503, detail=f"Neo4j not ready: {last_err}")


@app.get("/health")
async def health():
    neo4j_connected = False
    try:
        # Try to init quickly but don't raise
        await get_manager_with_retry(max_retries=1, delay_seconds=0.1)
        neo4j_connected = True
    except Exception:
        neo4j_connected = False
    return {
        "status": "ok",
        "service": "knowledge-graph-service",
        "neo4j_connected": neo4j_connected,
    }


@app.post("/reasoning/multi-hop")
async def multi_hop_query(request: Dict[str, Any]):
    mgr = await get_manager_with_retry()
    return await mgr.multi_hop_reasoning(
        request.get("query", ""), int(request.get("max_hops", 3))
    )


@app.post("/compliance/check")
async def check_compliance(portfolio: Dict[str, Any]):
    mgr = await get_manager_with_retry()
    return await mgr.compliance_check(portfolio)


@app.post("/graph/populate")
async def populate_graph():
    mgr = await get_manager_with_retry()
    with mgr.driver.session() as session:
        session.run(
            """
            MERGE (aapl:Security {symbol: 'AAPL'})
            SET aapl.sector = 'Technology',
                aapl.market_cap = 3000000000000,
                aapl.risk_score = 6,
                aapl.esg_score = 82

            MERGE (msft:Security {symbol: 'MSFT'})
            SET msft.sector = 'Technology',
                msft.market_cap = 2800000000000,
                msft.risk_score = 5,
                msft.esg_score = 85

            MERGE (aapl)-[:CORRELATES_WITH {coefficient: 0.75}]->(msft)

            MERGE (tech:Factor {name: 'Technology'})
            MERGE (growth:Factor {name: 'Growth'})
            MERGE (aapl)-[:HAS_FACTOR {weight: 0.9}]->(tech)
            MERGE (aapl)-[:HAS_FACTOR {weight: 0.8}]->(growth)

            MERGE (reg1:Regulation {id: 'FINRA-4210'})
            SET reg1.type = 'CONCENTRATION_LIMIT',
                reg1.max_single_position = 0.25,
                reg1.description = 'No single position > 25%'
            """
        )
    return {"status": "Graph populated with sample data"}
