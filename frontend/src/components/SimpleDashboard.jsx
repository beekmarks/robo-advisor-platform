import React, { useState, useEffect, useCallback } from 'react';

const VerificationStatus = ({ verification, compliance }) => (
  <div style={{
    padding: '15px',
    backgroundColor: verification?.valid ? '#e8f5e8' : '#fff3cd',
    borderRadius: '8px',
    marginTop: '15px'
  }}>
    <h4>Verification Status</h4>
    <div><strong>Constraints:</strong> {verification?.valid ? 'Satisfied' : 'Violations detected'}</div>
    <div><strong>Compliance:</strong> {compliance?.compliant ? 'Compliant' : 'Non-compliant'}</div>
    {compliance?.violations?.map((v, i) => (
      <div key={i} style={{ color: 'red', marginTop: '5px' }}>
        {v.regulation}: {v.symbol} exceeds {Math.round(v.limit * 100)}% limit
      </div>
    ))}
  </div>
);

const SimpleDashboard = ({ preferences, user }) => {
  const [marketData, setMarketData] = useState(null);
  const [portfolioData, setPortfolioData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [rebalanceCheck, setRebalanceCheck] = useState(null);
  const [rebalanceExec, setRebalanceExec] = useState(null);
  const [rebalancingBusy, setRebalancingBusy] = useState(false);
  const [verification, setVerification] = useState(null);
  const [compliance, setCompliance] = useState(null);
  const [kgQuery, setKgQuery] = useState('');
  const [kgResult, setKgResult] = useState(null);

  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    try {
      // Test market data service
      const marketResponse = await fetch('http://localhost:8082/quote/AAPL');
      if (marketResponse.ok) {
        const marketData = await marketResponse.json();
        setMarketData(marketData);
      }

      // Mock portfolio data based on preferences
      const mockPortfolio = generateMockPortfolio(preferences);
      setPortfolioData(mockPortfolio);

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    }
    setLoading(false);
  }, [preferences]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const generateMockPortfolio = (prefs) => {
    const riskLevel = prefs?.risk_tolerance || 5;
    const isConservative = riskLevel <= 3;
    const isAggressive = riskLevel >= 8;

    return {
      totalValue: 50000,
      dayChange: isAggressive ? 1250 : isConservative ? 125 : 625,
      dayChangePercent: isAggressive ? 2.5 : isConservative ? 0.25 : 1.25,
      allocations: [
        { 
          name: 'Stocks', 
          percentage: isConservative ? 40 : isAggressive ? 80 : 60,
          value: isConservative ? 20000 : isAggressive ? 40000 : 30000
        },
        { 
          name: 'Bonds', 
          percentage: isConservative ? 50 : isAggressive ? 10 : 30,
          value: isConservative ? 25000 : isAggressive ? 5000 : 15000
        },
        { 
          name: 'Cash', 
          percentage: isConservative ? 10 : isAggressive ? 10 : 10,
          value: 5000
        }
      ]
    };
  };

  const executeTrade = async () => {
    try {
      const response = await fetch('http://localhost:8086/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo-user',
          portfolio_id: 'demo-portfolio',
          symbol: 'AAPL',
          side: 'buy',
          quantity: 10
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Trade executed successfully! ${JSON.stringify(result, null, 2)}`);
      } else {
        const errText = await response.text();
        alert(`Trade execution failed: ${errText}`);
      }
    } catch (error) {
      alert(`Trade execution error: ${error.message}`);
    }
  };

  const buildDemoPortfolio = () => {
    // Simple, deterministic demo portfolio using two symbols
    const ninetyDaysAgo = new Date(Date.now() - 120 * 24 * 60 * 60 * 1000).toISOString();
    return {
      user_id: user?.id || 'demo-user',
      holdings: {
        AAPL: 10,
        MSFT: 10
      },
      // Intentional tilt to create drift vs current equal holdings
      target_allocation: {
        AAPL: 0.6,
        MSFT: 0.4
      },
      last_rebalanced: ninetyDaysAgo
    };
  };

  const checkRebalancing = async () => {
    setRebalancingBusy(true);
    setRebalanceExec(null);
    try {
      const response = await fetch('http://localhost:8084/check-rebalance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ portfolio: buildDemoPortfolio() })
      });
      const data = await response.json();
      setRebalanceCheck(data);
    } catch (err) {
      setRebalanceCheck({ error: err.message });
    } finally {
      setRebalancingBusy(false);
    }
  };

  const executeRebalancing = async () => {
    setRebalancingBusy(true);
    try {
      const response = await fetch('http://localhost:8084/execute-rebalance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio: buildDemoPortfolio(),
          trigger_type: 'strategic'
        })
      });
      const data = await response.json();
      setRebalanceExec(data);
    } catch (err) {
      setRebalanceExec({ error: err.message });
    } finally {
      setRebalancingBusy(false);
    }
  };

  const executeVerifiedRebalancing = async () => {
    setRebalancingBusy(true);
    try {
      const response = await fetch('http://localhost:8084/execute-rebalance/verified', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio: buildDemoPortfolio(),
          trigger_type: 'strategic'
        })
      });
      const data = await response.json();
      setVerification(data.verification || null);
      setCompliance(data.compliance || null);
      setRebalanceExec(data);
    } catch (err) {
      console.error('Verified rebalance failed', err);
    } finally {
      setRebalancingBusy(false);
    }
  };

  const queryKnowledgeGraph = async () => {
    try {
      const response = await fetch('http://localhost:8087/reasoning/multi-hop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: kgQuery, max_hops: 3 })
      });
      const data = await response.json();
      setKgResult(data);
    } catch (error) {
      console.error('KG query failed:', error);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <h2>📊 Loading Dashboard...</h2>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          🔄 Loading your personalized investment dashboard...
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <h2>📊 Your Investment Dashboard</h2>
        
        {/* Portfolio Summary */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '20px' }}>
          <div style={{ padding: '15px', backgroundColor: '#e8f5e8', borderRadius: '8px', textAlign: 'center' }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#155724' }}>Total Portfolio Value</h3>
            <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
              ${portfolioData?.totalValue?.toLocaleString()}
            </div>
          </div>
          
          <div style={{ 
            padding: '15px', 
            backgroundColor: portfolioData?.dayChange >= 0 ? '#e8f5e8' : '#f8d7da', 
            borderRadius: '8px', 
            textAlign: 'center' 
          }}>
            <h3 style={{ margin: '0 0 10px 0' }}>Today's Change</h3>
            <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
              ${portfolioData?.dayChange?.toLocaleString()} ({portfolioData?.dayChangePercent}%)
            </div>
          </div>
        </div>

        {/* Asset Allocation */}
        <div style={{ marginBottom: '20px' }}>
          <h3>🎯 Asset Allocation</h3>
          <div style={{ display: 'grid', gap: '10px' }}>
            {portfolioData?.allocations?.map((allocation, index) => (
              <div key={index} style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '10px',
                backgroundColor: '#f8f9fa',
                borderRadius: '4px'
              }}>
                <span style={{ fontWeight: 'bold' }}>{allocation.name}</span>
                <div>
                  <span style={{ marginRight: '15px' }}>{allocation.percentage}%</span>
                  <span>${allocation.value.toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Preferences Summary */}
        <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '8px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>🎯 Your Investment Profile</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
            <div><strong>Risk Tolerance:</strong> {preferences?.risk_tolerance || 'Not set'}/10</div>
            <div><strong>Time Horizon:</strong> {preferences?.time_horizon || 'Not set'} years</div>
            <div><strong>Goals:</strong> {preferences?.investment_goals?.join(', ') || 'Not set'}</div>
            <div><strong>Experience:</strong> {preferences?.investment_experience || 'Not set'}</div>
          </div>
        </div>

        {/* Market Data */}
        {marketData && (
          <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#fff3cd', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 15px 0' }}>📈 Market Data Sample (AAPL)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
              <div><strong>Price:</strong> ${marketData.price}</div>
              <div><strong>Change:</strong> {marketData.change} ({marketData.change_percent}%)</div>
              <div><strong>Volume:</strong> {marketData.volume?.toLocaleString()}</div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button className="btn" onClick={loadDashboardData}>
            🔄 Refresh Data
          </button>
          <button className="btn" onClick={executeTrade} style={{ backgroundColor: '#28a745' }}>
            📈 Test Trade Execution
          </button>
          <button className="btn" onClick={() => window.open('http://localhost:8085/health', '_blank')}>
            🤖 Test AI Service
          </button>
          <button className="btn" onClick={checkRebalancing} disabled={rebalancingBusy}>
            🧮 Check Rebalancing
          </button>
          <button className="btn" onClick={executeRebalancing} disabled={rebalancingBusy} style={{ backgroundColor: '#17a2b8' }}>
            🔧 Execute Rebalance
          </button>
          <button className="btn" onClick={executeVerifiedRebalancing} disabled={rebalancingBusy}>
            ✅ Execute Rebalance (Verified)
          </button>
        </div>

        {(rebalanceCheck || rebalanceExec) && (
          <div style={{ marginTop: '20px' }}>
            {rebalanceCheck && (
              <div style={{ padding: '15px', backgroundColor: '#f1f3f5', borderRadius: '8px', marginBottom: '10px' }}>
                <h3 style={{ marginTop: 0 }}>🧮 Rebalance Check</h3>
                <div><strong>Should Rebalance:</strong> {String(rebalanceCheck.should_rebalance)}</div>
                <div><strong>Drift:</strong> {rebalanceCheck.drift} (threshold {rebalanceCheck.threshold})</div>
                {rebalanceCheck.error && <div style={{ color: 'red' }}>Error: {rebalanceCheck.error}</div>}
              </div>
            )}

            {rebalanceExec && (
              <div style={{ padding: '15px', backgroundColor: '#e8f5e8', borderRadius: '8px' }}>
                <h3 style={{ marginTop: 0 }}>🔧 Rebalance Execution</h3>
                {rebalanceExec.error && <div style={{ color: 'red' }}>Error: {rebalanceExec.error}</div>}
                {Array.isArray(rebalanceExec.orders) && rebalanceExec.orders.length > 0 ? (
                  <div>
                    <div style={{ marginBottom: '8px' }}><strong>Orders:</strong></div>
                    <ul>
                      {rebalanceExec.orders.map((o, idx) => (
                        <li key={idx}>{o.action.toUpperCase()} {o.shares} {o.symbol}</li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <div>No orders generated (below minimum trade size or up-to-date).</div>
                )}
              </div>
            )}
          </div>
        )}

        {(verification || compliance) && (
          <VerificationStatus verification={verification} compliance={compliance} />
        )}

        <div style={{ marginTop: 20 }}>
          <h4>Knowledge Graph Query</h4>
          <input
            type="text"
            placeholder="e.g., find correlation paths for AAPL"
            value={kgQuery}
            onChange={(e) => setKgQuery(e.target.value)}
            style={{ width: '60%', marginRight: 10 }}
          />
          <button className="btn" onClick={queryKnowledgeGraph}>Query KG</button>
          {kgResult && (
            <pre style={{ marginTop: 10, background: '#f6f8fa', padding: 10 }}>
              {JSON.stringify(kgResult, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
};

export default SimpleDashboard;
