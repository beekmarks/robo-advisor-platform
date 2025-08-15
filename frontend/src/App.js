import React, { useState, useEffect } from 'react';
import SimpleOnboarding from './components/SimpleOnboarding';
import SimpleDashboard from './components/SimpleDashboard';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('onboarding');
  const [user, setUser] = useState(null);
  const [preferences, setPreferences] = useState(null);
  const [servicesStatus, setServicesStatus] = useState({});

  useEffect(() => {
    // Check backend services status
    checkServicesHealth();
  }, []);

  const checkServicesHealth = async () => {
    const services = [
      { name: 'user-service', url: 'http://localhost:8080/health' },
      { name: 'market-data-service', url: 'http://localhost:8082/health' },
      { name: 'portfolio-service', url: 'http://localhost:8083/health' },
  { name: 'rebalancing-service', url: 'http://localhost:8084/health' },
      { name: 'llm-service', url: 'http://localhost:8085/health' },
      { name: 'trade-execution-service', url: 'http://localhost:8086/health' }
    ];

    const status = {};
    for (const service of services) {
      try {
        const response = await fetch(service.url);
        status[service.name] = response.ok ? 'healthy' : 'unhealthy';
      } catch (error) {
        status[service.name] = 'unreachable';
      }
    }
    setServicesStatus(status);
  };

  const handleOnboardingComplete = (userPreferences) => {
    setPreferences(userPreferences);
    setCurrentView('dashboard');
  };

  const handleLogin = (userData) => {
    setUser(userData);
  };

  return (
    <div className="App">
      <header className="app-header">
  <h1>🤖 Robo-Advisor</h1>
        <div className="services-status">
          {Object.entries(servicesStatus).map(([service, status]) => (
            <span key={service} className={`status-indicator ${status}`}>
              {service}: {status}
            </span>
          ))}
        </div>
      </header>

      <main className="container">
        {currentView === 'onboarding' && (
          <SimpleOnboarding 
            onComplete={handleOnboardingComplete}
            onLogin={handleLogin}
          />
        )}
        
        {currentView === 'dashboard' && preferences && (
          <SimpleDashboard 
            preferences={preferences}
            user={user}
          />
        )}

        {currentView === 'test' && (
          <div className="card">
            <h2>🧪 System Test Panel</h2>
            <div className="test-section">
              <h3>Backend Services Health Check</h3>
              <button className="btn" onClick={checkServicesHealth}>
                Refresh Status
              </button>
              <div className="services-grid">
                {Object.entries(servicesStatus).map(([service, status]) => (
                  <div key={service} className={`service-card ${status}`}>
                    <h4>{service}</h4>
                    <span className={`status ${status}`}>{status}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="test-section">
              <h3>Quick API Tests</h3>
              <button className="btn" onClick={() => testMarketData()}>
                Test Market Data
              </button>
              <button className="btn" onClick={() => testLLMService()}>
                Test LLM Service
              </button>
              <button className="btn" onClick={() => testTradeExecution()}>
                Test Trade Execution
              </button>
            </div>
          </div>
        )}

        <div className="navigation">
          <button 
            className={`btn ${currentView === 'onboarding' ? 'active' : ''}`}
            onClick={() => setCurrentView('onboarding')}
          >
            Onboarding
          </button>
          <button 
            className={`btn ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => setCurrentView('dashboard')}
            disabled={!preferences}
          >
            Dashboard
          </button>
          <button 
            className={`btn ${currentView === 'test' ? 'active' : ''}`}
            onClick={() => setCurrentView('test')}
          >
            System Test
          </button>
        </div>
      </main>
    </div>
  );

  async function testMarketData() {
    try {
      const response = await fetch('http://localhost:8082/quote/AAPL');
      const data = await response.json();
      alert(`Market Data Test: ${JSON.stringify(data, null, 2)}`);
    } catch (error) {
      alert(`Market Data Test Failed: ${error.message}`);
    }
  }

  async function testLLMService() {
    try {
      const response = await fetch('http://localhost:8085/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'test-user',
          message: 'Hello, I want to start investing',
          conversation_history: []
        })
      });
      const data = await response.json();
      alert(`LLM Service Test: ${data.response}`);
    } catch (error) {
      alert(`LLM Service Test Failed: ${error.message}`);
    }
  }

  async function testTradeExecution() {
    try {
      const response = await fetch('http://localhost:8086/health');
      const data = await response.json();
      alert(`Trade Execution Test: ${JSON.stringify(data, null, 2)}`);
    } catch (error) {
      alert(`Trade Execution Test Failed: ${error.message}`);
    }
  }
}

export default App;
