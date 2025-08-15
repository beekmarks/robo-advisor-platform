import React, { useState } from 'react';

const SimpleOnboarding = ({ onComplete, onLogin }) => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome! I\'m here to help you get started with investing. What are your main financial goals?' }
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [preferences, setPreferences] = useState({
    risk_tolerance: null,
    investment_goals: [],
    time_horizon: null
  });
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage = { role: 'user', content: currentMessage };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8085/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo-user',
          message: currentMessage,
          conversation_history: messages
        })
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
        
        // Update preferences based on response
        if (data.extracted_preferences) {
          setPreferences(prev => ({
            ...prev,
            ...data.extracted_preferences
          }));
        }

        // Check if onboarding is complete
        if (data.conversation_complete) {
          setTimeout(() => {
            onComplete(data.extracted_preferences);
          }, 2000);
        }
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: 'Sorry, I\'m having trouble connecting to the AI service. Let me help you with a basic setup instead.' 
        }]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'I\'m having connection issues. Let me provide you with a quick setup instead.' 
      }]);
    }

    setCurrentMessage('');
    setIsLoading(false);
  };

  const quickSetup = () => {
    const basicPreferences = {
      risk_tolerance: 5,
      investment_goals: ['retirement', 'growth'],
      time_horizon: 10,
      investment_experience: 'beginner'
    };
    onComplete(basicPreferences);
  };

  return (
    <div className="card">
      <h2>🤖 Investment Onboarding Chat</h2>
      
      <div style={{ 
        height: '400px', 
        overflowY: 'auto', 
        border: '1px solid #ddd', 
        padding: '15px', 
        marginBottom: '15px',
        backgroundColor: '#f9f9f9'
      }}>
        {messages.map((msg, index) => (
          <div key={index} style={{
            marginBottom: '15px',
            padding: '10px',
            borderRadius: '8px',
            backgroundColor: msg.role === 'user' ? '#007bff' : '#e9ecef',
            color: msg.role === 'user' ? 'white' : 'black',
            marginLeft: msg.role === 'user' ? '20%' : '0',
            marginRight: msg.role === 'assistant' ? '20%' : '0'
          }}>
            <strong>{msg.role === 'user' ? '👤 You' : '🤖 AI Advisor'}:</strong>
            <div style={{ marginTop: '5px' }}>{msg.content}</div>
          </div>
        ))}
        {isLoading && (
          <div style={{ textAlign: 'center', color: '#666' }}>
            🤖 AI is thinking...
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
        <input
          type="text"
          value={currentMessage}
          onChange={(e) => setCurrentMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
          style={{
            flex: 1,
            padding: '10px',
            border: '1px solid #ddd',
            borderRadius: '4px'
          }}
          disabled={isLoading}
        />
        <button 
          className="btn" 
          onClick={sendMessage}
          disabled={isLoading || !currentMessage.trim()}
        >
          Send
        </button>
      </div>

      <div style={{ textAlign: 'center' }}>
        <button className="btn" onClick={quickSetup} style={{ backgroundColor: '#28a745' }}>
          Skip Chat - Use Quick Setup
        </button>
      </div>

      {Object.keys(preferences).some(key => preferences[key]) && (
        <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#e8f5e8', borderRadius: '8px' }}>
          <h4>📊 Extracted Preferences:</h4>
          <ul style={{ textAlign: 'left', margin: 0 }}>
            {preferences.risk_tolerance && <li>Risk Tolerance: {preferences.risk_tolerance}/10</li>}
            {preferences.investment_goals?.length > 0 && <li>Goals: {preferences.investment_goals.join(', ')}</li>}
            {preferences.time_horizon && <li>Time Horizon: {preferences.time_horizon} years</li>}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SimpleOnboarding;
