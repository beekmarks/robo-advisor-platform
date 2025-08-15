"""
Simplified LLM Service for Tech Safari 2K25 Robo-Advisor Platform
Mock implementation for real-world testing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import redis
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service - Conversational AI", version="1.0.0")

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Initialize clients
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

class ConversationMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None

class ConversationRequest(BaseModel):
    user_id: str
    message: str
    conversation_history: Optional[List[ConversationMessage]] = []
    context: Optional[Dict] = {}

class PreferenceExtractionResult(BaseModel):
    risk_tolerance: Optional[int] = None  # 1-10 scale
    investment_goals: List[str] = []
    time_horizon: Optional[int] = None  # years
    income_level: Optional[str] = None
    investment_experience: Optional[str] = None
    esg_preference: Optional[bool] = None
    factor_preferences: List[str] = []
    confidence_scores: Dict[str, float] = {}

class ConversationResponse(BaseModel):
    response: str
    extracted_preferences: PreferenceExtractionResult
    next_questions: List[str]
    conversation_complete: bool
    explanation: Optional[str] = None

class SimpleLLMService:
    def __init__(self):
        self.mock_responses = [
            "Thank you for sharing that! Based on what you've told me, I can see you're interested in building a solid investment foundation. Can you tell me more about your risk tolerance? Are you comfortable with some ups and downs in your portfolio if it means potentially higher returns?",
            "That's helpful to know about your risk preferences. Now, what are your main financial goals? Are you saving for retirement, a house, your children's education, or something else?",
            "Great! Understanding your timeline is important too. When do you expect to need this money? Are you investing for the short term (1-3 years), medium term (3-10 years), or long term (10+ years)?",
            "Perfect! Based on our conversation, I have a good understanding of your investment preferences. Let me create a personalized portfolio recommendation for you."
        ]
        self.response_index = 0

    async def process_conversation(self, request: ConversationRequest) -> ConversationResponse:
        """Process conversational input with mock responses"""
        try:
            # Mock response generation
            response = self.mock_responses[min(self.response_index, len(self.mock_responses) - 1)]
            self.response_index += 1
            
            # Mock preference extraction based on conversation count
            preferences = self._extract_mock_preferences(request)
            
            # Mock next questions
            next_questions = self._generate_mock_questions(preferences)
            
            # Check if onboarding is complete
            is_complete = self.response_index >= len(self.mock_responses)
            
            # Save conversation state
            await self._save_conversation_state(request.user_id, request, response, preferences)
            
            return ConversationResponse(
                response=response,
                extracted_preferences=preferences,
                next_questions=next_questions,
                conversation_complete=is_complete,
                explanation="This is a mock AI response for testing purposes."
            )
            
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")

    def _extract_mock_preferences(self, request: ConversationRequest) -> PreferenceExtractionResult:
        """Extract mock preferences based on conversation"""
        # Simple mock logic based on keywords in the message
        message_lower = request.message.lower()
        
        risk_tolerance = None
        if any(word in message_lower for word in ['conservative', 'safe', 'low risk']):
            risk_tolerance = 3
        elif any(word in message_lower for word in ['moderate', 'balanced']):
            risk_tolerance = 5
        elif any(word in message_lower for word in ['aggressive', 'high risk', 'growth']):
            risk_tolerance = 8
        
        investment_goals = []
        if 'retirement' in message_lower:
            investment_goals.append('retirement')
        if any(word in message_lower for word in ['house', 'home', 'property']):
            investment_goals.append('house')
        if any(word in message_lower for word in ['education', 'college', 'school']):
            investment_goals.append('education')
        
        time_horizon = None
        if any(word in message_lower for word in ['short', '1 year', '2 year']):
            time_horizon = 2
        elif any(word in message_lower for word in ['medium', '5 year', '10 year']):
            time_horizon = 7
        elif any(word in message_lower for word in ['long', '20 year', '30 year']):
            time_horizon = 25
        
        return PreferenceExtractionResult(
            risk_tolerance=risk_tolerance,
            investment_goals=investment_goals,
            time_horizon=time_horizon,
            confidence_scores={'risk_tolerance': 0.8, 'goals': 0.7}
        )

    def _generate_mock_questions(self, preferences: PreferenceExtractionResult) -> List[str]:
        """Generate mock follow-up questions"""
        questions = []
        
        if preferences.risk_tolerance is None:
            questions.append("How do you feel about market volatility?")
        
        if not preferences.investment_goals:
            questions.append("What are your main financial goals?")
        
        if preferences.time_horizon is None:
            questions.append("When do you expect to need this money?")
        
        return questions[:2]  # Return top 2 questions

    async def _save_conversation_state(self, user_id: str, request: ConversationRequest, 
                                     response: str, preferences: PreferenceExtractionResult):
        """Save conversation state to Redis"""
        try:
            context_key = f"conversation_context:{user_id}"
            context = {
                "preferences": preferences.dict(),
                "conversation_count": len(request.conversation_history) + 1,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Save to Redis with 24 hour TTL
            redis_client.setex(context_key, 86400, json.dumps(context))
            
        except Exception as e:
            logger.error(f"Error saving conversation state: {str(e)}")

# Initialize service
llm_service = SimpleLLMService()

# API Endpoints
@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """Main conversational endpoint"""
    return await llm_service.process_conversation(request)

@app.get("/preferences/{user_id}")
async def get_preferences(user_id: str):
    """Get extracted preferences for a user"""
    try:
        context_key = f"conversation_context:{user_id}"
        context_data = redis_client.get(context_key)
        
        if context_data:
            context = json.loads(context_data)
            return context.get("preferences", {})
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting preferences: {str(e)}")
        return {}

@app.post("/reset/{user_id}")
async def reset_conversation(user_id: str):
    """Reset conversation for a user"""
    context_key = f"conversation_context:{user_id}"
    redis_client.delete(context_key)
    return {"status": "reset", "user_id": user_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "llm-service",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "mock",
        "redis_connected": redis_client.ping()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
