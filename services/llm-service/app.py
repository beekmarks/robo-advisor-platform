"""
LLM Service for Robo-Advisor Platform
Provides conversational AI with preference extraction (simplified version)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import json
import redis
from datetime import datetime
import asyncio
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))

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

class LLMService:
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        # Determine if the API key looks valid (avoid obvious placeholders)
        key = (OPENAI_API_KEY or "").strip()
        is_placeholder = any(s in key.lower() for s in [
            "replace_with_your_openai_key",
            "your-openai-api-key",
            "your-ope",  # partial match seen in defaults
        ])
        # Configure OpenAI client with a sane timeout only when key seems valid
        self.client = OpenAI(timeout=LLM_TIMEOUT_SECONDS) if (key and not is_placeholder) else None
        # Default to a lightweight model; allow override via env
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.timeout = LLM_TIMEOUT_SECONDS

    def _initialize_knowledge_base(self) -> Dict:
        """Initialize financial knowledge base for RAG"""
        return {
            "investment_concepts": {
                "risk_tolerance": "Risk tolerance measures how much volatility an investor can handle in their portfolio. Conservative investors prefer stable returns, while aggressive investors accept higher volatility for potentially higher returns.",
                "diversification": "Diversification spreads investments across different asset classes to reduce risk. A well-diversified portfolio includes stocks, bonds, and other assets.",
                "dollar_cost_averaging": "Dollar-cost averaging involves investing a fixed amount regularly, regardless of market conditions, which can reduce the impact of market volatility.",
                "rebalancing": "Portfolio rebalancing involves periodically adjusting holdings to maintain target asset allocation percentages.",
                "tax_loss_harvesting": "Tax-loss harvesting involves selling losing investments to offset gains and reduce tax liability while maintaining portfolio allocation."
            },
            "asset_classes": {
                "stocks": "Stocks represent ownership in companies and offer potential for growth but with higher volatility.",
                "bonds": "Bonds are debt securities that provide steady income with lower risk than stocks.",
                "etfs": "Exchange-traded funds (ETFs) offer diversified exposure to various asset classes with low fees.",
                "real_estate": "Real estate investments provide inflation protection and portfolio diversification.",
                "commodities": "Commodities like gold and oil can hedge against inflation and provide diversification."
            },
            "factor_investing": {
                "value": "Value investing focuses on stocks trading below their intrinsic value.",
                "momentum": "Momentum investing targets stocks with strong recent performance trends.",
                "quality": "Quality investing emphasizes companies with strong fundamentals and stable earnings.",
                "low_volatility": "Low volatility investing targets stocks with lower price fluctuations."
            }
        }

    async def process_conversation(self, request: ConversationRequest) -> ConversationResponse:
        """Process conversational input and extract investment preferences"""
        try:
            # Get conversation context
            context = await self._get_conversation_context(request.user_id)
            
            # Generate AI response with preference extraction
            response_data = await self._generate_response(request, context)
            
            # Extract and update preferences
            preferences = await self._extract_preferences(request, context)
            
            # Determine next questions
            next_questions = await self._generate_next_questions(preferences, context)
            
            # Check if onboarding is complete
            is_complete = self._is_onboarding_complete(preferences)
            
            # Save conversation state
            await self._save_conversation_state(request.user_id, request, response_data, preferences)
            
            return ConversationResponse(
                response=response_data["response"],
                extracted_preferences=preferences,
                next_questions=next_questions,
                conversation_complete=is_complete,
                explanation=response_data.get("explanation")
            )
            
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")

    async def _generate_response(self, request: ConversationRequest, context: Dict) -> Dict:
        """Generate AI response using OpenAI with RAG"""
        try:
            # Retrieve relevant knowledge
            relevant_knowledge = await self._retrieve_knowledge(request.message)
            
            # Build conversation prompt
            system_prompt = self._build_system_prompt(relevant_knowledge, context)
            
            # Prepare conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in request.conversation_history[-5:]:  # Last 5 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current message
            messages.append({"role": "user", "content": request.message})
            
            # Generate response
            if self.client:
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.chat.completions.create,
                            model=self.model,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=500
                        ),
                        timeout=self.timeout,
                    )
                    ai_response = response.choices[0].message.content
                except asyncio.TimeoutError:
                    logger.warning("OpenAI response timed out; using fallback prompt")
                    ai_response = (
                        "Thanks for sharing. Can you tell me about your risk tolerance (1-10) and main goals "
                        "(e.g., retirement, house, education)?"
                    )
                except Exception as oe:
                    logger.error(f"OpenAI error: {oe}")
                    ai_response = (
                        "Thanks for sharing. Can you tell me about your risk tolerance (1-10) and main goals "
                        "(e.g., retirement, house, education)?"
                    )
            else:
                # Fallback mock response when OPENAI is not configured
                ai_response = (
                    "Thanks for sharing. Can you tell me about your risk tolerance (1-10) and main goals "
                    "(e.g., retirement, house, education)?"
                )
            
            # Generate explanation if needed
            explanation = await self._generate_explanation(request.message, ai_response, relevant_knowledge)
            
            return {
                "response": ai_response,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Could you please try rephrasing your question?",
                "explanation": None
            }

    def _build_system_prompt(self, knowledge: str, context: Dict) -> str:
        """Build system prompt for conversational AI"""
        return f"""
You are a friendly, knowledgeable financial advisor helping users understand their investment preferences through natural conversation. Your goal is to extract key information about their:

1. Risk tolerance (1-10 scale)
2. Investment goals (retirement, house, education, etc.)
3. Time horizon (years until needed)
4. Income level and stability
5. Investment experience
6. ESG/sustainable investing preferences
7. Factor investing interests (value, momentum, quality, low volatility)

Guidelines:
- Be conversational and empathetic
- Ask one question at a time
- Explain financial concepts in simple terms
- Use the provided knowledge base to give accurate information
- Don't make specific investment recommendations
- Focus on understanding their situation and preferences

Relevant Financial Knowledge:
{knowledge}

Current Context:
{json.dumps(context, indent=2)}

Remember: You're helping them discover their preferences, not giving specific investment advice.
"""

    async def _retrieve_knowledge(self, query: str) -> str:
        """Retrieve relevant knowledge from knowledge base using semantic search"""
        try:
            # Simple keyword matching for now (can be enhanced with vector search)
            relevant_info = []
            
            for category, items in self.knowledge_base.items():
                for concept, description in items.items():
                    if any(word in query.lower() for word in concept.split('_')):
                        relevant_info.append(f"{concept}: {description}")
            
            return "\n".join(relevant_info[:3])  # Top 3 relevant pieces
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return "General investment knowledge available upon request."

    async def _extract_preferences(self, request: ConversationRequest, context: Dict) -> PreferenceExtractionResult:
        """Extract investment preferences from conversation"""
        try:
            # Use OpenAI to extract structured preferences
            extraction_prompt = f"""
Analyze this conversation and extract investment preferences. Return a JSON object with:
- risk_tolerance: number 1-10 (if mentioned)
- investment_goals: list of goals mentioned
- time_horizon: number of years (if mentioned)
- income_level: "low", "medium", "high" (if mentioned)
- investment_experience: "beginner", "intermediate", "advanced" (if mentioned)
- esg_preference: boolean (if mentioned)
- factor_preferences: list of factors mentioned (value, momentum, quality, low_volatility)
- confidence_scores: object with confidence 0-1 for each extracted field

User message: {request.message}
Previous context: {json.dumps(context.get('preferences', {}), indent=2)}

Return only valid JSON:
"""

            if self.client:
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.chat.completions.create,
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": extraction_prompt}],
                            temperature=0.1,
                            max_tokens=300
                        ),
                        timeout=self.timeout,
                    )
                    extracted_text = response.choices[0].message.content
                except Exception as oe:
                    logger.error(f"OpenAI extraction error: {oe}")
                    extracted_text = json.dumps({})
            else:
                extracted_text = json.dumps({})
            
            try:
                extracted_data = json.loads(extracted_text)
                return PreferenceExtractionResult(**extracted_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse extracted preferences: {extracted_text}")
                return PreferenceExtractionResult()
                
        except Exception as e:
            logger.error(f"Error extracting preferences: {str(e)}")
            return PreferenceExtractionResult()

    async def _generate_next_questions(self, preferences: PreferenceExtractionResult, context: Dict) -> List[str]:
        """Generate relevant follow-up questions based on current preferences"""
        questions = []
        
        # Check what's missing and suggest next questions
        if preferences.risk_tolerance is None:
            questions.append("How comfortable are you with market volatility? Would you prefer steady, predictable returns or are you okay with ups and downs for potentially higher returns?")
        
        if not preferences.investment_goals:
            questions.append("What are your main financial goals? Are you saving for retirement, a house, your children's education, or something else?")
        
        if preferences.time_horizon is None:
            questions.append("When do you expect to need this money? Are you investing for the short term (1-3 years), medium term (3-10 years), or long term (10+ years)?")
        
        if preferences.investment_experience is None:
            questions.append("How would you describe your investment experience? Are you new to investing, or have you been managing investments for a while?")
        
        if preferences.esg_preference is None:
            questions.append("Are you interested in sustainable or socially responsible investing? Some people prefer investments that align with their values.")
        
        # Return top 2 most relevant questions
        return questions[:2]

    def _is_onboarding_complete(self, preferences: PreferenceExtractionResult) -> bool:
        """Check if enough preferences have been collected"""
        required_fields = [
            preferences.risk_tolerance is not None,
            len(preferences.investment_goals) > 0,
            preferences.time_horizon is not None,
            preferences.investment_experience is not None
        ]
        
        return sum(required_fields) >= 3  # At least 3 out of 4 key preferences

    async def _get_conversation_context(self, user_id: str) -> Dict:
        """Retrieve conversation context from Redis"""
        try:
            context_key = f"conversation_context:{user_id}"
            context_data = redis_client.get(context_key)
            
            if context_data:
                return json.loads(context_data)
            else:
                return {"preferences": {}, "conversation_count": 0}
                
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return {"preferences": {}, "conversation_count": 0}

    async def _save_conversation_state(self, user_id: str, request: ConversationRequest, 
                                     response_data: Dict, preferences: PreferenceExtractionResult):
        """Save conversation state to Redis"""
        try:
            context_key = f"conversation_context:{user_id}"
            
            # Get existing context
            context = await self._get_conversation_context(user_id)
            
            # Update context
            context["preferences"] = preferences.dict()
            context["conversation_count"] = context.get("conversation_count", 0) + 1
            context["last_updated"] = datetime.utcnow().isoformat()
            
            # Save to Redis with 24 hour TTL
            redis_client.setex(context_key, 86400, json.dumps(context))
            
        except Exception as e:
            logger.error(f"Error saving conversation state: {str(e)}")

    async def _generate_explanation(self, user_message: str, ai_response: str, knowledge: str) -> Optional[str]:
        """Generate explanation for AI reasoning (XAI component)"""
        try:
            explanation_prompt = f"""
Briefly explain why you gave this response to help the user understand your reasoning:

User asked: {user_message}
You responded: {ai_response}
Based on knowledge: {knowledge}

Provide a 1-2 sentence explanation of your reasoning:
"""

            if self.client:
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.chat.completions.create,
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": explanation_prompt}],
                            temperature=0.3,
                            max_tokens=100
                        ),
                        timeout=self.timeout,
                    )
                    return response.choices[0].message.content
                except Exception as oe:
                    logger.error(f"OpenAI explanation error: {oe}")
                    return None
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return None

# Initialize service
llm_service = LLMService()

# API Endpoints
@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """Main conversational endpoint"""
    return await llm_service.process_conversation(request)

@app.get("/preferences/{user_id}")
async def get_preferences(user_id: str):
    """Get extracted preferences for a user"""
    context = await llm_service._get_conversation_context(user_id)
    return context.get("preferences", {})

@app.post("/reset/{user_id}")
async def reset_conversation(user_id: str):
    """Reset conversation for a user"""
    context_key = f"conversation_context:{user_id}"
    redis_client.delete(context_key)
    return {"status": "reset", "user_id": user_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_connected = redis_client.ping()
    except:
        redis_connected = False
        
    return {
        "status": "healthy",
        "service": "llm-service",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "mock" if not OPENAI_API_KEY else "openai",
        "openai_configured": bool(OPENAI_API_KEY),
        "redis_connected": redis_connected
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
