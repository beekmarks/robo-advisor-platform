"""
User Service for Robo-Advisor Platform
Handles authentication, user management, and security
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
import bcrypt
import jwt
from datetime import datetime, timedelta
import os
import logging
import asyncpg
import redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="User Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/roboadvisor")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
security = HTTPBearer()

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_data: dict) -> str:
    """Create JWT token"""
    payload = {
        "user_id": user_data["id"],
        "email": user_data["email"],
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    # Get user from cache or database
    user_data = redis_client.get(f"user:{payload['user_id']}")
    if user_data:
        return json.loads(user_data)
    
    # If not in cache, would query database here
    # For demo, return payload data
    return {
        "id": payload["user_id"],
        "email": payload["email"]
    }

@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    try:
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Create user ID (in real app, would be from database)
        user_id = f"user_{int(datetime.utcnow().timestamp())}"
        
        # Store user data (simplified for demo)
        user_record = {
            "id": user_id,
            "email": user_data.email,
            "password_hash": hashed_password,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store in Redis (in real app, would store in PostgreSQL)
        redis_client.setex(f"user:{user_id}", 86400, json.dumps(user_record))
        redis_client.setex(f"user_email:{user_data.email}", 86400, user_id)
        
        logger.info(f"User registered: {user_data.email}")
        
        return UserResponse(
            id=user_id,
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            created_at=datetime.fromisoformat(user_record["created_at"])
        )
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin):
    """Authenticate user and return JWT token"""
    try:
        # Get user by email
        user_id = redis_client.get(f"user_email:{login_data.email}")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_data = redis_client.get(f"user:{user_id}")
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_record = json.loads(user_data)
        
        # Verify password
        if not verify_password(login_data.password, user_record["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create JWT token
        token = create_jwt_token(user_record)
        
        logger.info(f"User logged in: {login_data.email}")
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=JWT_EXPIRATION_HOURS * 3600
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/profile", response_model=UserResponse)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    user_data = redis_client.get(f"user:{current_user['id']}")
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_record = json.loads(user_data)
    return UserResponse(
        id=user_record["id"],
        email=user_record["email"],
        first_name=user_record["first_name"],
        last_name=user_record["last_name"],
        created_at=datetime.fromisoformat(user_record["created_at"])
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "user-service",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
