#!/usr/bin/env python3
"""
FastAPI-based OpenAI-compatible API server for Vertex AI Gemma3 model.

Installation:
pip install fastapi uvicorn google-auth google-auth-oauthlib requests pydantic

Usage:
1. Configure your Vertex AI settings
2. Run: python fastapi_openai_wrapper.py
3. API will be available at http://localhost:8000
4. Documentation at http://localhost:8000/docs
"""

import time
import uuid
import json
import asyncio
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import requests
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2 import service_account
import google.auth

# Pydantic models for OpenAI API compatibility
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=150, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=40, description="Top-k sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: Optional[int] = Field(default=150, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# Vertex AI Client
class VertexAIClient:
    def __init__(self, project_id: str, location: str, endpoint_id: str, credentials_path: Optional[str] = None):
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id
        
        self.endpoint_url = (
            f"https://{endpoint_id}.{location}-{project_id}.prediction.vertexai.goog"
            f"/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        )
        
        self.credentials = self._setup_credentials(credentials_path)
        
    def _setup_credentials(self, credentials_path: Optional[str]):
        try:
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                credentials, _ = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            return credentials
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to setup credentials: {e}")
    
    def _get_access_token(self) -> str:
        self.credentials.refresh(AuthRequest())
        return self.credentials.token
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Vertex AI endpoint"""
        try:
            access_token = self._get_access_token()
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            # Create payload for Gemma3
            payload = {
                "instances": [{
                    "inputs": prompt,
                }],
                "parameters": {
                    "max_output_tokens": kwargs.get("max_tokens", 150),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": kwargs.get("top_k", 40),
                }
            }
            
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Vertex AI API error: {response.text}"
                )
            
            result = response.json()
            return self._parse_response(result)
            
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {e}")
    
    def _parse_response(self, response_data: Dict) -> str:
        """Parse response from Vertex AI"""
        try:
            predictions = response_data.get("predictions", [])
            if not predictions:
                return "No response generated"
            
            prediction = predictions[0]
            
            if isinstance(prediction, str):
                return prediction
            elif isinstance(prediction, dict):
                for field in ["generated_text", "output", "text", "response", "content"]:
                    if field in prediction:
                        return prediction[field]
                
                for value in prediction.values():
                    if isinstance(value, str):
                        return value
            
            return str(prediction)
            
        except Exception as e:
            return f"Error parsing response: {e}"

# FastAPI Application
app = FastAPI(
    title="OpenAI-Compatible API for Vertex AI Gemma3",
    description="OpenAI-compatible REST API server for Vertex AI Gemma3 model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    PROJECT_ID = "your-project-id"
    LOCATION = "your-location"  # e.g., "us-central1"
    ENDPOINT_ID = "your-endpoint-id"
    CREDENTIALS_PATH = "path/to/your/service-account-key.json"  # Optional
    API_KEY = "sk-1234567890abcdef"  # Change this to your desired API key
    AVAILABLE_MODELS = ["gemma3", "gemma3-chat", "gemma-2-9b-it"]

config = Config()

# Initialize Vertex AI client
vertex_client = VertexAIClient(
    project_id=config.PROJECT_ID,
    location=config.LOCATION,
    endpoint_id=config.ENDPOINT_ID,
    credentials_path=config.CREDENTIALS_PATH if config.CREDENTIALS_PATH != "path/to/your/service-account-key.json" else None
)

# Authentication
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    api_key = authorization.split(" ")[1]
    if api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

# Utility functions
def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return len(text.split())

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt"""
    prompt_parts = []
    for message in messages:
        role = message.role.title()
        content = message.content
        prompt_parts.append(f"{role}: {content}")
    
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

# API Endpoints
@app.get("/v1/models", response_model=ModelList)
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    models = []
    for model_id in config.AVAILABLE_MODELS:
        models.append(ModelInfo(
            id=model_id,
            created=1677610602,
            owned_by="vertex-ai"
        ))
    
    return ModelList(data=models)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a chat completion"""
    if request.model not in config.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    
    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(request.messages)
        
        # Generate response
        response_text = await vertex_client.generate_text(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # Calculate tokens
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response_text)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a text completion"""
    if request.model not in config.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    
    try:
        # Handle single prompt or list of prompts
        if isinstance(request.prompt, list):
            prompt = request.prompt[0]  # Use first prompt for simplicity
        else:
            prompt = request.prompt
        
        # Generate response
        response_text = await vertex_client.generate_text(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Calculate tokens
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response_text)
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=request.model,
            choices=[CompletionChoice(
                index=0,
                text=response_text,
                finish_reason="stop"
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OpenAI-Compatible API for Vertex AI Gemma3",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting OpenAI-Compatible API Server for Vertex AI Gemma3")
    print("=" * 60)
    print(f"📍 Server URL: http://localhost:8000")
    print(f"📋 API Base: http://localhost:8000/v1")
    print(f"🔑 API Key: {config.API_KEY}")
    print(f"📚 Models: {', '.join(config.AVAILABLE_MODELS)}")
    print(f"📖 Documentation: http://localhost:8000/docs")
    print("\n🔧 Configuration:")
    print(f"  Project ID: {config.PROJECT_ID}")
    print(f"  Location: {config.LOCATION}")
    print(f"  Endpoint ID: {config.ENDPOINT_ID}")
    print("\n📝 Example cURL:")
    print("curl -X POST http://localhost:8000/v1/chat/completions \\")
    print(f"  -H 'Authorization: Bearer {config.API_KEY}' \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "model": "gemma3",')
    print('    "messages": [{"role": "user", "content": "Hello!"}],')
    print('    "max_tokens": 150')
    print("  }'")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")