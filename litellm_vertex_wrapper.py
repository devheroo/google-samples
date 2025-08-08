#!/usr/bin/env python3
"""
LiteLLM-based OpenAI-compatible API wrapper for Vertex AI Gemma3 model.
This creates a proxy server that translates OpenAI API calls to Vertex AI calls.

Installation:
pip install litellm[proxy] uvicorn

Usage:
1. Configure your settings below
2. Run: python litellm_vertex_wrapper.py
3. Access OpenAI-compatible API at http://localhost:8000
"""

import os
import sys
import yaml
from typing import Dict, Any
import uvicorn
from litellm import Router
import litellm

class VertexAIGemmaWrapper:
    def __init__(self):
        self.setup_environment()
        self.create_litellm_config()
        
    def setup_environment(self):
        """Setup environment variables for Vertex AI"""
        # Configure your Vertex AI settings
        self.project_id = "your-project-id"
        self.location = "your-location"  # e.g., "us-central1"
        self.endpoint_id = "your-endpoint-id"
        self.credentials_path = "path/to/your/service-account-key.json"
        
        # Validate configuration
        if any(val.startswith("your-") for val in [self.project_id, self.location, self.endpoint_id]):
            print("❌ Please configure PROJECT_ID, LOCATION, and ENDPOINT_ID")
            sys.exit(1)
        
        # Set environment variables for LiteLLM
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        os.environ["VERTEXAI_PROJECT"] = self.project_id
        os.environ["VERTEXAI_LOCATION"] = self.location
        
        print(f"✓ Configured for project: {self.project_id}")
        print(f"✓ Location: {self.location}")
        print(f"✓ Endpoint ID: {self.endpoint_id}")

    def create_litellm_config(self):
        """Create LiteLLM configuration for Vertex AI custom endpoint"""
        
        # Custom endpoint URL
        endpoint_url = f"https://{self.endpoint_id}.{self.location}-{self.project_id}.prediction.vertexai.goog/v1/projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}:predict"
        
        # LiteLLM configuration
        config = {
            "model_list": [
                {
                    "model_name": "gemma3",  # This is what clients will use
                    "litellm_params": {
                        "model": "vertex_ai_beta/gemma-2-9b-it",  # Base model identifier
                        "vertex_project": self.project_id,
                        "vertex_location": self.location,
                        "api_base": endpoint_url,
                        "custom_llm_provider": "vertex_ai"
                    }
                },
                {
                    "model_name": "gemma3-chat",  # Alternative name for chat completions
                    "litellm_params": {
                        "model": "vertex_ai_beta/gemma-2-9b-it",
                        "vertex_project": self.project_id,
                        "vertex_location": self.location,
                        "api_base": endpoint_url,
                        "custom_llm_provider": "vertex_ai"
                    }
                }
            ],
            "general_settings": {
                "master_key": "sk-1234567890abcdef",  # Set your API key
                "database_url": None,  # Optional: add Redis/PostgreSQL for logging
            },
            "litellm_settings": {
                "drop_params": True,  # Drop unsupported parameters
                "set_verbose": True,   # Enable detailed logging
                "request_timeout": 60,
                "telemetry": False
            }
        }
        
        # Save config to file
        with open("litellm_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✓ Created LiteLLM configuration file")
        return config

def start_server():
    """Start the LiteLLM proxy server"""
    wrapper = VertexAIGemmaWrapper()
    
    print("\n" + "="*60)
    print("🚀 Starting OpenAI-Compatible API Server for Vertex AI Gemma3")
    print("="*60)
    print(f"📍 Server will run on: http://localhost:8000")
    print(f"📋 API Base URL: http://localhost:8000/v1")
    print(f"🔑 API Key: sk-1234567890abcdef")
    print(f"📚 Available Models: gemma3, gemma3-chat")
    print("\n📖 Example Usage:")
    print("curl -X POST http://localhost:8000/v1/chat/completions \\")
    print("  -H 'Authorization: Bearer sk-1234567890abcdef' \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "model": "gemma3",')
    print('    "messages": [{"role": "user", "content": "Hello!"}],')
    print('    "max_tokens": 150')
    print("  }'")
    print("="*60)
    
    # Start LiteLLM proxy server
    try:
        os.system("litellm --config litellm_config.yaml --port 8000 --host 0.0.0.0")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    start_server()

# Alternative: Custom FastAPI Implementation
"""
If you prefer a more customized approach, here's a FastAPI-based solution:

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from your_vertex_client import VertexAIGemma3Client  # Your existing client

app = FastAPI(title="OpenAI-Compatible API for Vertex AI Gemma3")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

def verify_api_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    api_key = authorization.split(" ")[1]
    if api_key != "sk-1234567890abcdef":  # Your API key
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Initialize your Vertex AI client
        client = VertexAIGemma3Client(
            project_id="your-project-id",
            location="your-location", 
            endpoint_id="your-endpoint-id"
        )
        
        # Convert messages to prompt
        prompt = ""
        for msg in request.messages:
            prompt += f"{msg.role.title()}: {msg.content}\n"
        prompt += "Assistant:"
        
        # Generate response
        response_text = client.generate_text(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Format OpenAI-compatible response
        return {
            "id": f"chatcmpl-{hash(prompt) % 10**10}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma3",
                "object": "model",
                "created": 1677610602,
                "owned_by": "vertex-ai"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""