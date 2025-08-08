#!/usr/bin/env python3
"""
Python application for interacting with Gemma3 model deployed on Vertex AI Model Garden
as a custom model endpoint.

Prerequisites:
1. Install dependencies: pip install google-auth google-auth-oauthlib google-auth-httplib2 requests
2. Set up Google Cloud authentication (service account or application default credentials)
3. Configure your project settings below
"""

import json
import requests
import sys
from typing import Dict, Any, Optional
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth

class VertexAIGemma3Client:
    def __init__(self, project_id: str, location: str, endpoint_id: str, credentials_path: Optional[str] = None):
        """
        Initialize the Vertex AI Gemma3 client
        
        Args:
            project_id: Google Cloud project ID
            location: Region where the endpoint is deployed (e.g., 'us-central1')
            endpoint_id: The endpoint ID from Model Garden deployment
            credentials_path: Path to service account JSON file (optional if using ADC)
        """
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id
        
        # Construct the endpoint URL
        self.endpoint_url = (
            f"https://{endpoint_id}.{location}-{project_id}.prediction.vertexai.goog"
            f"/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        )
        
        # Initialize credentials
        self.credentials = self._setup_credentials(credentials_path)
        
    def _setup_credentials(self, credentials_path: Optional[str]):
        """Setup Google Cloud credentials"""
        try:
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                # Use Application Default Credentials
                credentials, _ = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            return credentials
        except Exception as e:
            print(f"Error setting up credentials: {e}")
            sys.exit(1)
    
    def _get_access_token(self) -> str:
        """Get a valid access token"""
        self.credentials.refresh(Request())
        return self.credentials.token
    
    def _create_request_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Create request payload for Gemma3 model
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters like max_tokens, temperature, etc.
        """
        # Default parameters optimized for Gemma3
        parameters = {
            "max_output_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
        }
        
        # Instance payload - format may vary based on your specific deployment
        instance = {
            "inputs": prompt,
            # Alternative formats that might be used:
            # "prompt": prompt,
            # "text": prompt,
        }
        
        payload = {
            "instances": [instance],
            "parameters": parameters
        }
        
        return payload
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Gemma3 model
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Get access token
            access_token = self._get_access_token()
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            payload = self._create_request_payload(prompt, **kwargs)
            
            # Make request
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return self._parse_response(response.json())
            else:
                raise Exception(f"Request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error: {e}"
    
    def _parse_response(self, response_data: Dict) -> str:
        """
        Parse the response from Vertex AI endpoint
        The response format may vary depending on how the model was deployed
        """
        try:
            predictions = response_data.get("predictions", [])
            if not predictions:
                return "No predictions in response"
            
            prediction = predictions[0]
            
            # Try different possible response formats
            if isinstance(prediction, str):
                return prediction
            elif isinstance(prediction, dict):
                # Common response field names
                for field in ["generated_text", "output", "text", "response", "content"]:
                    if field in prediction:
                        return prediction[field]
                
                # If no standard field found, return the first string value
                for value in prediction.values():
                    if isinstance(value, str):
                        return value
            
            # Fallback: return the raw prediction
            return str(prediction)
            
        except Exception as e:
            return f"Error parsing response: {e}\nRaw response: {response_data}"
    
    def summarize_text(self, text: str) -> str:
        """Summarize the given text"""
        prompt = f"""Please provide a concise summary of the following text:

{text}

Summary:"""
        return self.generate_text(prompt, max_tokens=150, temperature=0.3)
    
    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on the given context"""
        prompt = f"""Based on the following context, please answer the question:

Context: {context}

Question: {question}

Answer:"""
        return self.generate_text(prompt, max_tokens=200, temperature=0.2)
    
    def chat(self, message: str, conversation_history: str = "") -> str:
        """Have a conversation with the model"""
        if conversation_history:
            prompt = f"{conversation_history}\nUser: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"
        
        return self.generate_text(prompt, max_tokens=300, temperature=0.8)

def main():
    """Main application loop"""
    print("=== Vertex AI Gemma3 Model Garden Application ===\n")
    
    # Configuration - Replace with your actual values
    PROJECT_ID = "your-project-id"
    LOCATION = "your-location"  # e.g., "us-central1"
    ENDPOINT_ID = "your-endpoint-id"
    CREDENTIALS_PATH = "path/to/your/service-account-key.json"  # Optional if using ADC
    
    # Validate configuration
    if any(val.startswith("your-") for val in [PROJECT_ID, LOCATION, ENDPOINT_ID]):
        print("Please configure PROJECT_ID, LOCATION, and ENDPOINT_ID in the code.")
        sys.exit(1)
    
    # Initialize client
    try:
        client = VertexAIGemma3Client(
            project_id=PROJECT_ID,
            location=LOCATION,
            endpoint_id=ENDPOINT_ID,
            credentials_path=CREDENTIALS_PATH if CREDENTIALS_PATH != "path/to/your/service-account-key.json" else None
        )
        print("✓ Successfully connected to Vertex AI endpoint")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        sys.exit(1)
    
    conversation_history = ""
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Generate text")
        print("2. Summarize text")
        print("3. Question & Answer")
        print("4. Chat conversation")
        print("5. Test connection")
        print("6. Exit")
        print("="*50)
        
        choice = input("Choose an option (1-6): ").strip()
        
        if choice == "1":
            print("\nEnter your prompt:")
            prompt = input("> ")
            if prompt.strip():
                print("\nGenerating...")
                result = client.generate_text(prompt)
                print(f"\nResponse:\n{result}")
        
        elif choice == "2":
            print("\nEnter text to summarize:")
            text = input("> ")
            if text.strip():
                print("\nSummarizing...")
                summary = client.summarize_text(text)
                print(f"\nSummary:\n{summary}")
        
        elif choice == "3":
            print("\nEnter the context:")
            context = input("> ")
            print("\nEnter your question:")
            question = input("> ")
            if context.strip() and question.strip():
                print("\nProcessing...")
                answer = client.answer_question(context, question)
                print(f"\nAnswer:\n{answer}")
        
        elif choice == "4":
            print("\nChat mode - type 'quit' to exit chat")
            while True:
                message = input("\nYou: ")
                if message.lower() == 'quit':
                    break
                if message.strip():
                    response = client.chat(message, conversation_history)
                    print(f"Assistant: {response}")
                    conversation_history += f"\nUser: {message}\nAssistant: {response}"
        
        elif choice == "5":
            print("\nTesting connection...")
            test_response = client.generate_text("Hello, how are you?", max_tokens=50)
            print(f"Test response: {test_response}")
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please choose 1-6.")

if __name__ == "__main__":
    main()
