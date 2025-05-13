# ollama_client.py
import requests
from typing import List, Dict, Any, Optional

class OllamaModel:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using the Ollama model"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    
    def embedding(self, text: str) -> List[float]:
        """Get embeddings for the given text using Ollama model"""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text}
        )
        
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

class MultiModelSystem:
    def __init__(self):
        # Initialize models
        self.models = {
            "deepseek": OllamaModel("deepseek-r1:latest"),
            "llama3": OllamaModel("llama3.2:latest"),
            "gemma": OllamaModel("gemma:2b")
        }
    
    def get_model(self, model_name: str) -> OllamaModel:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
