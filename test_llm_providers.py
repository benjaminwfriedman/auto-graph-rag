#!/usr/bin/env python3
"""Test script for LLM provider integration."""

import os
import sys
from pathlib import Path
import logging

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_graph_rag.llm.providers import LLMProviderFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_provider_creation():
    """Test creating different providers."""
    print("\n=== Testing Provider Creation ===\n")
    
    # Test auto-detection
    print("1. Testing auto provider:")
    try:
        provider = LLMProviderFactory.create(provider="auto")
        print(f"   ✅ Created provider: {provider.__class__.__name__}")
        print(f"   Model: {provider.get_model_name()}")
        print(f"   Supports streaming: {provider.supports_streaming}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test OpenAI if key is available
    if os.getenv("OPENAI_API_KEY"):
        print("\n2. Testing OpenAI provider:")
        try:
            provider = LLMProviderFactory.create(provider="openai", model="gpt-3.5-turbo")
            print(f"   ✅ Created provider: {provider.__class__.__name__}")
            print(f"   Model: {provider.get_model_name()}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print("\n2. Skipping OpenAI test (no API key)")
    
    # Test local provider with small model
    print("\n3. Testing local provider (small model):")
    try:
        # Try with a very small model that's more likely to work
        provider = LLMProviderFactory.create(
            provider="local", 
            model="microsoft/phi-2",  # 2.7B model, small and efficient
            device="cpu"  # Force CPU for testing
        )
        print(f"   ✅ Created provider: {provider.__class__.__name__}")
        print(f"   Model: {provider.get_model_name()}")
    except Exception as e:
        print(f"   ⚠️  Failed (this is expected if model not downloaded): {e}")


def test_basic_chat():
    """Test basic chat functionality."""
    print("\n=== Testing Basic Chat ===\n")
    
    # Simple test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
    ]
    
    # Test with auto provider
    try:
        provider = LLMProviderFactory.create(provider="auto")
        print(f"Testing chat with {provider.get_model_name()}...")
        
        response = provider.chat(messages)
        print(f"Response: {response}")
        print("✅ Chat test passed")
    except Exception as e:
        print(f"❌ Chat test failed: {e}")


def test_structured_output():
    """Test structured JSON output."""
    print("\n=== Testing Structured Output ===\n")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that returns JSON."},
        {"role": "user", "content": "List 3 colors as a JSON array of strings."}
    ]
    
    try:
        provider = LLMProviderFactory.create(provider="auto")
        print(f"Testing structured output with {provider.get_model_name()}...")
        
        response = provider.chat_structured(messages)
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        if isinstance(response, (list, dict)):
            print("✅ Structured output test passed")
        else:
            print("⚠️  Response is not structured as expected")
    except Exception as e:
        print(f"❌ Structured output test failed: {e}")


def test_with_modules():
    """Test integration with GraphExplorer and DataGenerator."""
    print("\n=== Testing Module Integration ===\n")
    
    try:
        from auto_graph_rag.modules import GraphExplorer, DataGenerator
        
        # Test GraphExplorer
        print("1. Testing GraphExplorer with auto provider:")
        explorer = GraphExplorer(llm_provider="auto")
        print(f"   ✅ GraphExplorer initialized")
        
        # Test DataGenerator
        print("\n2. Testing DataGenerator with auto provider:")
        generator = DataGenerator(llm_provider="auto")
        print(f"   ✅ DataGenerator initialized")
        
    except Exception as e:
        print(f"❌ Module integration test failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LLM Provider Integration Tests")
    print("="*60)
    
    # List available models
    print("\n=== Available Models ===\n")
    models = LLMProviderFactory.list_available_models()
    for provider, model_list in models.items():
        print(f"{provider}:")
        for model in model_list:
            print(f"  - {model}")
    
    # Run tests
    test_provider_creation()
    test_basic_chat()
    test_structured_output()
    test_with_modules()
    
    print("\n" + "="*60)
    print("Tests Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()