"""Unified LLM provider system for Auto-Graph-RAG."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Basic chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional provider-specific arguments
            
        Returns:
            String response from the model
        """
        pass
    
    @abstractmethod
    def chat_structured(self, messages: List[Dict[str, str]], 
                       output_schema: Optional[Any] = None, **kwargs) -> Union[Dict, List]:
        """Chat with structured output (JSON mode).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            output_schema: Optional schema for structured output
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Parsed JSON response as dict or list
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model being used."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider using LangChain."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, 
                 temperature: float = 0.0, **kwargs):
        """Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name
            api_key: API key (uses env var if not provided)
            temperature: Temperature for generation
            **kwargs: Additional OpenAI parameters
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai required for OpenAI provider. Install with: pip install langchain-openai")
        
        self.model = model
        self.temperature = temperature
        
        # Check for API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
        
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Basic chat completion using OpenAI."""
        try:
            # Convert to LangChain message format
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            
            lc_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            
            response = self.llm.invoke(lc_messages, **kwargs)
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    def chat_structured(self, messages: List[Dict[str, str]], 
                       output_schema: Optional[Any] = None, **kwargs) -> Union[Dict, List]:
        """Chat with structured JSON output."""
        try:
            # Add JSON instruction to last message
            messages_copy = messages.copy()
            if messages_copy:
                messages_copy[-1]["content"] += "\n\nReturn your response as valid JSON."
            
            # Use JSON mode if available (GPT-4 Turbo and later)
            if "gpt-4" in self.model or "gpt-3.5-turbo" in self.model:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.chat(messages_copy, **kwargs)
            
            # Parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError(f"Could not parse JSON from response: {response}")
                
        except Exception as e:
            logger.error(f"OpenAI structured chat error: {e}")
            raise
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model
    
    @property
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True


class LocalLlamaProvider(BaseLLMProvider):
    """Local Llama model provider using transformers or llama-cpp."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
                 device: str = "auto", quantization: str = "auto",
                 use_llama_cpp: bool = False, cache_dir: Optional[str] = None,
                 temperature: float = 0.0, **kwargs):
        """Initialize local Llama provider.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto, cuda, mps, cpu)
            quantization: Quantization mode (auto, 4bit, 8bit, none)
            use_llama_cpp: Use llama.cpp backend for CPU efficiency
            cache_dir: Cache directory for models
            temperature: Temperature for generation
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.use_llama_cpp = use_llama_cpp
        self.temperature = temperature
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "auto-graph-rag" / "models")
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        if use_llama_cpp:
            self._setup_llama_cpp(**kwargs)
        else:
            self._setup_transformers(**kwargs)
        
        logger.info(f"Initialized Local Llama provider with model: {model_name} on device: {self.device}")
    
    def _setup_transformers(self, **kwargs):
        """Setup using HuggingFace transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise ImportError("transformers and torch required for local models. Install with: pip install transformers torch")
        
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("🎮 CUDA GPU detected")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("🍎 Apple Silicon GPU detected")
            else:
                self.device = "cpu"
                logger.warning("⚠️ No GPU detected, using CPU (this will be slow)")
        
        # Get model loading kwargs
        model_kwargs = self._get_model_kwargs()
        model_kwargs.update(kwargs)
        
        logger.info(f"📥 Loading model {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **model_kwargs
        )
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            device_map="auto" if self.device != "cpu" else None
        )
        
        logger.info(f"✅ Model loaded successfully")
    
    def _setup_llama_cpp(self, **kwargs):
        """Setup using llama.cpp for efficient CPU inference."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required for llama.cpp backend. Install with: pip install llama-cpp-python")
        
        # For llama.cpp, we need GGUF format models
        # This is a simplified version - in production, you'd download/convert GGUF models
        model_path = self._get_gguf_model_path()
        
        n_gpu_layers = -1 if self.device in ["cuda", "mps"] else 0
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **kwargs
        )
        
        logger.info(f"✅ Llama.cpp model loaded from {model_path}")
    
    def _get_model_kwargs(self):
        """Get model loading kwargs based on device and memory."""
        import torch
        
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # Handle quantization
        if self.quantization == "auto":
            # Auto-detect based on model size and available memory
            if self.device == "cuda":
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                if "70b" in self.model_name.lower():
                    # 70B models need aggressive quantization
                    if mem_gb < 80:
                        try:
                            import bitsandbytes
                            kwargs["load_in_4bit"] = True
                            kwargs["bnb_4bit_compute_dtype"] = torch.float16
                            logger.info("Using 4-bit quantization for 70B model")
                        except ImportError:
                            logger.warning("bitsandbytes not available, loading in fp16")
                            kwargs["torch_dtype"] = torch.float16
                elif "8b" in self.model_name.lower() or "7b" in self.model_name.lower():
                    if mem_gb < 16:
                        try:
                            import bitsandbytes
                            kwargs["load_in_8bit"] = True
                            logger.info("Using 8-bit quantization for 7B/8B model")
                        except ImportError:
                            kwargs["torch_dtype"] = torch.float16
                    else:
                        kwargs["torch_dtype"] = torch.float16
                else:
                    # Smaller models
                    kwargs["torch_dtype"] = torch.float16
            
            elif self.device == "mps":
                # Apple Silicon
                kwargs["torch_dtype"] = torch.float16
            
            else:
                # CPU - use smaller dtype for efficiency
                kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        elif self.quantization == "4bit":
            try:
                import bitsandbytes
                kwargs["load_in_4bit"] = True
                kwargs["bnb_4bit_compute_dtype"] = torch.float16
            except ImportError:
                logger.warning("bitsandbytes not installed, cannot use 4-bit quantization")
        
        elif self.quantization == "8bit":
            try:
                import bitsandbytes
                kwargs["load_in_8bit"] = True
            except ImportError:
                logger.warning("bitsandbytes not installed, cannot use 8-bit quantization")
        
        else:
            # No quantization
            kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
        
        return kwargs
    
    def _get_gguf_model_path(self) -> Path:
        """Get or download GGUF model for llama.cpp."""
        # Simplified - in production, implement proper GGUF model downloading
        # For now, just return a path where the model would be
        gguf_name = self.model_name.replace("/", "_") + ".gguf"
        model_path = Path(self.cache_dir) / "gguf" / gguf_name
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {model_path}. "
                f"Please download a GGUF version of {self.model_name} "
                f"or use transformers backend instead."
            )
        
        return model_path
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Basic chat completion using local model."""
        try:
            if self.use_llama_cpp:
                # Format messages for llama.cpp
                prompt = self._format_chat_prompt(messages)
                response = self.model(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", self.temperature),
                    stop=kwargs.get("stop", None)
                )
                return response["choices"][0]["text"]
            
            else:
                # Use transformers pipeline
                prompt = self._format_chat_prompt(messages)
                
                # Generate response
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", self.temperature),
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                return outputs[0]["generated_text"]
                
        except Exception as e:
            logger.error(f"Local Llama chat error: {e}")
            raise
    
    def chat_structured(self, messages: List[Dict[str, str]], 
                       output_schema: Optional[Any] = None, **kwargs) -> Union[Dict, List]:
        """Chat with structured JSON output."""
        try:
            # Add JSON instruction to messages
            messages_copy = messages.copy()
            if messages_copy:
                messages_copy.append({
                    "role": "system",
                    "content": "You must return your response as valid JSON only. No other text."
                })
            
            response = self.chat(messages_copy, **kwargs)
            
            # Parse JSON response
            try:
                # Clean response - remove markdown code blocks if present
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                
                return json.loads(response.strip())
                
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                logger.error(f"Could not parse JSON from response: {response}")
                raise ValueError(f"Could not parse JSON from response")
                
        except Exception as e:
            logger.error(f"Local Llama structured chat error: {e}")
            raise
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a chat prompt for the model."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use the tokenizer's chat template if available
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Fallback to simple formatting
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name
    
    @property
    def supports_streaming(self) -> bool:
        """Transformers supports streaming, llama.cpp has limited support."""
        return not self.use_llama_cpp


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    # Model size recommendations
    MODEL_RECOMMENDATIONS = {
        "small": "meta-llama/Llama-3.2-1B-Instruct",  # Fast, minimal resources
        "medium": "meta-llama/Llama-3.2-3B-Instruct",  # Balanced
        "large": "meta-llama/Llama-3.1-8B-Instruct",   # High quality
        "xlarge": "meta-llama/Llama-3.1-70B-Instruct"  # Best quality, high resources
    }
    
    @staticmethod
    def create(provider: str = "auto", model: Optional[str] = None, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider.
        
        Args:
            provider: Provider type - "openai", "local", or "auto"
            model: Model name/identifier or size (small/medium/large/xlarge)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLM provider instance
        """
        # Handle model size shortcuts
        if model in LLMProviderFactory.MODEL_RECOMMENDATIONS:
            model = LLMProviderFactory.MODEL_RECOMMENDATIONS[model]
            logger.info(f"Using recommended model for '{model}' size: {model}")
        
        if provider == "auto":
            # Auto-detect best provider
            if os.getenv("OPENAI_API_KEY"):
                logger.info("🔑 OpenAI API key found, using OpenAI provider")
                return OpenAIProvider(model or "gpt-4", **kwargs)
            else:
                logger.info("🦙 No OpenAI key found, using local Llama model")
                logger.info("💡 Tip: Set OPENAI_API_KEY environment variable to use OpenAI models")
                
                # Default to smaller model for local if not specified
                if not model:
                    model = LLMProviderFactory.MODEL_RECOMMENDATIONS["medium"]
                    logger.info(f"No model specified, defaulting to {model}")
                
                return LocalLlamaProvider(model, **kwargs)
        
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY") and "api_key" not in kwargs:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            return OpenAIProvider(model or "gpt-4", **kwargs)
        
        elif provider == "local":
            if not model:
                model = LLMProviderFactory.MODEL_RECOMMENDATIONS["medium"]
                logger.info(f"No model specified, defaulting to {model}")
            return LocalLlamaProvider(model, **kwargs)
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'local', or 'auto'")
    
    @staticmethod
    def list_available_models() -> Dict[str, List[str]]:
        """List available models for each provider."""
        return {
            "openai": [
                "gpt-4",
                "gpt-4-turbo-preview", 
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ],
            "local": [
                "meta-llama/Llama-3.2-1B-Instruct (small)",
                "meta-llama/Llama-3.2-3B-Instruct (medium)",
                "meta-llama/Llama-3.1-8B-Instruct (large)",
                "meta-llama/Llama-3.1-70B-Instruct (xlarge)",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "microsoft/phi-2"
            ],
            "shortcuts": list(LLMProviderFactory.MODEL_RECOMMENDATIONS.keys())
        }