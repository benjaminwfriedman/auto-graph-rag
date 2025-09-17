"""Query engine for executing natural language queries."""

from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


class QueryEngine:
    """Execute natural language queries using fine-tuned model."""
    
    def __init__(
        self,
        model_path: str,
        kuzu_adapter,
        device: str = "auto",
        model_name: Optional[str] = None
    ):
        """Initialize query engine.
        
        Args:
            model_path: Path to fine-tuned model
            kuzu_adapter: KuzuAdapter instance for executing queries
            device: Device to run model on
            model_name: Original model name for format detection
        """
        self.model_path = Path(model_path)
        self.kuzu_adapter = kuzu_adapter
        self.device = device
        self.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device if self.device == "auto" else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        if self.device != "auto":
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def query(
        self,
        question: str,
        max_retries: int = 3,
        temperature: float = 0.1,
        return_cypher: bool = True,
        format_results: bool = True
    ) -> Dict[str, Any]:
        """Execute a natural language query.
        
        Args:
            question: Natural language question
            max_retries: Maximum retries for failed queries
            temperature: Generation temperature
            return_cypher: Whether to return generated Cypher
            format_results: Whether to format results
            
        Returns:
            Query results and metadata
        """
        # Generate Cypher query
        cypher = self._generate_cypher(question, temperature)
        
        # Clean the generated query
        cypher = self._clean_cypher(cypher)
        
        # Try to execute the query
        results = None
        error = None
        attempts = 0
        
        while attempts < max_retries and results is None:
            try:
                results = self.kuzu_adapter.execute_cypher(cypher)
                break
            except Exception as e:
                error = str(e)
                logger.warning(f"Query failed (attempt {attempts + 1}): {error}")
                
                # Try to fix common issues
                if attempts < max_retries - 1:
                    cypher = self._fix_cypher_errors(cypher, error)
                
                attempts += 1
        
        # Build response
        response = {
            "question": question,
            "success": results is not None,
            "attempts": attempts
        }
        
        if return_cypher:
            response["cypher"] = cypher
        
        if results is not None:
            response["results"] = results
            response["count"] = len(results)
            
            if format_results:
                from .result_formatter import ResultFormatter
                formatter = ResultFormatter()
                response["formatted"] = formatter.format_results(
                    results, question, cypher
                )
        else:
            response["error"] = error
        
        return response
    
    def _generate_cypher(
        self,
        question: str,
        temperature: float = 0.1
    ) -> str:
        """Generate Cypher query from question.
        
        Args:
            question: Natural language question
            temperature: Generation temperature
            
        Returns:
            Generated Cypher query
        """
        # Check if we should use Llama 3.1 format
        use_llama_31_format = (
            self.model_name and ("Llama-3.1" in self.model_name or "Llama-3.2" in self.model_name) and "Instruct" in self.model_name
        )
        
        if use_llama_31_format:
            # Format prompt using Llama 3.1 Instruct format
            system_prompt = "You are an expert at generating Cypher queries for Neo4j graph databases. Generate only valid Cypher queries with no explanation."
            user_prompt = f"Generate a Cypher query to answer this question: {question}"
            
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Use simple format for TinyLlama and other models
            prompt = f"Generate only a Cypher query to answer this question. Return only the query, no explanation.\n\nQuestion: {question}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to same device as model
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Auto-detect model device and move inputs there
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=0.95,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    def _clean_cypher(self, cypher: str) -> str:
        """Clean generated Cypher query.
        
        Args:
            cypher: Raw generated query
            
        Returns:
            Cleaned query
        """
        # Remove markdown code blocks
        cypher = re.sub(r'```(?:cypher)?\n?', '', cypher)
        cypher = re.sub(r'```$', '', cypher)
        
        # Extract just the Cypher query part
        lines = cypher.split('\n')
        query_lines = []
        in_query = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip explanation text
            if any(line.startswith(prefix) for prefix in [
                'This query', 'The query', 'Explanation:', 'Output:', 'Answer:', 
                'This Cypher query', 'The Cypher query', 'Result:'
            ]):
                continue
                
            # Look for Cypher keywords to identify actual query
            cypher_keywords = ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'RETURN', 'WITH', 'WHERE']
            if any(keyword in line.upper() for keyword in cypher_keywords):
                in_query = True
                query_lines.append(line)
            elif in_query:
                # Continue collecting query lines until we hit explanation text
                if not any(line.startswith(prefix) for prefix in ['This', 'The', 'Note:', '//']):
                    query_lines.append(line)
                else:
                    break
        
        # If no proper query found, try to extract from the first substantial line
        if not query_lines:
            for line in lines:
                line = line.strip()
                if line and not any(line.startswith(prefix) for prefix in [
                    'This', 'The', 'Here', 'To', 'You', 'I', 'We', 'Note:', 'Output:', 'Answer:'
                ]):
                    query_lines.append(line)
                    break
        
        result = ' '.join(query_lines).strip()
        
        # Final cleanup - remove any trailing explanations
        if ';' in result:
            result = result.split(';')[0] + ';'
        
        return result
    
    def _fix_cypher_errors(self, cypher: str, error: str) -> str:
        """Attempt to fix common Cypher errors.
        
        Args:
            cypher: Failed query
            error: Error message
            
        Returns:
            Potentially fixed query
        """
        error_lower = error.lower()
        
        # Fix missing RETURN clause
        if "return" not in cypher.upper() and "match" in cypher.upper():
            cypher += " RETURN *"
        
        # Fix property access syntax
        if "property" in error_lower:
            # Fix node.property to node['property'] syntax if needed
            cypher = re.sub(r'(\w+)\.(\w+)', r"\1['\2']", cypher)
        
        # Fix table/label names
        if "table" in error_lower or "label" in error_lower:
            # Try to extract the problematic label and fix case
            import re
            match = re.search(r"'(\w+)'", error)
            if match:
                bad_label = match.group(1)
                # Try different case variations
                for variant in [bad_label.lower(), bad_label.upper(), bad_label.capitalize()]:
                    if variant != bad_label:
                        cypher = cypher.replace(bad_label, variant)
                        break
        
        return cypher
    
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute multiple queries.
        
        Args:
            questions: List of questions
            **kwargs: Additional arguments for query()
            
        Returns:
            List of query results
        """
        results = []
        for question in questions:
            result = self.query(question, **kwargs)
            results.append(result)
        
        return results