"""Standalone module for executing queries with fine-tuned models."""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import argparse
import logging

from .base import QueryExecutorInterface
from ..inference.query_engine import QueryEngine as InferenceEngine
from ..ingestion.kuzu_adapter import KuzuAdapter


logger = logging.getLogger(__name__)


class QueryExecutor(QueryExecutorInterface):
    """Execute queries using fine-tuned models and graph databases."""
    
    def __init__(self):
        """Initialize query executor."""
        self.query_engine = None
        self.kuzu_adapter = None
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        if 'question' not in kwargs:
            raise ValueError("question is required")
        
        if 'model_path' not in kwargs and 'model' not in kwargs:
            raise ValueError("Either model_path or model instance is required")
        
        if 'db_path' not in kwargs and 'kuzu_adapter' not in kwargs:
            raise ValueError("Either db_path or kuzu_adapter is required")
        
        if 'model_path' in kwargs:
            model_path = Path(kwargs['model_path'])
            if not model_path.exists():
                raise ValueError(f"Model path not found: {model_path}")
        
        if 'db_path' in kwargs:
            db_path = Path(kwargs['db_path'])
            if not db_path.exists():
                raise ValueError(f"Database path not found: {db_path}")
        
        return True
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute query."""
        if 'model_path' in kwargs and 'db_path' in kwargs:
            return self.query_with_model(**kwargs)
        else:
            return self.query_with_instances(**kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            "name": "QueryExecutor",
            "version": "1.0.0",
            "description": "Execute queries with fine-tuned models",
            "inputs": {
                "with_paths": ["question", "model_path", "db_path", "return_cypher", "format_results"],
                "with_instances": ["question", "model", "kuzu_adapter", "return_cypher", "format_results"]
            },
            "outputs": ["success", "cypher", "results", "count", "error"]
        }
    
    def query_with_model(
        self,
        question: str,
        model_path: Path,
        db_path: Path,
        model_name: str = "auto-detect",
        return_cypher: bool = True,
        format_results: bool = True,
        max_retries: int = 2,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Execute query with specified model and database paths.
        
        Args:
            question: Natural language question
            model_path: Path to fine-tuned model
            db_path: Path to Kuzu database
            model_name: Base model name for format detection
            return_cypher: Whether to return the generated Cypher
            format_results: Whether to format the results
            max_retries: Maximum query retries on failure
            temperature: Model temperature for generation
            
        Returns:
            Query results dictionary
        """
        model_path = Path(model_path)
        db_path = Path(db_path)
        
        logger.info(f"Executing query with model at {model_path}")
        logger.info(f"Question: {question}")
        
        # Create database adapter
        kuzu_adapter = KuzuAdapter(db_path, create_new=False)
        
        # Auto-detect model name if needed
        if model_name == "auto-detect":
            model_name = self._detect_model_name(model_path)
        
        # Create query engine
        query_engine = InferenceEngine(
            model_path=str(model_path),
            kuzu_adapter=kuzu_adapter,
            model_name=model_name
        )
        
        # Execute query
        result = query_engine.query(
            question=question,
            max_retries=max_retries,
            temperature=temperature,
            return_cypher=return_cypher,
            format_results=format_results
        )
        
        return result
    
    def query_with_instances(
        self,
        question: str,
        model: Any,
        kuzu_adapter: Any,
        return_cypher: bool = True,
        format_results: bool = True,
        max_retries: int = 2,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Execute query with existing model and adapter instances.
        
        Args:
            question: Natural language question
            model: Loaded model instance
            kuzu_adapter: KuzuAdapter instance
            return_cypher: Whether to return the generated Cypher
            format_results: Whether to format the results
            max_retries: Maximum query retries on failure
            temperature: Model temperature for generation
            
        Returns:
            Query results dictionary
        """
        logger.info(f"Executing query with model instance")
        logger.info(f"Question: {question}")
        
        # Create query engine with instances
        # Note: This would require modifications to QueryEngine to accept instances
        # For now, we'll raise an error indicating this needs implementation
        raise NotImplementedError(
            "Query execution with model instances requires QueryEngine modifications. "
            "Use query_with_model instead."
        )
    
    def batch_query(
        self,
        questions: list[str],
        model_path: Path,
        db_path: Path,
        model_name: str = "auto-detect",
        return_cypher: bool = True,
        format_results: bool = True,
        max_retries: int = 2,
        temperature: float = 0.1
    ) -> list[Dict[str, Any]]:
        """Execute multiple queries in batch.
        
        Args:
            questions: List of natural language questions
            model_path: Path to fine-tuned model
            db_path: Path to Kuzu database
            model_name: Base model name for format detection
            return_cypher: Whether to return the generated Cypher
            format_results: Whether to format the results
            max_retries: Maximum query retries on failure
            temperature: Model temperature for generation
            
        Returns:
            List of query results
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing query {i}/{len(questions)}")
            try:
                result = self.query_with_model(
                    question=question,
                    model_path=model_path,
                    db_path=db_path,
                    model_name=model_name,
                    return_cypher=return_cypher,
                    format_results=format_results,
                    max_retries=max_retries,
                    temperature=temperature
                )
                results.append({
                    "question": question,
                    "index": i,
                    **result
                })
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.append({
                    "question": question,
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def _detect_model_name(self, model_path: Path) -> str:
        """Detect the base model name from the fine-tuned model directory.
        
        Args:
            model_path: Path to fine-tuned model
            
        Returns:
            Detected model name
        """
        # Try to read config.json
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Look for base model in various config fields
                    for field in ['_name_or_path', 'model_name', 'base_model']:
                        if field in config:
                            return config[field]
            except Exception as e:
                logger.warning(f"Could not read config.json: {e}")
        
        # Fallback: try to infer from directory name
        dir_name = model_path.name.lower()
        if 'llama' in dir_name:
            if '3.2' in dir_name and '1b' in dir_name:
                return "meta-llama/Llama-3.2-1B-Instruct"
            elif '3.2' in dir_name and '3b' in dir_name:
                return "meta-llama/Llama-3.2-3B-Instruct"
        
        # Default fallback
        logger.warning(f"Could not detect model name, using default")
        return "meta-llama/Llama-3.2-1B-Instruct"


def main():
    """CLI entry point for standalone query execution."""
    parser = argparse.ArgumentParser(description='Execute queries with fine-tuned model')
    parser.add_argument('--model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--db', required=True, help='Path to Kuzu database')
    parser.add_argument('--question', help='Single question to ask')
    parser.add_argument('--questions-file', help='File with questions (one per line)')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--model-name', default='auto-detect',
                       help='Base model name (default: auto-detect)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Model temperature')
    parser.add_argument('--max-retries', type=int, default=2,
                       help='Max retries per question')
    parser.add_argument('--no-cypher', action='store_true',
                       help='Don\'t return generated Cypher')
    parser.add_argument('--no-format', action='store_true',
                       help='Don\'t format results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if not args.question and not args.questions_file:
        print("❌ Error: Either --question or --questions-file is required")
        return 1
    
    # Create executor
    executor = QueryExecutor()
    
    # Prepare questions
    if args.question:
        questions = [args.question]
    else:
        with open(args.questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
    
    print(f"🤖 Executing {len(questions)} question(s) with model at: {args.model}")
    print(f"📊 Database: {args.db}")
    
    # Execute queries
    if len(questions) == 1:
        result = executor.query_with_model(
            question=questions[0],
            model_path=Path(args.model),
            db_path=Path(args.db),
            model_name=args.model_name,
            return_cypher=not args.no_cypher,
            format_results=not args.no_format,
            max_retries=args.max_retries,
            temperature=args.temperature
        )
        results = [{"question": questions[0], **result}]
    else:
        results = executor.batch_query(
            questions=questions,
            model_path=Path(args.model),
            db_path=Path(args.db),
            model_name=args.model_name,
            return_cypher=not args.no_cypher,
            format_results=not args.no_format,
            max_retries=args.max_retries,
            temperature=args.temperature
        )
    
    # Print results
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Q: {result['question']}")
        if result.get('success'):
            if 'cypher' in result:
                cypher = result['cypher'].replace('\n', ' ').strip()
                if len(cypher) > 100:
                    cypher = cypher[:100] + "..."
                print(f"    🔍 Cypher: {cypher}")
            print(f"    📊 Results: {result.get('count', 0)} rows")
        else:
            error = result.get('error', 'Unknown error')
            print(f"    ❌ Error: {error[:100]}...")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()