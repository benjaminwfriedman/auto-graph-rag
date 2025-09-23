"""Standalone module for generating training data."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import argparse
import logging
import os

from .base import DataGeneratorInterface
from ..generation.question_generator import QuestionGenerator as LLMQuestionGen
from ..training.dataset_builder import DatasetBuilder
from ..ingestion.kuzu_adapter import KuzuAdapter


logger = logging.getLogger(__name__)


class DataGenerator(DataGeneratorInterface):
    """Generate training data for graph query models."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4"
    ):
        """Initialize data generator.
        
        Args:
            llm_provider: LLM provider for generation
            llm_model: Model name for generation
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.question_gen = LLMQuestionGen(llm_provider, llm_model)
        self.dataset_builder = DatasetBuilder()
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        if 'schema_path' not in kwargs and 'schema' not in kwargs:
            raise ValueError("Either schema_path or schema is required")
        
        if 'num_examples' not in kwargs:
            raise ValueError("num_examples is required")
        
        if 'schema_path' in kwargs:
            schema_path = Path(kwargs['schema_path'])
            if not schema_path.exists():
                raise ValueError(f"Schema file not found: {schema_path}")
        
        return True
    
    def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Execute data generation."""
        if 'schema_path' in kwargs:
            return self.generate_from_schema(**kwargs)
        else:
            return self.generate_from_dict(**kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            "name": "DataGenerator",
            "version": "1.0.0",
            "description": "Generate training data from graph schemas",
            "inputs": {
                "from_schema": ["schema_path", "num_examples", "output_path", "complexity_distribution", "db_path"],
                "from_dict": ["schema", "num_examples", "output_path", "complexity_distribution", "kuzu_adapter"]
            },
            "outputs": ["dataset", "statistics"]
        }
    
    def generate_from_schema(
        self,
        schema_path: Path,
        num_examples: int,
        output_path: Path,
        complexity_distribution: Optional[Dict[int, float]] = None,
        db_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Generate training data from schema file.
        
        Args:
            schema_path: Path to schema JSON file
            num_examples: Number of examples to generate
            output_path: Path to save dataset
            complexity_distribution: Distribution of query complexities
            db_path: Optional database path for validation
            
        Returns:
            Generated dataset
        """
        schema_path = Path(schema_path)
        logger.info(f"Loading schema from {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Create adapter if database provided for validation
        kuzu_adapter = None
        if db_path:
            db_path = Path(db_path)
            if db_path.exists():
                logger.info(f"Using database at {db_path} for validation")
                kuzu_adapter = KuzuAdapter(db_path, create_new=False)
        
        return self.generate_from_dict(
            schema=schema,
            num_examples=num_examples,
            output_path=output_path,
            complexity_distribution=complexity_distribution,
            kuzu_adapter=kuzu_adapter
        )
    
    def generate_from_dict(
        self,
        schema: Dict[str, Any],
        num_examples: int,
        output_path: Optional[Path] = None,
        complexity_distribution: Optional[Dict[int, float]] = None,
        kuzu_adapter: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate training data from schema dictionary.
        
        Args:
            schema: Schema dictionary
            num_examples: Number of examples to generate
            output_path: Optional path to save dataset
            complexity_distribution: Distribution of query complexities
            kuzu_adapter: Optional adapter for validation
            
        Returns:
            Generated dataset
        """
        logger.info(f"Generating {num_examples} training examples")
        
        # Debug the schema
        node_count = len(schema.get("nodes", {}))
        edge_count = len(schema.get("edges", {}))
        logger.info(f"Schema has {node_count} node types and {edge_count} edge types")
        
        if node_count == 0 and edge_count == 0:
            logger.error("Schema is empty! Cannot generate training data.")
            logger.error(f"Schema content: {schema}")
            return []
        
        if node_count == 0:
            logger.warning("No node types in schema - this may limit data generation")
        
        if edge_count == 0:
            logger.warning("No edge types in schema - this may limit relationship queries")
        
        # Default complexity distribution
        if not complexity_distribution:
            complexity_distribution = {
                1: 0.2,  # Simple lookups
                2: 0.3,  # Filtered queries
                3: 0.3,  # Relationships
                4: 0.15, # Aggregations
                5: 0.05  # Complex paths
            }
        
        logger.info(f"Using complexity distribution: {complexity_distribution}")
        
        # Debug: Check what tables actually exist in the database
        if kuzu_adapter:
            try:
                logger.info("Checking what tables actually exist in the database...")
                actual_schema = kuzu_adapter.get_schema_info()
                actual_node_tables = [t["name"] for t in actual_schema.get("node_tables", [])]
                actual_edge_tables = [t["name"] for t in actual_schema.get("edge_tables", [])]
                logger.info(f"Actual node tables: {actual_node_tables}")
                logger.info(f"Actual edge tables: {actual_edge_tables}")
                
                # Test a simple query to see what works
                if actual_node_tables:
                    test_table = actual_node_tables[0]
                    test_query = f"MATCH (n:{test_table}) RETURN n LIMIT 1"
                    logger.info(f"Testing query: {test_query}")
                    try:
                        result = kuzu_adapter.execute_cypher(test_query)
                        logger.info(f"Test query succeeded, got {len(result)} results")
                    except Exception as e:
                        logger.error(f"Test query failed: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to check actual tables: {e}")
        
        # Check if schema is large and needs batched approach
        edge_count = len(schema.get("edges", {}))
        use_batched = edge_count > 15  # Use batched approach for large schemas
        
        if use_batched:
            logger.info(f"Large schema detected ({edge_count} edge types). Using batched generation approach.")
            # Generate dataset using batched approach
            dataset = self.question_gen.generate_batched(
                schema=schema,
                num_examples=num_examples,
                complexity_distribution=complexity_distribution,
                kuzu_adapter=kuzu_adapter,
                validate_queries=kuzu_adapter is not None,
                include_results=True
            )
        else:
            logger.info(f"Using standard generation approach for {edge_count} edge types.")
            # Generate dataset using standard approach
            dataset = self.question_gen.generate(
                schema=schema,
                num_examples=num_examples,
                complexity_distribution=complexity_distribution,
                kuzu_adapter=kuzu_adapter,
                validate_queries=kuzu_adapter is not None,
                include_results=True
            )
        
        logger.info(f"Generated {len(dataset)} valid examples")
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            self.dataset_builder.save_dataset(dataset, str(output_path))
            logger.info(f"Dataset saved to {output_path}")
        
        return dataset
    
    @classmethod
    def load_dataset(cls, dataset_path: Path) -> List[Dict[str, Any]]:
        """Load a previously generated dataset.
        
        Args:
            dataset_path: Path to dataset file (JSONL)
            
        Returns:
            Dataset list
        """
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    
    def get_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_examples": len(dataset),
            "complexity_distribution": {},
            "intent_distribution": {},
            "avg_question_length": 0,
            "avg_cypher_length": 0
        }
        
        question_lengths = []
        cypher_lengths = []
        
        for example in dataset:
            # Complexity
            complexity = example.get('complexity', 0)
            stats['complexity_distribution'][complexity] = \
                stats['complexity_distribution'].get(complexity, 0) + 1
            
            # Intent
            intent = example.get('intent', 'unknown')
            stats['intent_distribution'][intent] = \
                stats['intent_distribution'].get(intent, 0) + 1
            
            # Lengths
            if 'question' in example:
                question_lengths.append(len(example['question']))
            if 'cypher' in example:
                cypher_lengths.append(len(example['cypher']))
        
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        if cypher_lengths:
            stats['avg_cypher_length'] = sum(cypher_lengths) / len(cypher_lengths)
        
        return stats


def main():
    """CLI entry point for standalone data generation."""
    parser = argparse.ArgumentParser(description='Generate training data from graph schema')
    parser.add_argument('--schema', required=True, help='Schema JSON file path')
    parser.add_argument('--output', required=True, help='Output dataset file path')
    parser.add_argument('--num-examples', type=int, default=100,
                       help='Number of examples to generate')
    parser.add_argument('--db-path', help='Optional database path for validation')
    parser.add_argument('--complexity', help='Complexity distribution as JSON string')
    parser.add_argument('--llm-provider', default='openai',
                       help='LLM provider (default: openai)')
    parser.add_argument('--llm-model', default='gpt-4',
                       help='LLM model (default: gpt-4)')
    parser.add_argument('--stats', action='store_true',
                       help='Print dataset statistics')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Check for API key
    if args.llm_provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    # Parse complexity distribution if provided
    complexity_dist = None
    if args.complexity:
        try:
            complexity_dist = json.loads(args.complexity)
            # Convert string keys to integers
            complexity_dist = {int(k): v for k, v in complexity_dist.items()}
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON for complexity distribution")
            return 1
    
    # Create generator
    generator = DataGenerator(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
    
    # Generate dataset
    dataset = generator.generate_from_schema(
        schema_path=Path(args.schema),
        num_examples=args.num_examples,
        output_path=Path(args.output),
        complexity_distribution=complexity_dist,
        db_path=Path(args.db_path) if args.db_path else None
    )
    
    print(f"✅ Generated {len(dataset)} training examples!")
    print(f"💾 Dataset saved to: {args.output}")
    
    # Print statistics if requested
    if args.stats:
        stats = generator.get_statistics(dataset)
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Avg question length: {stats['avg_question_length']:.1f} chars")
        print(f"  Avg cypher length: {stats['avg_cypher_length']:.1f} chars")
        print(f"  Complexity distribution: {stats['complexity_distribution']}")
        print(f"  Intent distribution: {stats['intent_distribution']}")


if __name__ == "__main__":
    main()