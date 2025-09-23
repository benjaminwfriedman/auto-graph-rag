"""Standalone module for exploring and understanding graph schemas."""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import argparse
import logging
import os

from .base import GraphExplorerInterface
from ..ingestion.kuzu_adapter import KuzuAdapter
from ..exploration.graph_agent import GraphExplorer as LLMExplorer


logger = logging.getLogger(__name__)


class GraphExplorer(GraphExplorerInterface):
    """Explore and understand graph schemas using LLM."""
    
    def __init__(
        self,
        llm_provider: str = "auto",
        llm_model: str = None
    ):
        """Initialize graph explorer.
        
        Args:
            llm_provider: LLM provider ('openai', 'local', or 'auto' for automatic selection)
            llm_model: Model name (auto-selected if None)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.explorer = LLMExplorer(provider=llm_provider, model=llm_model)
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        if 'db_path' not in kwargs and 'kuzu_adapter' not in kwargs:
            raise ValueError("Either db_path or kuzu_adapter is required")
        
        if 'db_path' in kwargs:
            db_path = Path(kwargs['db_path'])
            if not db_path.exists():
                raise ValueError(f"Database path does not exist: {db_path}")
        
        return True
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute graph exploration."""
        if 'kuzu_adapter' in kwargs:
            return self.explore_from_adapter(**kwargs)
        else:
            return self.explore_from_db(**kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            "name": "GraphExplorer",
            "version": "1.0.0",
            "description": "Explore and understand graph schemas using LLM (supports OpenAI API or local GPU inference)",
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "inputs": {
                "from_db": ["db_path", "max_samples", "save_schema_to"],
                "from_adapter": ["kuzu_adapter", "max_samples", "save_schema_to"]
            },
            "outputs": ["schema", "summary", "node_types", "edge_types"]
        }
    
    def explore_from_db(
        self,
        db_path: Path,
        max_samples: int = 100,
        save_schema_to: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Explore graph from existing database.
        
        Args:
            db_path: Path to Kuzu database
            max_samples: Maximum samples to analyze
            save_schema_to: Optional path to save schema JSON
            
        Returns:
            Graph schema dictionary
        """
        db_path = Path(db_path)
        logger.info(f"Exploring graph at {db_path}")
        
        # Create adapter for existing database (don't delete it!)
        kuzu_adapter = KuzuAdapter(db_path, create_new=False)
        
        return self.explore_from_adapter(
            kuzu_adapter=kuzu_adapter,
            max_samples=max_samples,
            save_schema_to=save_schema_to
        )
    
    def explore_from_adapter(
        self,
        kuzu_adapter: Any,
        max_samples: int = 100,
        save_schema_to: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Explore graph from adapter instance.
        
        Args:
            kuzu_adapter: KuzuAdapter instance
            max_samples: Maximum samples to analyze
            save_schema_to: Optional path to save schema JSON
            
        Returns:
            Graph schema dictionary
        """
        logger.info(f"Exploring graph schema (max_samples={max_samples})")
        
        # Use LLM explorer to analyze the graph
        try:
            schema_dict = self.explorer.explore(kuzu_adapter, max_samples)
            logger.info(f"LLM exploration successful, found {len(schema_dict.get('nodes', {}))} node types")
        except Exception as e:
            logger.error(f"LLM exploration failed: {e}")
            raise
        
        # The original explorer already returns a dictionary, so we can use it directly
        # Just ensure the format matches what we expect
        if "example_values" not in str(schema_dict.get("nodes", {})):
            # Fix the key name if needed (original uses "examples")
            for node_name, node_info in schema_dict.get("nodes", {}).items():
                if "examples" in node_info and "example_values" not in node_info:
                    node_info["example_values"] = node_info.pop("examples")
        
        # Save schema if requested
        if save_schema_to:
            save_path = Path(save_schema_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            logger.info(f"Schema saved to {save_path}")
        
        return schema_dict
    
    @classmethod
    def load_schema(cls, schema_path: Path) -> Dict[str, Any]:
        """Load a previously saved schema.
        
        Args:
            schema_path: Path to schema JSON file
            
        Returns:
            Schema dictionary
        """
        with open(schema_path, 'r') as f:
            return json.load(f)


def main():
    """CLI entry point for standalone graph exploration."""
    parser = argparse.ArgumentParser(description='Explore Kuzu graph schema')
    parser.add_argument('--db-path', required=True, help='Path to Kuzu database')
    parser.add_argument('--output', help='Output schema file path')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum samples to analyze')
    parser.add_argument('--llm-provider', default='openai',
                       help='LLM provider (default: openai)')
    parser.add_argument('--llm-model', default='gpt-4',
                       help='LLM model (default: gpt-4)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Check for API key
    if args.llm_provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    # Create explorer
    explorer = GraphExplorer(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
    
    # Explore the graph
    schema = explorer.explore_from_db(
        db_path=Path(args.db_path),
        max_samples=args.max_samples,
        save_schema_to=Path(args.output) if args.output else None
    )
    
    # Print results
    print(f"✅ Graph exploration complete!")
    print(f"\n📊 Summary: {schema['summary']}")
    print(f"\n🔵 Node Types: {list(schema['nodes'].keys())}")
    print(f"🔗 Edge Types: {list(schema['edges'].keys())}")
    
    if args.output:
        print(f"\n💾 Schema saved to: {args.output}")


if __name__ == "__main__":
    main()