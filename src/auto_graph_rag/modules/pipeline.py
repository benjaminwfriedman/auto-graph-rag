"""Pipeline orchestration for modular components."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
import argparse
from dataclasses import dataclass, asdict

from .graph_builder import GraphBuilder
from .graph_explorer import GraphExplorer
from .data_generator import DataGenerator
from .model_trainer import ModelTrainer
from .query_executor import QueryExecutor


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    
    # Input/Output paths
    input_graph: Optional[str] = None
    input_nodes: Optional[str] = None  # JSON file with nodes array
    input_edges: Optional[str] = None  # JSON file with edges array
    schema_file: Optional[str] = None
    dataset_file: Optional[str] = None
    model_path: Optional[str] = None
    
    # Output paths
    db_path: str = "./pipeline_db"
    output_schema: Optional[str] = None
    output_dataset: Optional[str] = None
    output_model: Optional[str] = None
    
    # Pipeline steps to run
    build_graph: bool = True
    explore_schema: bool = True
    generate_data: bool = True
    train_model: bool = True
    test_queries: bool = True
    
    # Module parameters
    graph_name: str = "default"
    max_samples: int = 100
    num_examples: int = 100
    complexity_distribution: Optional[Dict[int, float]] = None
    
    # Model parameters
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    lora_rank: int = 16
    
    # LLM parameters
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    
    # Test queries
    test_questions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'PipelineConfig':
        """Load from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, config_path: Path):
        """Save to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Pipeline:
    """Orchestrate modular pipeline execution."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.state = {}  # Track pipeline state
        
        # Initialize modules
        self.graph_builder = GraphBuilder()
        self.graph_explorer = GraphExplorer(config.llm_provider, config.llm_model)
        self.data_generator = DataGenerator(config.llm_provider, config.llm_model)
        self.model_trainer = ModelTrainer()
        self.query_executor = QueryExecutor()
    
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Returns:
            Pipeline execution results
        """
        results = {"steps": [], "success": True}
        
        try:
            # Step 1: Build graph
            if self.config.build_graph:
                logger.info("=== Step 1: Building Graph ===")
                build_result = self._build_graph()
                results["steps"].append({"name": "build_graph", "result": build_result})
                self.state["db_path"] = build_result["db_path"]
            
            # Step 2: Explore schema
            if self.config.explore_schema:
                logger.info("=== Step 2: Exploring Schema ===")
                schema_result = self._explore_schema()
                results["steps"].append({"name": "explore_schema", "result": schema_result})
                self.state["schema"] = schema_result
            
            # Step 3: Generate training data
            if self.config.generate_data:
                logger.info("=== Step 3: Generating Training Data ===")
                data_result = self._generate_data()
                results["steps"].append({"name": "generate_data", "result": data_result})
                self.state["dataset"] = data_result
            
            # Step 4: Train model
            if self.config.train_model:
                logger.info("=== Step 4: Training Model ===")
                train_result = self._train_model()
                results["steps"].append({"name": "train_model", "result": train_result})
                self.state["model_path"] = self.config.output_model or "./pipeline_model"
            
            # Step 5: Test queries
            if self.config.test_queries:
                logger.info("=== Step 5: Testing Queries ===")
                test_result = self._test_queries()
                results["steps"].append({"name": "test_queries", "result": test_result})
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def _build_graph(self) -> Dict[str, Any]:
        """Build graph step."""
        if self.config.input_graph:
            # Load NetworkX graph from Python file
            import importlib.util
            spec = importlib.util.spec_from_file_location("graph_module", self.config.input_graph)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            graph = module.graph
            node_labels = getattr(module, 'node_labels', None)
            edge_labels = getattr(module, 'edge_labels', None)
            
            return self.graph_builder.build_from_networkx(
                graph=graph,
                db_path=Path(self.config.db_path),
                graph_name=self.config.graph_name,
                node_labels=node_labels,
                edge_labels=edge_labels
            )
        
        elif self.config.input_nodes and self.config.input_edges:
            # Load from JSON files
            with open(self.config.input_nodes, 'r') as f:
                nodes = json.load(f)
            with open(self.config.input_edges, 'r') as f:
                edges = json.load(f)
            
            return self.graph_builder.build_from_data(
                nodes=nodes,
                edges=edges,
                db_path=Path(self.config.db_path),
                graph_name=self.config.graph_name
            )
        
        else:
            raise ValueError("Either input_graph or input_nodes+input_edges required for build_graph")
    
    def _explore_schema(self) -> Dict[str, Any]:
        """Explore schema step."""
        db_path = self.state.get("db_path") or self.config.db_path
        
        return self.graph_explorer.explore_from_db(
            db_path=Path(db_path),
            max_samples=self.config.max_samples,
            save_schema_to=Path(self.config.output_schema) if self.config.output_schema else None
        )
    
    def _generate_data(self) -> List[Dict[str, Any]]:
        """Generate training data step."""
        # Use schema from previous step or load from file
        if "schema" in self.state:
            schema = self.state["schema"]
            kuzu_adapter = None
            if "db_path" in self.state:
                from ..ingestion.kuzu_adapter import KuzuAdapter
                kuzu_adapter = KuzuAdapter(Path(self.state["db_path"]))
            
            return self.data_generator.generate_from_dict(
                schema=schema,
                num_examples=self.config.num_examples,
                output_path=Path(self.config.output_dataset) if self.config.output_dataset else None,
                complexity_distribution=self.config.complexity_distribution,
                kuzu_adapter=kuzu_adapter
            )
        
        elif self.config.schema_file:
            return self.data_generator.generate_from_schema(
                schema_path=Path(self.config.schema_file),
                num_examples=self.config.num_examples,
                output_path=Path(self.config.output_dataset) if self.config.output_dataset else None,
                complexity_distribution=self.config.complexity_distribution,
                db_path=Path(self.state.get("db_path", self.config.db_path))
            )
        
        else:
            raise ValueError("Either previous schema exploration or schema_file required for generate_data")
    
    def _train_model(self) -> Any:
        """Train model step."""
        # Use dataset from previous step or load from file
        if "dataset" in self.state:
            dataset = self.state["dataset"]
            
            return self.model_trainer.train_from_data(
                dataset=dataset,
                model_name=self.config.base_model,
                output_dir=Path(self.config.output_model or "./pipeline_model"),
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                lora_rank=self.config.lora_rank
            )
        
        elif self.config.dataset_file:
            return self.model_trainer.train_from_file(
                dataset_path=Path(self.config.dataset_file),
                model_name=self.config.base_model,
                output_dir=Path(self.config.output_model or "./pipeline_model"),
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                lora_rank=self.config.lora_rank
            )
        
        else:
            raise ValueError("Either previous data generation or dataset_file required for train_model")
    
    def _test_queries(self) -> List[Dict[str, Any]]:
        """Test queries step."""
        model_path = self.state.get("model_path") or self.config.model_path
        db_path = self.state.get("db_path") or self.config.db_path
        
        if not model_path:
            raise ValueError("Model path required for test queries")
        
        # Default test questions if none provided
        questions = self.config.test_questions or [
            "List all nodes",
            "Show all relationships",
            "Find connected components"
        ]
        
        return self.query_executor.batch_query(
            questions=questions,
            model_path=Path(model_path),
            db_path=Path(db_path)
        )


def main():
    """CLI entry point for pipeline execution."""
    parser = argparse.ArgumentParser(description='Run modular Auto-Graph-RAG pipeline')
    parser.add_argument('--config', help='Pipeline configuration file (JSON)')
    parser.add_argument('--save-config', help='Save configuration template and exit')
    
    # Input options
    parser.add_argument('--input-graph', help='Input NetworkX graph Python file')
    parser.add_argument('--input-nodes', help='Input nodes JSON file')
    parser.add_argument('--input-edges', help='Input edges JSON file')
    parser.add_argument('--schema-file', help='Input schema JSON file')
    parser.add_argument('--dataset-file', help='Input dataset JSONL file')
    parser.add_argument('--model-path', help='Input model path')
    
    # Output options
    parser.add_argument('--db-path', default='./pipeline_db', help='Database path')
    parser.add_argument('--output-schema', help='Output schema file')
    parser.add_argument('--output-dataset', help='Output dataset file')
    parser.add_argument('--output-model', help='Output model directory')
    
    # Pipeline control
    parser.add_argument('--skip-build', action='store_true', help='Skip graph building')
    parser.add_argument('--skip-explore', action='store_true', help='Skip schema exploration')
    parser.add_argument('--skip-generate', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--skip-test', action='store_true', help='Skip query testing')
    
    # Parameters
    parser.add_argument('--num-examples', type=int, default=100, help='Number of training examples')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-1B-Instruct', help='Base model')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Save configuration template if requested
    if args.save_config:
        config = PipelineConfig()
        config.save(Path(args.save_config))
        print(f"✅ Configuration template saved to: {args.save_config}")
        return
    
    # Load or create configuration
    if args.config:
        config = PipelineConfig.from_file(Path(args.config))
    else:
        config = PipelineConfig(
            input_graph=args.input_graph,
            input_nodes=args.input_nodes,
            input_edges=args.input_edges,
            schema_file=args.schema_file,
            dataset_file=args.dataset_file,
            model_path=args.model_path,
            db_path=args.db_path,
            output_schema=args.output_schema,
            output_dataset=args.output_dataset,
            output_model=args.output_model,
            build_graph=not args.skip_build,
            explore_schema=not args.skip_explore,
            generate_data=not args.skip_generate,
            train_model=not args.skip_train,
            test_queries=not args.skip_test,
            num_examples=args.num_examples,
            epochs=args.epochs,
            base_model=args.base_model
        )
    
    # Run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run()
    
    # Print results
    if results["success"]:
        print("✅ Pipeline completed successfully!")
        for step in results["steps"]:
            print(f"  ✓ {step['name']}")
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    main()