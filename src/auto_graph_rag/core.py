"""Core GraphRAG class that orchestrates the entire pipeline."""

from typing import Optional, Dict, Any, List
import networkx as nx
from pathlib import Path
import json

from .ingestion.networkx_loader import NetworkXLoader
from .ingestion.kuzu_adapter import KuzuAdapter
from .exploration.graph_agent import GraphExplorer
from .generation.question_generator import QuestionGenerator
from .training.dataset_builder import DatasetBuilder
from .training.fine_tuner import FineTuner
from .inference.query_engine import QueryEngine


class GraphRAG:
    """Main interface for the Auto-Graph-RAG system."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4",
        target_model: str = "llama-3.2-1b",
        db_path: Optional[Path] = None,
    ):
        """Initialize GraphRAG system.
        
        Args:
            llm_provider: LLM provider for exploration ('openai', 'anthropic')
            llm_model: Model name for exploration
            target_model: Target model for fine-tuning
            db_path: Path to Kuzu database (default: ./kuzu_db)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.target_model = target_model
        self.db_path = db_path or Path("./kuzu_db")
        
        self.nx_loader = NetworkXLoader()
        self.kuzu_adapter = KuzuAdapter(self.db_path)
        self.explorer = GraphExplorer(llm_provider, llm_model)
        self.question_gen = QuestionGenerator(llm_provider, llm_model)
        self.dataset_builder = DatasetBuilder()
        self.fine_tuner = None
        self.query_engine = None
        self.current_schema = None
        
    def ingest_graph(
        self,
        graph: nx.Graph,
        name: str = "default",
        node_labels: Optional[Dict[Any, str]] = None,
        edge_labels: Optional[Dict[Any, str]] = None,
    ) -> Dict[str, Any]:
        """Ingest a NetworkX graph into Kuzu database.
        
        Args:
            graph: NetworkX graph object
            name: Name for the graph
            node_labels: Optional mapping of nodes to type labels
            edge_labels: Optional mapping of edges to type labels
            
        Returns:
            Dictionary with ingestion statistics
        """
        processed_graph = self.nx_loader.process_graph(
            graph, node_labels, edge_labels
        )
        
        stats = self.kuzu_adapter.create_from_networkx(
            processed_graph, name
        )
        
        return stats
    
    def explore_graph(
        self,
        max_samples: int = 100,
        save_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explore graph structure using LLM agent.
        
        Args:
            max_samples: Maximum number of samples to analyze
            save_to: Optional path to save schema JSON
            
        Returns:
            Learned graph schema
        """
        schema = self.explorer.explore(
            self.kuzu_adapter,
            max_samples=max_samples
        )
        
        self.current_schema = schema
        
        if save_to:
            with open(save_to, 'w') as f:
                json.dump(schema, f, indent=2)
        
        return schema
    
    def generate_training_data(
        self,
        num_examples: int = 1000,
        complexity_distribution: Optional[Dict[int, float]] = None,
        save_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate training data for fine-tuning.
        
        Args:
            num_examples: Number of examples to generate
            complexity_distribution: Distribution of query complexities
            save_to: Optional path to save dataset
            
        Returns:
            List of training examples
        """
        if not self.current_schema:
            raise ValueError("Must explore graph first to learn schema")
        
        complexity_dist = complexity_distribution or {
            1: 0.2,  # Simple lookups
            2: 0.3,  # Filtered queries
            3: 0.3,  # Relationships
            4: 0.15, # Aggregations
            5: 0.05  # Complex paths
        }
        
        # Generate question-cypher pairs with validation
        dataset = self.question_gen.generate(
            self.current_schema,
            num_examples,
            complexity_dist,
            kuzu_adapter=self.kuzu_adapter,
            validate_queries=True,
            include_results=True
        )
        
        if save_to:
            self.dataset_builder.save_dataset(dataset, save_to)
        
        return dataset
    
    def fine_tune(
        self,
        dataset: List[Dict[str, Any]],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        lora_rank: int = 16,
        output_dir: Optional[str] = None,
        batch_size: int = 4,
        sample_prompts: Optional[List[str]] = None,
    ) -> Any:
        """Fine-tune target model on generated dataset.
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            learning_rate: Learning rate
            lora_rank: LoRA rank for efficient fine-tuning
            output_dir: Directory to save fine-tuned model
            batch_size: Training batch size
            sample_prompts: Optional prompts for generating samples during training
            
        Returns:
            Fine-tuned model
        """
        # Prepare dataset for training
        formatted_dataset = self.dataset_builder.prepare_for_training(
            dataset,
            model_type=self.target_model.split('/')[-1].split('-')[0].lower(),  # e.g., "llama" from "meta-llama/Llama-3.2-1B"
            train_split=0.9,
            model_name=self.target_model  # Pass full name for format detection
        )
        
        self.fine_tuner = FineTuner(
            model_name=self.target_model,
            lora_rank=lora_rank
        )
        
        model = self.fine_tuner.train(
            formatted_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            output_dir=output_dir or f"./models/{self.target_model.split('/')[-1]}-cypher",
            sample_prompts=sample_prompts
        )
        
        return model
    
    def query(
        self,
        question: str,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query the graph using fine-tuned model.
        
        Args:
            question: Natural language question
            model_path: Path to fine-tuned model (uses last trained if None)
            
        Returns:
            Query results
        """
        if not self.query_engine:
            # Determine model path
            if model_path:
                final_model_path = model_path
            elif self.fine_tuner and hasattr(self.fine_tuner, 'output_dir') and self.fine_tuner.output_dir:
                final_model_path = self.fine_tuner.output_dir
            else:
                raise ValueError("No model available. Please train a model first or provide model_path")
            
            self.query_engine = QueryEngine(
                final_model_path,
                self.kuzu_adapter,
                model_name=self.target_model
            )
        
        return self.query_engine.query(question)