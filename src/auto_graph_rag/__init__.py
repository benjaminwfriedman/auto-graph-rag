"""Auto-Graph-RAG: Fine-tune language models for Cypher query generation."""

# Original monolithic interface (for backward compatibility)
from .core import GraphRAG

# Original individual components
from .ingestion.networkx_loader import NetworkXLoader
from .exploration.graph_agent import GraphExplorer
from .generation.question_generator import QuestionGenerator
from .training.fine_tuner import FineTuner
from .inference.query_engine import QueryEngine

# New modular components
from .modules import (
    GraphBuilder,
    GraphExplorer as ModularGraphExplorer,
    DataGenerator,
    ModelTrainer,
    QueryExecutor
)

__version__ = "0.1.0"
__all__ = [
    # Original interface
    "GraphRAG",
    "NetworkXLoader",
    "GraphExplorer",
    "QuestionGenerator", 
    "FineTuner",
    "QueryEngine",
    # Modular interface
    "GraphBuilder",
    "ModularGraphExplorer",
    "DataGenerator",
    "ModelTrainer", 
    "QueryExecutor",
]