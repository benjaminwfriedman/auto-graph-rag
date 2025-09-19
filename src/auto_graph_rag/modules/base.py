"""Base interfaces for modular components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path


class BaseModule(ABC):
    """Base class for all modules."""
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the module's main functionality."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get module information and capabilities."""
        pass


class GraphBuilderInterface(BaseModule):
    """Interface for graph creation modules."""
    
    @abstractmethod
    def build_from_networkx(
        self,
        graph: Any,
        db_path: Path,
        graph_name: str = "default",
        node_labels: Optional[Dict[Any, str]] = None,
        edge_labels: Optional[Dict[Any, str]] = None
    ) -> Dict[str, Any]:
        """Build graph database from NetworkX graph."""
        pass
    
    @abstractmethod
    def build_from_data(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        db_path: Path,
        graph_name: str = "default"
    ) -> Dict[str, Any]:
        """Build graph database from raw data."""
        pass


class GraphExplorerInterface(BaseModule):
    """Interface for graph understanding modules."""
    
    @abstractmethod
    def explore_from_db(
        self,
        db_path: Path,
        max_samples: int = 100,
        save_schema_to: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Explore graph from existing database."""
        pass
    
    @abstractmethod
    def explore_from_adapter(
        self,
        kuzu_adapter: Any,
        max_samples: int = 100,
        save_schema_to: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Explore graph from adapter instance."""
        pass


class DataGeneratorInterface(BaseModule):
    """Interface for training data generation modules."""
    
    @abstractmethod
    def generate_from_schema(
        self,
        schema_path: Path,
        num_examples: int,
        output_path: Path,
        complexity_distribution: Optional[Dict[int, float]] = None,
        db_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Generate training data from schema file."""
        pass
    
    @abstractmethod
    def generate_from_dict(
        self,
        schema: Dict[str, Any],
        num_examples: int,
        output_path: Optional[Path] = None,
        complexity_distribution: Optional[Dict[int, float]] = None,
        kuzu_adapter: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate training data from schema dictionary."""
        pass


class ModelTrainerInterface(BaseModule):
    """Interface for model fine-tuning modules."""
    
    @abstractmethod
    def train_from_file(
        self,
        dataset_path: Path,
        model_name: str,
        output_dir: Path,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        lora_rank: int = 16
    ) -> Any:
        """Train model from dataset file."""
        pass
    
    @abstractmethod
    def train_from_data(
        self,
        dataset: List[Dict[str, Any]],
        model_name: str,
        output_dir: Path,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        lora_rank: int = 16
    ) -> Any:
        """Train model from dataset in memory."""
        pass


class QueryExecutorInterface(BaseModule):
    """Interface for inference modules."""
    
    @abstractmethod
    def query_with_model(
        self,
        question: str,
        model_path: Path,
        db_path: Path,
        return_cypher: bool = True,
        format_results: bool = True
    ) -> Dict[str, Any]:
        """Execute query with specified model and database."""
        pass
    
    @abstractmethod
    def query_with_instances(
        self,
        question: str,
        model: Any,
        kuzu_adapter: Any,
        return_cypher: bool = True,
        format_results: bool = True
    ) -> Dict[str, Any]:
        """Execute query with existing model and adapter instances."""
        pass