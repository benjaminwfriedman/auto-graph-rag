"""Standalone module for fine-tuning models."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import argparse
import logging

from .base import ModelTrainerInterface
from ..training.fine_tuner import FineTuner
from ..training.dataset_builder import DatasetBuilder


logger = logging.getLogger(__name__)


class ModelTrainer(ModelTrainerInterface):
    """Fine-tune models on graph query datasets."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.dataset_builder = DatasetBuilder()
        self.fine_tuner = None
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        if 'dataset_path' not in kwargs and 'dataset' not in kwargs:
            raise ValueError("Either dataset_path or dataset is required")
        
        if 'model_name' not in kwargs:
            raise ValueError("model_name is required")
        
        if 'output_dir' not in kwargs:
            raise ValueError("output_dir is required")
        
        if 'dataset_path' in kwargs:
            dataset_path = Path(kwargs['dataset_path'])
            if not dataset_path.exists():
                raise ValueError(f"Dataset file not found: {dataset_path}")
        
        return True
    
    def execute(self, **kwargs) -> Any:
        """Execute model training."""
        if 'dataset_path' in kwargs:
            return self.train_from_file(**kwargs)
        else:
            return self.train_from_data(**kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            "name": "ModelTrainer",
            "version": "1.0.0",
            "description": "Fine-tune models on graph query datasets",
            "inputs": {
                "from_file": ["dataset_path", "model_name", "output_dir", "epochs", "learning_rate", "batch_size", "lora_rank"],
                "from_data": ["dataset", "model_name", "output_dir", "epochs", "learning_rate", "batch_size", "lora_rank"]
            },
            "outputs": ["model", "training_stats", "model_path"]
        }
    
    def train_from_file(
        self,
        dataset_path: Path,
        model_name: str,
        output_dir: Path,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        lora_rank: int = 16,
        sample_prompts: Optional[List[str]] = None
    ) -> Any:
        """Train model from dataset file.
        
        Args:
            dataset_path: Path to dataset JSONL file
            model_name: Base model name to fine-tune
            output_dir: Directory to save fine-tuned model
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            lora_rank: LoRA rank for efficient fine-tuning
            sample_prompts: Optional prompts for monitoring during training
            
        Returns:
            Trained model
        """
        dataset_path = Path(dataset_path)
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(dataset)} training examples")
        
        return self.train_from_data(
            dataset=dataset,
            model_name=model_name,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            lora_rank=lora_rank,
            sample_prompts=sample_prompts
        )
    
    def train_from_data(
        self,
        dataset: List[Dict[str, Any]],
        model_name: str,
        output_dir: Path,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        lora_rank: int = 16,
        sample_prompts: Optional[List[str]] = None
    ) -> Any:
        """Train model from dataset in memory.
        
        Args:
            dataset: Training dataset
            model_name: Base model name to fine-tune
            output_dir: Directory to save fine-tuned model
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            lora_rank: LoRA rank for efficient fine-tuning
            sample_prompts: Optional prompts for monitoring during training
            
        Returns:
            Trained model
        """
        output_dir = Path(output_dir)
        logger.info(f"Training {model_name} on {len(dataset)} examples")
        logger.info(f"Output directory: {output_dir}")
        
        # Prepare dataset for training
        logger.info("Preparing dataset for training...")
        formatted_dataset = self.dataset_builder.prepare_for_training(
            dataset,
            model_type=model_name.split('/')[-1].split('-')[0].lower(),
            train_split=0.9,
            model_name=model_name
        )
        
        # Initialize fine-tuner
        logger.info(f"Initializing fine-tuner (LoRA rank: {lora_rank})")
        self.fine_tuner = FineTuner(
            model_name=model_name,
            lora_rank=lora_rank
        )
        
        # Train the model
        logger.info("Starting training...")
        model = self.fine_tuner.train(
            formatted_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            output_dir=str(output_dir),
            sample_prompts=sample_prompts
        )
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        return model
    
    def get_training_stats(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Get training statistics from output directory.
        
        Args:
            output_dir: Training output directory
            
        Returns:
            Training statistics if available
        """
        output_dir = Path(output_dir)
        
        stats = {
            "model_path": str(output_dir),
            "files": [],
            "training_samples": None,
            "loss_curve": None
        }
        
        # Check for output files
        if output_dir.exists():
            stats["files"] = [f.name for f in output_dir.iterdir()]
            
            # Load training samples if available
            samples_file = output_dir / "training_samples.json"
            if samples_file.exists():
                with open(samples_file, 'r') as f:
                    stats["training_samples"] = json.load(f)
            
            # Check for loss curve
            loss_file = output_dir / "training_loss_curve.png"
            if loss_file.exists():
                stats["loss_curve"] = str(loss_file)
        
        return stats
    
    @classmethod
    def load_model_info(cls, model_path: Path) -> Dict[str, Any]:
        """Load information about a trained model.
        
        Args:
            model_path: Path to trained model directory
            
        Returns:
            Model information
        """
        model_path = Path(model_path)
        
        info = {
            "path": str(model_path),
            "exists": model_path.exists(),
            "files": [],
            "config": None
        }
        
        if model_path.exists():
            info["files"] = [f.name for f in model_path.iterdir()]
            
            # Try to load config
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    info["config"] = json.load(f)
        
        return info


def main():
    """CLI entry point for standalone model training."""
    parser = argparse.ArgumentParser(description='Fine-tune model on graph query dataset')
    parser.add_argument('--dataset', required=True, help='Dataset JSONL file path')
    parser.add_argument('--model', required=True, help='Base model name to fine-tune')
    parser.add_argument('--output', required=True, help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--sample-prompts', nargs='+',
                       help='Sample prompts for monitoring training')
    parser.add_argument('--stats', action='store_true',
                       help='Print training statistics after completion')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train the model
    try:
        model = trainer.train_from_file(
            dataset_path=Path(args.dataset),
            model_name=args.model,
            output_dir=Path(args.output),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            sample_prompts=args.sample_prompts
        )
        
        print(f"✅ Model training completed!")
        print(f"💾 Model saved to: {args.output}")
        
        # Print statistics if requested
        if args.stats:
            stats = trainer.get_training_stats(Path(args.output))
            print(f"\n📊 Training Statistics:")
            print(f"  Output files: {stats['files']}")
            if stats['training_samples']:
                print(f"  Training samples generated: {len(stats['training_samples'])}")
            if stats['loss_curve']:
                print(f"  Loss curve saved: {stats['loss_curve']}")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    main()