"""Build and format datasets for fine-tuning."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd
from datasets import Dataset, DatasetDict


class DatasetBuilder:
    """Build and format training datasets."""
    
    def __init__(self):
        """Initialize dataset builder."""
        self.current_dataset = None
    
    def prepare_for_training(
        self,
        raw_data: List[Dict[str, Any]],
        model_type: str = "llama",
        train_split: float = 0.9,
        add_instruction: bool = True,
        model_name: Optional[str] = None
    ) -> DatasetDict:
        """Prepare dataset for fine-tuning.
        
        Args:
            raw_data: Raw question-cypher pairs
            model_type: Target model type for formatting
            train_split: Proportion for training split
            add_instruction: Whether to add instruction prefix
            model_name: Full model name to determine exact format
            
        Returns:
            HuggingFace DatasetDict with train/validation splits
        """
        # Format data based on model type
        formatted_data = self._format_for_model(
            raw_data, model_type, add_instruction, model_name
        )
        
        # Filter out invalid entries
        valid_data = [
            d for d in formatted_data 
            if d.get("is_valid", True) and d.get("input") and d.get("output")
        ]
        
        # Create train/validation split
        split_idx = int(len(valid_data) * train_split)
        train_data = valid_data[:split_idx]
        val_data = valid_data[split_idx:]
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        self.current_dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        return self.current_dataset
    
    def _format_for_model(
        self,
        raw_data: List[Dict[str, Any]],
        model_type: str,
        add_instruction: bool,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Format data for specific model type.
        
        Args:
            raw_data: Raw training data
            model_type: Target model type
            add_instruction: Whether to add instruction
            model_name: Full model name
            
        Returns:
            Formatted data
        """
        formatted = []
        
        instruction = ""
        if add_instruction:
            instruction = "Generate only a Cypher query to answer this question. Return only the query, no explanation.\n\nQuestion: "
        
        for item in raw_data:
            # Check if we should use Llama 3.1 Instruct format
            use_llama_31_format = (
                model_name and ("Llama-3.1" in model_name or "Llama-3.2" in model_name) and "Instruct" in model_name
            )
            
            if model_type in ["llama", "mistral", "codellama"]:
                if use_llama_31_format:
                    # Use Llama 3.1/3.2 Instruct chat format
                    system_prompt = "You are an expert at generating Cypher queries for Neo4j graph databases. Generate only valid Cypher queries with no explanation."
                    user_prompt = f"Generate a Cypher query to answer this question: {item.get('question', '')}"
                    cypher = item.get('cypher', '')
                    
                    # Llama 3.1 Instruct format
                    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    
                    entry = {
                        "input": formatted_prompt,
                        "output": cypher,
                        "text": f"{formatted_prompt}{cypher}<|eot_id|>",
                        "complexity": item.get("complexity", 1),
                        "is_valid": item.get("is_valid", True)
                    }
                else:
                    # Use simple format for TinyLlama and other models
                    prompt = f"{instruction}{item.get('question', '')}"
                    cypher = item.get('cypher', '')
                    
                    entry = {
                        "input": prompt,
                        "output": cypher,
                        "text": f"{prompt}\n\n{cypher}",
                        "complexity": item.get("complexity", 1),
                        "is_valid": item.get("is_valid", True)
                    }
            elif model_type == "phi":
                # Phi-style formatting - let model generate complete query
                prompt = f"Instruct: {instruction}{item.get('question', '')}\nOutput:"
                cypher = item.get('cypher', '')
                
                entry = {
                    "input": prompt,
                    "output": cypher,
                    "text": f"{prompt} {cypher}",
                    "complexity": item.get("complexity", 1),
                    "is_valid": item.get("is_valid", True)
                }
            else:
                # Generic formatting - let model generate complete query
                prompt = f"{instruction}{item.get('question', '')}"
                cypher = item.get('cypher', '')
                
                entry = {
                    "input": prompt,
                    "output": cypher,
                    "text": f"{prompt}\n\n{cypher}",
                    "complexity": item.get("complexity", 1),
                    "is_valid": item.get("is_valid", True)
                }
            
            formatted.append(entry)
        
        return formatted
    
    def save_dataset(
        self,
        dataset: Any,
        output_path: str,
        format: str = "jsonl"
    ):
        """Save dataset to file.
        
        Args:
            dataset: Dataset to save
            output_path: Output file path
            format: Output format (jsonl, json, parquet, csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(dataset, DatasetDict):
            # Save train and validation separately
            for split_name, split_data in dataset.items():
                split_path = output_path.parent / f"{output_path.stem}_{split_name}{output_path.suffix}"
                self._save_split(split_data, split_path, format)
        elif isinstance(dataset, Dataset):
            self._save_split(dataset, output_path, format)
        elif isinstance(dataset, list):
            # Raw data list
            if format == "jsonl":
                with open(output_path, 'w') as f:
                    for item in dataset:
                        f.write(json.dumps(item) + '\n')
            elif format == "json":
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
            elif format == "csv":
                pd.DataFrame(dataset).to_csv(output_path, index=False)
            elif format == "parquet":
                pd.DataFrame(dataset).to_parquet(output_path)
    
    def _save_split(
        self,
        dataset: Dataset,
        output_path: Path,
        format: str
    ):
        """Save a single dataset split.
        
        Args:
            dataset: Dataset split to save
            output_path: Output path
            format: Output format
        """
        if format == "jsonl":
            dataset.to_json(output_path, lines=True)
        elif format == "json":
            dataset.to_json(output_path)
        elif format == "csv":
            dataset.to_csv(output_path)
        elif format == "parquet":
            dataset.to_parquet(output_path)
    
    def load_dataset(
        self,
        path: str,
        format: str = "jsonl"
    ) -> Dataset:
        """Load dataset from file.
        
        Args:
            path: File path
            format: File format
            
        Returns:
            Loaded dataset
        """
        if format == "jsonl":
            return Dataset.from_json(path)
        elif format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
            return Dataset.from_list(data)
        elif format == "csv":
            return Dataset.from_csv(path)
        elif format == "parquet":
            return Dataset.from_parquet(path)
    
    def get_statistics(
        self,
        dataset: Any
    ) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_examples": 0,
            "complexity_distribution": {},
            "avg_input_length": 0,
            "avg_output_length": 0,
            "valid_queries": 0,
            "invalid_queries": 0
        }
        
        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                split_stats = self._get_split_stats(split_data)
                stats[f"{split_name}_examples"] = split_stats["count"]
        elif isinstance(dataset, Dataset):
            stats = self._get_split_stats(dataset)
        elif isinstance(dataset, list):
            stats = self._get_list_stats(dataset)
        
        return stats
    
    def _get_split_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """Get statistics for a dataset split."""
        return {
            "count": len(dataset),
            "columns": dataset.column_names,
            "features": str(dataset.features)
        }
    
    def _get_list_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """Get statistics for raw data list."""
        stats = {
            "total_examples": len(data),
            "complexity_distribution": {},
            "avg_question_length": 0,
            "avg_cypher_length": 0,
            "valid_queries": 0,
            "invalid_queries": 0
        }
        
        if not data:
            return stats
        
        # Calculate statistics
        question_lengths = []
        cypher_lengths = []
        
        for item in data:
            # Complexity distribution
            complexity = item.get("complexity", 0)
            stats["complexity_distribution"][complexity] = \
                stats["complexity_distribution"].get(complexity, 0) + 1
            
            # Lengths
            if "question" in item:
                question_lengths.append(len(item["question"]))
            if "cypher" in item:
                cypher_lengths.append(len(item["cypher"]))
            
            # Validity
            if item.get("is_valid", True):
                stats["valid_queries"] += 1
            else:
                stats["invalid_queries"] += 1
        
        # Averages
        if question_lengths:
            stats["avg_question_length"] = sum(question_lengths) / len(question_lengths)
        if cypher_lengths:
            stats["avg_cypher_length"] = sum(cypher_lengths) / len(cypher_lengths)
        
        return stats