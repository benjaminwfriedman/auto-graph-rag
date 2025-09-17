#!/usr/bin/env python3
"""Test script for fine-tuning and inference sections only."""

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from auto_graph_rag import GraphRAG
import networkx as nx

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_minimal_graph():
    """Create a minimal graph just for the Kuzu adapter."""
    G = nx.DiGraph()
    # Just a few nodes to satisfy the GraphRAG initialization
    G.add_node("emp1", type="Employee", name="Alice", department="Engineering")
    G.add_node("dept1", type="Department", name="Engineering")
    G.add_edge("emp1", "dept1", type="BELONGS_TO")
    return G


def load_existing_dataset(filepath="company_training_data.jsonl"):
    """Load existing training data from JSONL file."""
    dataset = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        logger.info(f"✅ Loaded {len(dataset)} training examples from {filepath}")
    except FileNotFoundError:
        logger.error(f"❌ Training data not found at {filepath}")
        logger.info("Please run the full example first to generate training data")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading training data: {e}")
        return None
    
    return dataset


def main():
    """Test fine-tuning and inference only."""
    
    logger.info("=" * 60)
    logger.info("Testing Fine-tuning and Inference")
    logger.info("=" * 60)
    
    # Check for required keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found!")
        return
    
    if not os.getenv("HF_TOKEN"):
        logger.warning("⚠️ HF_TOKEN not found - may not be able to access gated models")
    
    # Initialize the system (minimal setup)
    logger.info("\n📦 Initializing GraphRAG system...")
    rag = GraphRAG(
        llm_provider="openai",
        llm_model="gpt-4",
        target_model='meta-llama/Llama-3.2-1B-Instruct',
        db_path="./test_kuzu_db"
    )
    
    # Create minimal graph for initialization
    graph = create_minimal_graph()
    node_labels = {node: data["type"] for node, data in graph.nodes(data=True)}
    edge_labels = {(u, v): data["type"] for u, v, data in graph.edges(data=True)}
    
    logger.info("📊 Setting up minimal graph...")
    rag.ingest_graph(graph, name="test", node_labels=node_labels, edge_labels=edge_labels)
    
    # Load existing training data
    logger.info("\n📚 Loading training data...")
    dataset = load_existing_dataset()
    if not dataset:
        return
    
    # Display dataset info
    logger.info(f"Dataset size: {len(dataset)} examples")
    complexities = {}
    for item in dataset:
        c = item.get('complexity', 0)
        complexities[c] = complexities.get(c, 0) + 1
    logger.info(f"Complexity distribution: {complexities}")
    
    # Show a few examples
    logger.info("\n📝 Sample training data:")
    for i, example in enumerate(dataset[:3], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Question: {example.get('question', 'N/A')}")
        logger.info(f"  Cypher: {example.get('cypher', 'N/A')[:100]}...")
    
    # Fine-tuning section
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Starting Fine-tuning")
    logger.info("=" * 60)
    
    try:
        # Custom sample prompts for monitoring
        sample_prompts = [
            "Who works in Engineering?",
            "List all active projects",
            "Find employees with salary over 100000"
        ]
        
        logger.info("Fine-tuning parameters:")
        logger.info("  - Model: meta-llama/Llama-3.2-1B-Instruct")
        logger.info("  - Epochs: 3")
        logger.info("  - Learning rate: 5e-4")
        logger.info("  - Batch size: 1")
        logger.info("  - LoRA rank: 8")
        
        model = rag.fine_tune(
            dataset,
            epochs=3,  # Quick test with 2 epochs
            learning_rate=5e-4,
            batch_size=1,
            lora_rank=8,
            output_dir="./models/test-cypher-model"
        )
        
        logger.info("✅ Fine-tuning completed successfully!")
        
        # Check if outputs were saved
        output_dir = Path("./models/test-cypher-model")
        if output_dir.exists():
            logger.info(f"\n📁 Model saved to: {output_dir}")
            if (output_dir / "training_loss_curve.png").exists():
                logger.info("  ✅ Loss curve saved")
            if (output_dir / "training_samples.json").exists():
                logger.info("  ✅ Training samples saved")
                # Load and show final samples
                with open(output_dir / "training_samples.json", 'r') as f:
                    samples = json.load(f)
                    if samples:
                        logger.info(f"  📊 Generated {len(samples)} sample outputs during training")
                        last_sample = samples[-1]
                        logger.info(f"  Last epoch sample: {last_sample.get('generated', 'N/A')[:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}")
        logger.info("Skipping to inference tests...")
        return
    
    # Inference section
    logger.info("\n" + "=" * 60)
    logger.info("🔍 Testing Inference")
    logger.info("=" * 60)
    
    test_questions = [
        "Who works in Engineering?",
        "List all employees",
        "Find active projects",
        "What is the average salary?",
        "Show employees who report to Alice"
    ]
    
    logger.info("\n🤖 Testing fine-tuned model:")
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[{i}] Q: {question}")
        try:
            result = rag.query(question)
            if result.get("success"):
                logger.info(f"    ✅ Generated Cypher: {result.get('cypher', 'N/A')[:100]}...")
                logger.info(f"    📊 Results: {result.get('count', 0)} rows")
            else:
                logger.warning(f"    ⚠️ Query failed: {result.get('error', 'Unknown')[:100]}...")
        except Exception as e:
            logger.error(f"    ❌ Error: {str(e)[:100]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("✨ Test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()