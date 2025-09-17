# Auto-Graph-RAG

Fine-tune language models to generate Cypher queries for knowledge graph question-answering.

## Overview

Auto-Graph-RAG is a Python package that automates the process of training small language models (SLMs) to generate Cypher queries from natural language questions. It:

1. **Ingests** NetworkX graphs into a Kuzu database
2. **Explores** the graph structure using an LLM to understand the schema
3. **Generates** diverse training data (question-Cypher pairs) using the learned schema
4. **Fine-tunes** smaller models for efficient Cypher generation
5. **Deploys** the fine-tuned model for graph RAG (Retrieval-Augmented Generation)

## Features

- 🔄 **Automatic Schema Learning**: Uses LLMs to explore and understand graph structure
- 🎯 **Smart Training Data Generation**: Creates diverse, validated question-Cypher pairs
- ⚡ **Efficient Fine-tuning**: Uses LoRA/QLoRA for parameter-efficient training
- 🔍 **Query Validation**: Validates generated queries against the actual database
- 📊 **Multiple Complexity Levels**: Generates queries from simple lookups to complex path traversals
- 🚀 **Production Ready**: Includes inference engine with error handling and retry logic

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-graph-rag.git
cd auto-graph-rag

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for fine-tuning)
- OpenAI API key (for exploration and data generation)

## Quick Start

```python
import networkx as nx
from auto_graph_rag import GraphRAG

# Initialize the system
rag = GraphRAG(
    llm_provider="openai",
    llm_model="gpt-4",
    target_model="meta-llama/Llama-3.2-1B"
)

# Create a sample graph
G = nx.DiGraph()
G.add_node("emp1", type="Employee", name="Alice", department="Engineering")
G.add_node("proj1", type="Project", name="Alpha", budget=500000)
G.add_edge("emp1", "proj1", type="WORKED_ON", hours=320)

# Ingest the graph
rag.ingest_graph(G, name="company")

# Explore and learn schema
schema = rag.explore_graph(max_samples=100)

# Generate training data
dataset = rag.generate_training_data(
    num_examples=1000,
    complexity_distribution={
        1: 0.2,  # Simple lookups
        2: 0.3,  # Filtered queries
        3: 0.3,  # Relationships
        4: 0.15, # Aggregations
        5: 0.05  # Complex paths
    }
)

# Fine-tune model
model = rag.fine_tune(
    dataset,
    epochs=3,
    learning_rate=2e-5
)

# Query the graph
result = rag.query("Who worked on project Alpha?")
print(result["formatted"])
```

## Architecture

### Components

1. **Ingestion Module** (`ingestion/`)
   - `NetworkXLoader`: Processes NetworkX graphs
   - `KuzuAdapter`: Manages Kuzu database operations

2. **Exploration Module** (`exploration/`)
   - `GraphExplorer`: LLM-based graph structure analysis
   - `SchemaLearner`: Schema storage and management

3. **Generation Module** (`generation/`)
   - `QuestionGenerator`: Creates diverse question-Cypher pairs with validation

4. **Training Module** (`training/`)
   - `DatasetBuilder`: Formats data for different model architectures
   - `FineTuner`: Handles LoRA/QLoRA fine-tuning

5. **Inference Module** (`inference/`)
   - `QueryEngine`: Executes natural language queries
   - `ResultFormatter`: Formats query results

## Training Data Generation

The system generates training data at five complexity levels:

### Level 1: Simple Lookups
```
Q: "Find all employees"
A: MATCH (e:Employee) RETURN e
```

### Level 2: Filtered Queries
```
Q: "Find employees in Engineering department"
A: MATCH (e:Employee) WHERE e.department = 'Engineering' RETURN e
```

### Level 3: Relationship Traversals
```
Q: "Who worked on project Alpha?"
A: MATCH (e:Employee)-[:WORKED_ON]->(p:Project {name: 'Alpha'}) RETURN e.name
```

### Level 4: Aggregations
```
Q: "What is the average salary in Engineering?"
A: MATCH (e:Employee) WHERE e.department = 'Engineering' RETURN AVG(e.salary)
```

### Level 5: Complex Paths
```
Q: "Find the shortest path between Alice and Bob"
A: MATCH path = shortestPath((a:Employee {name: 'Alice'})-[*]-(b:Employee {name: 'Bob'})) RETURN path
```

## Supported Models

### Base Models for Fine-tuning
- Llama 3.2 (1B, 3B)
- Mistral 7B
- Phi-3 mini
- CodeLlama 7B

### LLM Providers for Exploration
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude) - coming soon

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
# .env file:
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Or use environment variables directly:

```bash
export OPENAI_API_KEY="your-api-key"
export HF_TOKEN="your-hf-token"
```

### Custom Configuration

```python
rag = GraphRAG(
    llm_provider="openai",
    llm_model="gpt-4",
    target_model="meta-llama/Llama-3.2-1B",
    db_path="./my_kuzu_db"
)
```

## Examples

See the `examples/` directory for complete examples:

- `company_graph_example.py`: Company knowledge graph with employees and projects
- More examples coming soon!

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff src/

# Type checking
mypy src/
```

## Limitations

- Currently supports Kuzu database only
- Requires GPU for efficient fine-tuning
- OpenAI API costs for exploration and data generation
- Generated queries may need refinement for complex schemas

## Roadmap

- [ ] Support for additional graph databases (Neo4j, ArangoDB)
- [ ] Multi-modal graph support (images, embeddings)
- [ ] Distributed training support
- [ ] Web UI for exploration and testing
- [ ] Pre-trained models for common graph patterns
- [ ] Support for graph mutations (CREATE, UPDATE, DELETE)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Auto-Graph-RAG in your research, please cite:

```bibtex
@software{auto-graph-rag,
  title = {Auto-Graph-RAG: Automated Fine-tuning for Graph Query Generation},
  year = {2024},
  url = {https://github.com/yourusername/auto-graph-rag}
}
```

## Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM orchestration
- Uses [Kuzu](https://kuzudb.com/) embedded graph database
- Fine-tuning powered by [HuggingFace Transformers](https://huggingface.co/transformers/) and [PEFT](https://github.com/huggingface/peft)