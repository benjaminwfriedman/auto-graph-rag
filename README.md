# Auto-Graph-RAG

🚀 **Automatically fine-tune small language models to generate Cypher queries for your knowledge graphs**

Auto-Graph-RAG transforms any NetworkX graph into a question-answering system by training specialized models to convert natural language into graph queries. No manual query writing required!

## 🎯 Key Features

- **🔄 Automatic Pipeline**: From graph → schema exploration → training data → fine-tuned model → deployed QA system
- **🧩 Modular Architecture**: Use components independently or compose them for custom workflows
- **⚡ Efficient Fine-tuning**: LoRA/QLoRA training makes it work on consumer GPUs
- **✅ Validated Queries**: All generated Cypher queries are validated against the actual database
- **📊 Progressive Complexity**: Generates training data from simple lookups to complex aggregations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/benjaminwfriedman/auto-graph-rag.git
cd auto-graph-rag

# Install the package
pip install -e .
```

### Basic Usage

```python
import networkx as nx
from auto_graph_rag import GraphRAG

# Initialize the system
rag = GraphRAG(
    llm_provider="openai",
    llm_model="gpt-4",
    target_model="meta-llama/Llama-3.2-1B-Instruct"
)

# Create your graph
G = nx.DiGraph()
G.add_node("emp1", type="Employee", name="Alice", department="Engineering", salary=120000)
G.add_node("dept1", type="Department", name="Engineering", budget=2000000)
G.add_edge("emp1", "dept1", type="BELONGS_TO")

# Run the full pipeline
rag.ingest_graph(G, name="company")
schema = rag.explore_graph()
dataset = rag.generate_training_data(num_examples=1000)
model = rag.fine_tune(dataset, epochs=15)

# Query your graph with natural language
result = rag.query("Which employees work in Engineering?")
print(result["formatted"])
```

## 📦 Modular Architecture

Auto-Graph-RAG is built with independent, composable modules. Use them separately or chain them together:

### 1️⃣ GraphBuilder - Create Graph Databases
```python
from auto_graph_rag.modules import GraphBuilder

builder = GraphBuilder()
stats = builder.build_from_networkx(
    graph=your_networkx_graph,
    db_path="./company_db",
    graph_name="company"
)
# Output: {'node_tables': ['Employee', 'Department'], 'total_nodes': 9, ...}
```

### 2️⃣ GraphExplorer - Understand Your Schema
```python
from auto_graph_rag.modules import GraphExplorer

explorer = GraphExplorer(llm_provider="openai", llm_model="gpt-4")
schema = explorer.explore_from_db(
    db_path="./company_db",
    save_schema_to="./schema.json"
)
# Output: Comprehensive schema with descriptions, relationships, and examples
```

### 3️⃣ DataGenerator - Create Training Data
```python
from auto_graph_rag.modules import DataGenerator

generator = DataGenerator(llm_provider="openai")
dataset = generator.generate_from_schema(
    schema_path="./schema.json",
    num_examples=100,
    complexity_distribution={
        1: 0.3,  # Simple lookups: "Find all employees"
        2: 0.3,  # Filters: "Employees in Engineering"  
        3: 0.2,  # Relationships: "Who worked on Project Alpha?"
        4: 0.2   # Aggregations: "Average salary by department"
    }
)
```

### 4️⃣ ModelTrainer - Fine-tune Your Model
```python
from auto_graph_rag.modules import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_from_file(
    dataset_path="./dataset.jsonl",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    epochs=15,
    learning_rate=5e-4,
    batch_size=4,
    lora_rank=8
)
```

### 5️⃣ QueryExecutor - Deploy Your QA System
```python
from auto_graph_rag.modules import QueryExecutor

executor = QueryExecutor()
result = executor.query_with_model(
    question="What is the average salary by department?",
    model_path="./company_model",
    db_path="./company_db",
    format_results=True
)
```

## 📊 Example Output

Here's what Auto-Graph-RAG produces from a simple company graph:

### Graph Statistics
```
📊 Created sample graph:
  - Nodes: 9 (4 Employees, 3 Departments, 2 Projects)
  - Edges: 9 (BELONGS_TO, WORKED_ON, OWNED_BY relationships)
```

### Schema Discovery
```
📋 Discovered Schema:
  Summary: This graph represents the structure of a company, including
           employees, departments, and projects with their relationships.
  Node Types: ['Employee', 'Department', 'Project']  
  Edge Types: ['BELONGS_TO', 'WORKED_ON', 'OWNED_BY']
```

### Training Data Generation
```
📊 Dataset Statistics:
  Total examples: 100
  Complexity distribution: {1: 30, 2: 30, 3: 20, 4: 20}
  
💡 Sample Training Examples:
  Q: "Who are all the employees in the company?"
  A: MATCH (e:Employee) RETURN e.name
  
  Q: "What is the average salary in Engineering?"
  A: MATCH (e:Employee)-[:BELONGS_TO]->(d:Department {name:'Engineering'}) 
     RETURN AVG(e.salary)
```

### Model Performance
```
🎯 Model Training (15 epochs):
  Training Loss: 3.81 → 0.16
  Validation Loss: 3.60 → 0.41
  Trainable params: 5.6M (0.45% of total)
```

### Query Results
```
❓ Question: "What is the average salary by department?"
✅ Generated Cypher: 
   MATCH (e:Employee)-[:BELONGS_TO]->(d:Department) 
   RETURN d.name AS Department, AVG(e.salary) AS AvgSalary

📊 Results:
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Department   ┃ AvgSalary ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Engineering  │ 107,500   │
│ Marketing    │ 85,000    │
│ Sales        │ 90,000    │
└──────────────┴───────────┘
```

## 🔧 Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- OpenAI API key (for exploration and data generation)
- HuggingFace token (optional, for some models)

### Environment Setup

```bash
# Create .env file
cp .env.example .env

# Add your API keys
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here  # Optional
```

## 📚 Documentation

### Training Data Complexity Levels

1. **Simple Lookups**: Basic node retrieval
   - `"Find all employees"` → `MATCH (e:Employee) RETURN e`

2. **Filtered Queries**: Conditional searches
   - `"Employees in Engineering"` → `MATCH (e:Employee) WHERE e.department = 'Engineering' RETURN e`

3. **Relationship Traversals**: Multi-hop queries
   - `"Who worked on Project Alpha?"` → `MATCH (e:Employee)-[:WORKED_ON]->(p:Project {name:'Alpha'}) RETURN e.name`

4. **Aggregations**: Statistical queries
   - `"Average salary by department"` → Complex aggregation with GROUP BY

5. **Complex Paths**: Advanced graph algorithms
   - `"Shortest path between Alice and Bob"` → Path finding queries

### Supported Models

**Base Models for Fine-tuning:**
- Llama 3.2 (1B, 3B) - Recommended for most use cases


**LLM Providers for Exploration:**
- OpenAI (GPT-4, GPT-3.5)


## 🎓 Examples

Check out the [`demo_notebook.ipynb`](demo_notebook.ipynb) for a complete walkthrough showing:
- Building a company knowledge graph
- Exploring the schema with LLMs
- Generating diverse training data
- Fine-tuning a Llama model
- Deploying the QA system

More examples in the `examples/` directory:
- `company_graph_example.py` - Full company graph with employees, departments, and projects
- `social_network_example.py` - Social media graph with users and posts (coming soon)
- `supply_chain_example.py` - Supply chain optimization (coming soon)

## 🚧 Limitations

- Currently supports Kuzu database only (Neo4j support coming soon)
- OpenAI API costs for exploration and data generation

## 🗺️ Roadmap

- [ ] Support for additional graph databases (Neo4j)
- [ ] Distributed training support


## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📖 Citation

If you use Auto-Graph-RAG in your research, please cite:

```bibtex
@software{auto-graph-rag,
  title = {Auto-Graph-RAG: Automated Fine-tuning for Graph Query Generation},
  author = {Friedman, Benjamin W.},
  year = {2025},
  url = {https://github.com/benjaminwfriedman/auto-graph-rag}
}
```

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM orchestration
- Uses [Kuzu](https://kuzudb.com/) embedded graph database
- Fine-tuning powered by [HuggingFace Transformers](https://huggingface.co/transformers/) and [PEFT](https://github.com/huggingface/peft)

---

<p align="center">
  I love graphs ❤️
</p>