#!/usr/bin/env python3
"""
Auto-Graph-RAG Nations Dataset Demo Script

This script demonstrates Auto-Graph-RAG using the Nations dataset from PyKEEN,
a real-world knowledge graph of international relations between 14 countries.

Dataset Overview:
- 14 countries: USA, USSR, China, UK, Brazil, India, Egypt, Israel, Jordan, Indonesia, Cuba, Poland, Burma, Netherlands  
- 55 relationship types: embassy, militaryalliance, economicaid, treaties, tourism, trade, etc.
- 1,592 relationship instances: Real diplomatic, economic, and political relationships

Usage:
    python nation_graph_example.py
"""

import sys
import os
from pathlib import Path
import pykeen.datasets
import networkx as nx
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_environment():
    """Setup environment and working directory."""
    print("🔍 Setting up environment...")
    
    # Create working directory
    work_dir = Path("./nations_demo_workspace")
    work_dir.mkdir(exist_ok=True)
    print(f"📁 Working directory: {work_dir.absolute()}")
    
    # Add src to path if it exists
    local_src = Path("./src")
    if local_src.exists():
        sys.path.append(str(local_src))
        print("✅ Added local src to Python path")
    
    # Check environment variables
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_hf = bool(os.getenv("HF_TOKEN"))
    
    print(f"🔑 OpenAI API Key: {'✅ Available' if has_openai else '❌ Missing - Required for this demo'}")
    print(f"🤗 HuggingFace Token: {'✅ Available' if has_hf else '⚠️ Missing - Recommended for model downloads'}")
    
    if not has_openai:
        print("\n❌ Please set your OPENAI_API_KEY environment variable to continue.")
        print("   You can set it in a .env file or export it:")
        print("   export OPENAI_API_KEY='your-key-here'")
        return None, False
    
    return work_dir, True

def load_nations_dataset():
    """Load the Nations dataset from PyKEEN and convert to NetworkX."""
    print("\n📦 Loading Nations dataset from PyKEEN...")
    
    # Load dataset
    dataset = pykeen.datasets.Nations()
    training = dataset.training
    
    print(f"📊 Dataset Statistics:")
    print(f"  - Countries: {training.num_entities}")
    print(f"  - Relationship types: {training.num_relations}")
    print(f"  - Total relationships: {training.num_triples}")
    
    # Get mappings
    entity_to_id = training.entity_to_id
    relation_to_id = training.relation_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    print(f"\n🌍 Countries in dataset:")
    countries = sorted(entity_to_id.keys())
    for i, country in enumerate(countries):
        if i % 5 == 0 and i > 0:
            print()  # New line every 5 countries
        print(f"{country:12}", end=" ")
    print()
    
    print(f"\n🔗 Top 10 relationship types:")
    triples = training.mapped_triples
    relation_counts = Counter()
    for i in range(len(triples)):
        h, r, t = triples[i]
        relation = id_to_relation[r.item()]
        relation_counts[relation] += 1
    
    for relation, count in relation_counts.most_common(10):
        print(f"  - {relation:20}: {count:3d} instances")
    
    # Create NetworkX graph
    print(f"\n🔄 Converting to NetworkX graph...")
    G = nx.DiGraph()
    
    # Add nodes (countries) with metadata
    for country in entity_to_id.keys():
        G.add_node(country, type="Country", name=country, region="Unknown")
    
    # Add edges (relationships)
    for i in range(len(triples)):
        h, r, t = triples[i]
        head_country = id_to_entity[h.item()]
        relation = id_to_relation[r.item()]
        tail_country = id_to_entity[t.item()]
        
        # Add edge with relation type
        G.add_edge(head_country, tail_country, type=relation, relation=relation)
    
    print(f"✅ NetworkX graph created:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print(f"  - Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")
    
    return G, relation_counts

def visualize_dataset(relation_counts):
    """Create visualizations of the dataset."""
    print("\n📊 Creating dataset visualization...")
    
    # Visualize relationship distribution
    plt.figure(figsize=(12, 6))
    top_relations = dict(relation_counts.most_common(15))
    plt.bar(range(len(top_relations)), list(top_relations.values()))
    plt.xticks(range(len(top_relations)), list(top_relations.keys()), rotation=45, ha='right')
    plt.title('Top 15 International Relationship Types in Nations Dataset')
    plt.ylabel('Number of Instances')
    plt.tight_layout()
    plt.savefig('nations_relationships.png', dpi=150, bbox_inches='tight')
    plt.show()

def show_example_relationships(nations_graph):
    """Show sample relationships from the graph."""
    print("\n💡 Example relationships:")
    edges = list(nations_graph.edges(data=True))
    import random
    random.seed(42)
    sample_edges = random.sample(edges, min(10, len(edges)))
    
    for source, target, data in sample_edges:
        relation = data['relation']
        print(f"  {source:12} --[{relation:15}]--> {target}")

def build_graph_database(nations_graph, work_dir):
    """Build the graph database using GraphBuilder."""
    print("\n🏗️ Step 2: Build Graph Database with GraphBuilder")
    
    # Import Auto-Graph-RAG modules
    try:
        from auto_graph_rag.modules import GraphBuilder
    except ImportError:
        print("❌ Could not import GraphBuilder. Make sure auto-graph-rag is installed.")
        return None
    
    # Initialize GraphBuilder
    builder = GraphBuilder()
    
    print("🔧 GraphBuilder Info:")
    info = builder.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n📦 Building Nations graph database...")
    
    # Extract node and edge labels
    node_labels = {node: data["type"] for node, data in nations_graph.nodes(data=True)}
    edge_labels = {(u, v): data["type"] for u, v, data in nations_graph.edges(data=True)}
    
    print(f"🏷️ Creating labels:")
    print(f"  - Node types: {set(node_labels.values())}")
    print(f"  - Edge types: {len(set(edge_labels.values()))} unique relationship types")
    
    # Build the database
    db_path = work_dir / "nations_db"
    stats = builder.build_from_networkx(
        graph=nations_graph,
        db_path=db_path,
        graph_name="nations",
        node_labels=node_labels,
        edge_labels=edge_labels
    )
    
    print("\n✅ Nations graph database created!")
    print(f"📊 Database Statistics:")
    for key, value in stats.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"  {key}: {len(value)} items (showing first 5: {value[:5]}...)")
        else:
            print(f"  {key}: {value}")
    
    return db_path

def explore_schema(db_path, work_dir):
    """Explore the graph schema using GraphExplorer."""
    print("\n🔍 Step 3: Explore Schema with GraphExplorer")
    
    try:
        from auto_graph_rag.modules import GraphExplorer
    except ImportError:
        print("❌ Could not import GraphExplorer. Make sure auto-graph-rag is installed.")
        return None
    
    # Initialize GraphExplorer
    explorer = GraphExplorer(llm_provider="openai", llm_model="gpt-4")
    
    print("🔧 GraphExplorer Info:")
    info = explorer.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n🕵️ Exploring Nations graph schema...")
    print("This will analyze the international relationships using GPT-4...")
    
    # Explore the database
    schema_path = work_dir / "nations_schema.json"
    schema = explorer.explore_from_db(
        db_path=db_path,
        max_samples=20,  # Sample more for richer analysis
        save_schema_to=schema_path
    )
    
    print("\n✅ Schema exploration complete!")
    
    # Display schema results
    print(f"\n📋 Discovered Schema Summary:")
    print(f"  {schema['summary']}")
    
    print(f"\n🌍 Node Types: {list(schema['nodes'].keys())}")
    print(f"🔗 Edge Types: {len(list(schema['edges'].keys()))} relationship types discovered")
    
    # Show details for Country node type
    if 'Country' in schema['nodes']:
        country_info = schema['nodes']['Country']
        print(f"\n🏛️ Country Node Details:")
        print(f"  Description: {country_info['description']}")
        print(f"  Properties: {country_info['properties']}")
        print(f"  Example: {country_info.get('example_values', 'N/A')}")
    
    # Show some interesting relationship types
    print(f"\n🔗 Sample International Relationships:")
    for i, (rel_type, rel_info) in enumerate(list(schema['edges'].items())[:8]):
        examples = rel_info.get('examples', [])
        example_text = f" (e.g., {examples[0]})" if examples else ""
        print(f"  {i+1}. {rel_type}: {rel_info['description']}{example_text}")
    
    return schema_path, schema

def generate_training_data(schema_path, work_dir, db_path):
    """Generate training data using DataGenerator."""
    print("\n📝 Step 4: Generate Training Data with DataGenerator")
    
    try:
        from auto_graph_rag.modules import DataGenerator
    except ImportError:
        print("❌ Could not import DataGenerator. Make sure auto-graph-rag is installed.")
        return None
    
    # Initialize DataGenerator
    generator = DataGenerator(llm_provider="openai", llm_model="gpt-4")
    
    print("🔧 DataGenerator Info:")
    info = generator.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n📝 Generating training data for international relations...")
    print("This will create diverse question-Cypher pairs about countries and their relationships...")
    
    # Generate dataset from the schema
    dataset_path = work_dir / "nations_dataset.jsonl"
    dataset = generator.generate_from_schema(
        schema_path=schema_path,
        num_examples=120,  # Good balance for demo
        output_path=dataset_path,
        complexity_distribution={
            1: 0.25,  # Simple lookups: "List all countries", "What countries exist?"
            2: 0.35,  # Filtered queries: "Countries with embassies", "Military alliances"
            3: 0.25,  # Relationships: "Which countries have diplomatic relations?"
            4: 0.15,  # Aggregations: "Count of relationships by type"
        },
        db_path=db_path  # For query validation
    )
    
    print("\n✅ Training data generated!")
    
    # Analyze the dataset
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total examples: {len(dataset)}")
    
    # Complexity and intent analysis
    complexity_counts = Counter(item.get('complexity', 0) for item in dataset)
    intent_counts = Counter(item.get('intent', 'unknown') for item in dataset)
    
    print(f"  Complexity distribution: {dict(complexity_counts)}")
    print(f"  Intent types: {len(intent_counts)} different intents")
    
    # Show sample examples focused on international relations
    print(f"\n💡 Sample Training Examples for International Relations:")
    
    # Find examples with interesting relations
    interesting_examples = []
    keywords = ['embassy', 'military', 'economic', 'alliance', 'tourism', 'trade', 'diplomatic']
    
    for example in dataset:
        question = example.get('question', '').lower()
        cypher = example.get('cypher', '').lower()
        if any(keyword in question or keyword in cypher for keyword in keywords):
            interesting_examples.append(example)
    
    # Show a mix of examples
    examples_to_show = interesting_examples[:3] + dataset[:2]  # 3 interesting + 2 random
    
    for i, example in enumerate(examples_to_show[:5], 1):
        print(f"\n  Example {i}:")
        print(f"    Question: {example.get('question', 'N/A')}")
        cypher = example.get('cypher', 'N/A')
        if len(cypher) > 80:
            cypher = cypher[:77] + "..."
        print(f"    Cypher: {cypher}")
        print(f"    Complexity: {example.get('complexity', 'N/A')}")
        print(f"    Intent: {example.get('intent', 'N/A')}")
    
    return dataset_path, dataset

def train_model(dataset_path, work_dir):
    """Fine-tune model using ModelTrainer."""
    print("\n🎯 Step 5: Fine-tune Model with ModelTrainer")
    print("⚠️  Note: Model training is computationally intensive. Using small, efficient configuration.")
    
    try:
        from auto_graph_rag.modules import ModelTrainer
    except ImportError:
        print("❌ Could not import ModelTrainer. Make sure auto-graph-rag is installed.")
        return None
    
    # Initialize ModelTrainer
    trainer = ModelTrainer()
    
    print("🔧 ModelTrainer Info:")
    info = trainer.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Configuration for efficient training
    model_config = {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",  # Small, efficient model
        "epochs": 10,  # Reduced for demo
        "learning_rate": 5e-4,
        "batch_size": 4,
        "lora_rank": 8,
        "output_dir": str(work_dir / "nations_model")
    }
    
    print(f"\n⚙️ Training Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Starting model training for international relations...")
    print(f"Training a specialized model to understand diplomatic, economic, and political relationships!")
    
    # Train the model
    model = trainer.train_from_file(
        dataset_path=dataset_path,
        model_name=model_config["base_model"],
        output_dir=Path(model_config["output_dir"]),
        epochs=model_config["epochs"],
        learning_rate=model_config["learning_rate"],
        batch_size=model_config["batch_size"],
        lora_rank=model_config["lora_rank"]
    )
    
    print("\n✅ Model training completed!")
    print(f"🎉 Nations-specialized model ready for international relations queries!")
    
    return work_dir / "nations_model"

def query_system(model_path, db_path):
    """Query the system using QueryExecutor."""
    print("\n💬 Step 6: Query the International Relations Graph with QueryExecutor")
    
    try:
        from auto_graph_rag.modules import QueryExecutor
    except ImportError:
        print("❌ Could not import QueryExecutor. Make sure auto-graph-rag is installed.")
        return
    
    # Initialize QueryExecutor
    executor = QueryExecutor()
    
    print("🔧 QueryExecutor Info:")
    info = executor.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Using Assets:")
    print(f"  Model: {model_path} (Nations-specialized model)")
    print(f"  Database: {db_path} (Nations knowledge graph)")
    
    # International relations test questions
    test_questions = [
        "What countries are in the dataset?",
        "Which countries have embassy relationships?",
        "What military alliances exist between countries?",
        "Which countries have economic aid relationships?",
        "What tourism relationships exist in the data?",
        "How many different types of relationships are there?",
        "Which countries have the most international relationships?",
        "What diplomatic relationships exist between USA and other countries?"
    ]
    
    print(f"\n❓ Test Questions for International Relations:")
    for i, question in enumerate(test_questions, 1):
        print(f"  {i}. {question}")
    
    print(f"\n🤖 Query Execution with Nations-Specialized Model:")
    print(f"Let's see how our model handles international relations queries!")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {question}")
        print(f"{'='*60}")
        
        result = executor.query_with_model(
            question=question,
            model_path=model_path,
            db_path=db_path,
            return_cypher=True,
            format_results=True
        )
        
        if result['success']:
            print(f"✅ Generated Cypher: {result['cypher']}")
            print(f"📊 Results: {result['count']} rows")
            
            if 'results' in result and result['results']:
                print(f"\n🔍 Sample Results:")
                # Show more results for international relations
                max_results = min(8, len(result['results']))
                for j, row in enumerate(result['results'][:max_results]):
                    print(f"    {j+1:2d}. {row}")
                if len(result['results']) > max_results:
                    print(f"    ... and {len(result['results']) - max_results} more")
        else:
            print(f"❌ Error: {result['error']}")
        
        print()  # Add space between queries

def advanced_analysis(executor, model_path, db_path):
    """Perform advanced analysis with sophisticated queries."""
    print("\n🔬 Advanced International Relations Analysis:")
    print("Let's explore deeper geopolitical patterns...\n")
    
    # Advanced questions
    advanced_questions = [
        "Which countries have both military and economic relationships?",
        "What are the different types of diplomatic activities?",
        "Which countries are most central in the international network?",
        "Are there any countries with negative relationships like boycotts or accusations?",
        "What cultural exchange relationships exist (books, students, tourism)?"
    ]
    
    for i, question in enumerate(advanced_questions, 1):
        print(f"🌍 Advanced Query {i}: {question}")
        
        result = executor.query_with_model(
            question=question,
            model_path=model_path,
            db_path=db_path,
            return_cypher=True,
            format_results=True
        )
        
        if result['success']:
            print(f"  💡 Query: {result['cypher']}")
            print(f"  📊 Found: {result['count']} results")
            
            if result['results']:
                # Show top results
                for j, row in enumerate(result['results'][:5]):
                    print(f"     • {row}")
                if len(result['results']) > 5:
                    print(f"     ... and {len(result['results']) - 5} more")
        else:
            print(f"  ❌ Error: {result['error']}")
        
        print()  # Space between queries

def print_summary():
    """Print final summary and insights."""
    print("\n" + "="*80)
    print("🎉 CONGRATULATIONS! Auto-Graph-RAG Nations Demo Complete!")
    print("="*80)
    
    print("""
### What we accomplished:

1. 📦 **Loaded Real Data**: Used the Nations dataset with 14 countries and 55 relationship types
2. 🏗️ **Built Graph Database**: Converted international relations into a queryable Kuzu database  
3. 🔍 **Schema Discovery**: Let GPT-4 understand the diplomatic, economic, and political relationships
4. 📝 **Generated Training Data**: Created 120+ question-Cypher pairs about international relations
5. 🎯 **Fine-tuned Model**: Trained Llama-3.2-1B to understand geopolitical queries
6. 💬 **Deployed QA System**: Built a natural language interface for exploring international relations

### Key Insights from the Nations Dataset:

- **Embassy relationships** are the most common diplomatic connection
- **Military alliances** and **economic aid** reveal geopolitical structures  
- **Cultural exchanges** through tourism, students, and book translations
- **Diplomatic activities** like conferences and official visits
- **Negative relationships** including boycotts and accusations

### Next Steps:

1. **Expand the Dataset**: Add more countries or time-series data
2. **Custom Relationships**: Define domain-specific relationship types  
3. **Multi-modal Data**: Include geographic, economic, or demographic data
4. **Production Deployment**: Scale for larger knowledge graphs

This demonstrates how Auto-Graph-RAG can transform any knowledge graph into an intelligent QA system!
""")

def main():
    """Main function to run the complete demo."""
    print("🌍 Auto-Graph-RAG Nations Dataset Demo")
    print("="*50)
    print("Demonstrating AI-powered question answering over international relations")
    print()
    
    # Setup environment
    work_dir, has_openai = setup_environment()
    if not has_openai:
        return 1
    
    try:
        # Step 1: Load and visualize dataset
        nations_graph, relation_counts = load_nations_dataset()
        visualize_dataset(relation_counts) 
        show_example_relationships(nations_graph)
        
        # Step 2: Build graph database
        db_path = build_graph_database(nations_graph, work_dir)
        if not db_path:
            return 1
        
        # Step 3: Explore schema
        schema_path, schema = explore_schema(db_path, work_dir)
        if not schema_path:
            return 1
        
        # Step 4: Generate training data
        dataset_path, dataset = generate_training_data(schema_path, work_dir, db_path)
        if not dataset_path:
            return 1
        
        # Step 5: Train model
        model_path = train_model(dataset_path, work_dir)
        if not model_path:
            return 1
        
        # Step 6: Query system
        from auto_graph_rag.modules import QueryExecutor
        executor = QueryExecutor()
        query_system(model_path, db_path)
        
        # Advanced analysis
        advanced_analysis(executor, model_path, db_path)
        
        # Summary
        print_summary()
        
        print(f"\n📁 All assets saved in: {work_dir.absolute()}")
        print("✅ Demo completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())