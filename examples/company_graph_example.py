"""Example usage of Auto-Graph-RAG with a company knowledge graph."""

import networkx as nx
from auto_graph_rag import GraphRAG
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_company_graph():
    """Create a sample company graph for demonstration."""
    G = nx.DiGraph()
    
    # Add employees
    employees = [
        ("emp1", {"name": "Alice Johnson", "department": "Engineering", "salary": 120000.0, "level": "Senior"}),
        ("emp2", {"name": "Bob Smith", "department": "Engineering", "salary": 95000.0, "level": "Junior"}),
        ("emp3", {"name": "Carol White", "department": "Marketing", "salary": 85000.0, "level": "Mid"}),
        ("emp4", {"name": "David Brown", "department": "Sales", "salary": 90000.0, "level": "Senior"}),
        ("emp5", {"name": "Eve Davis", "department": "Engineering", "salary": 110000.0, "level": "Mid"}),
    ]
    
    for emp_id, attrs in employees:
        G.add_node(emp_id, type="Employee", **attrs)
    
    # Add projects
    projects = [
        ("proj1", {"name": "Alpha", "budget": 500000.0, "status": "Active", "deadline": "2024-12-31"}),
        ("proj2", {"name": "Beta", "budget": 300000.0, "status": "Planning", "deadline": "2025-03-31"}),
        ("proj3", {"name": "Gamma", "budget": 750000.0, "status": "Active", "deadline": "2024-09-30"}),
    ]
    
    for proj_id, attrs in projects:
        G.add_node(proj_id, type="Project", **attrs)
    
    # Add departments
    departments = [
        ("dept1", {"name": "Engineering", "budget": 2000000.0, "head_count": 25}),
        ("dept2", {"name": "Marketing", "budget": 800000.0, "head_count": 10}),
        ("dept3", {"name": "Sales", "budget": 1200000.0, "head_count": 15}),
    ]
    
    for dept_id, attrs in departments:
        G.add_node(dept_id, type="Department", **attrs)
    
    # Add relationships
    # Employees work on projects
    G.add_edge("emp1", "proj1", type="WORKED_ON", hours=320.0, role="Lead Developer")
    G.add_edge("emp1", "proj3", type="WORKED_ON", hours=160.0, role="Architect")
    G.add_edge("emp2", "proj1", type="WORKED_ON", hours=480.0, role="Developer")
    G.add_edge("emp3", "proj2", type="WORKED_ON", hours=200.0, role="Marketing Lead")
    G.add_edge("emp4", "proj2", type="WORKED_ON", hours=150.0, role="Sales Consultant")
    G.add_edge("emp5", "proj1", type="WORKED_ON", hours=400.0, role="Developer")
    G.add_edge("emp5", "proj3", type="WORKED_ON", hours=240.0, role="Developer")
    
    # Employees belong to departments
    G.add_edge("emp1", "dept1", type="BELONGS_TO", since="2020-01-15")
    G.add_edge("emp2", "dept1", type="BELONGS_TO", since="2023-03-01")
    G.add_edge("emp3", "dept2", type="BELONGS_TO", since="2021-06-15")
    G.add_edge("emp4", "dept3", type="BELONGS_TO", since="2019-09-01")
    G.add_edge("emp5", "dept1", type="BELONGS_TO", since="2022-01-10")
    
    # Employees report to other employees (management structure)
    G.add_edge("emp2", "emp1", type="REPORTS_TO")
    G.add_edge("emp5", "emp1", type="REPORTS_TO")
    G.add_edge("emp3", "emp4", type="REPORTS_TO")
    
    # Projects belong to departments
    G.add_edge("proj1", "dept1", type="OWNED_BY")
    G.add_edge("proj2", "dept2", type="OWNED_BY")
    G.add_edge("proj3", "dept1", type="OWNED_BY")
    
    return G


def main():
    """Main example workflow."""
    
    logger.info("=" * 60)
    logger.info("Auto-Graph-RAG Example: Company Knowledge Graph")
    logger.info("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found!")
        logger.info("Please create a .env file in the project root with:")
        logger.info("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Step 1: Initialize the system
    logger.info("\nStep 1: Initializing GraphRAG system...")
    rag = GraphRAG(
        llm_provider="openai",
        llm_model="gpt-4",
        target_model='meta-llama/Llama-3.2-1B-Instruct',  # You have access!
        db_path="./example_kuzu_db"
    )
    
    # Step 2: Create and ingest graph
    logger.info("\nStep 2: Creating sample company graph...")
    graph = create_sample_company_graph()
    
    logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Define node and edge labels
    node_labels = {node: data["type"] for node, data in graph.nodes(data=True)}
    edge_labels = {(u, v): data["type"] for u, v, data in graph.edges(data=True)}
    
    logger.info("\nIngesting graph into Kuzu database...")
    stats = rag.ingest_graph(
        graph,
        name="company",
        node_labels=node_labels,
        edge_labels=edge_labels
    )
    logger.info(f"Ingestion stats: {stats}")
    
    # Step 3: Explore and learn schema
    logger.info("\nStep 3: Exploring graph to learn schema...")
    schema = rag.explore_graph(
        max_samples=20,
        save_to="company_schema.json"
    )
    
    logger.info("\nLearned Schema:")
    logger.info(f"Summary: {schema.get('summary', 'N/A')}")
    logger.info(f"Node types: {list(schema.get('nodes', {}).keys())}")
    logger.info(f"Edge types: {list(schema.get('edges', {}).keys())}")
    
    # Step 4: Generate training data
    logger.info("\nStep 4: Generating training data...")
    logger.info("This will generate diverse question-cypher pairs...")
    
    dataset = rag.generate_training_data(
        num_examples=150,  # Reduced for faster demo
        complexity_distribution={
            1: 0.2,  # Simple lookups
            2: 0.2,  # Filtered queries
            3: 0.2,  # Relationships
            4: 0.2,  # Skip aggregations for speed
            5: 0.2   # Skip complex paths for speed
        },
        save_to="company_training_data.jsonl"
    )
    
    logger.info(f"Generated {len(dataset)} training examples")
    
    # Show some examples
    logger.info("\nSample training data:")
    for i, example in enumerate(dataset[:3], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Question: {example.get('question', 'N/A')}")
        logger.info(f"  Cypher: {example.get('cypher', 'N/A')}")
        logger.info(f"  Complexity: {example.get('complexity', 'N/A')}")
    
    # Step 5: Fine-tune model
    logger.info("\nStep 5: Fine-tuning model...")
    logger.info("Note: Fine-tuning may take time depending on your hardware.")
    
    try:
        model = rag.fine_tune(
            dataset,
            epochs=10,  # Two epochs to get more loss points
            learning_rate=5e-4,  # Higher LR for faster convergence
            batch_size=1,  # Minimal batch size for CPU
            lora_rank=8,  # Smaller LoRA rank for efficiency
            output_dir="./models/company-cypher-model"
        )
        logger.info("✅ Fine-tuning completed successfully!")
    except Exception as e:
        logger.warning(f"⚠️ Fine-tuning failed: {e}")
        logger.info("This is often due to hardware limitations or missing dependencies.")
    
    # Step 6: Query the graph using the fine-tuned model
    logger.info("\nStep 6: Querying the graph...")
    logger.info("Testing the fine-tuned model (if available)...")
    
    test_questions = [
        "Who worked on project Alpha?",
        "What is the average salary in Engineering?", 
        "Find all employees who report to Alice Johnson",
        "Which projects are owned by the Engineering department?",
        "Count the number of active projects"
    ]
    
    logger.info("\nTest Queries:")
    for question in test_questions:
        logger.info(f"\nQ: {question}")
        try:
            # Try to use the fine-tuned model
            result = rag.query(question)
            if result.get("success"):
                logger.info(f"A: Found {result.get('count', 0)} results")
                if result.get("formatted"):
                    logger.info(f"   {result['formatted'][:200]}...")  # Show first 200 chars
            else:
                logger.warning(f"   Query failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"   Query execution failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()