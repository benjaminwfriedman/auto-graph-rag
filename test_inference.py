#!/usr/bin/env python3
"""Test script for inference using an existing trained model."""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from auto_graph_rag import GraphRAG
from auto_graph_rag.inference.query_engine import QueryEngine
from auto_graph_rag.ingestion.kuzu_adapter import KuzuAdapter
import networkx as nx

# Load environment variables
load_dotenv()

# Setup logging with cleaner format
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_company_graph():
    """Recreate the company graph to query against."""
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
    G.add_edge("emp1", "proj1", type="WORKED_ON", hours=320, role="Lead Developer")
    G.add_edge("emp1", "proj3", type="WORKED_ON", hours=160, role="Architect")
    G.add_edge("emp2", "proj1", type="WORKED_ON", hours=480, role="Developer")
    G.add_edge("emp3", "proj2", type="WORKED_ON", hours=200, role="Marketing Lead")
    G.add_edge("emp4", "proj2", type="WORKED_ON", hours=150, role="Sales Consultant")
    G.add_edge("emp5", "proj1", type="WORKED_ON", hours=400, role="Developer")
    G.add_edge("emp5", "proj3", type="WORKED_ON", hours=240, role="Developer")
    
    G.add_edge("emp1", "dept1", type="BELONGS_TO", since="2020-01-15")
    G.add_edge("emp2", "dept1", type="BELONGS_TO", since="2023-03-01")
    G.add_edge("emp3", "dept2", type="BELONGS_TO", since="2021-06-15")
    G.add_edge("emp4", "dept3", type="BELONGS_TO", since="2019-09-01")
    G.add_edge("emp5", "dept1", type="BELONGS_TO", since="2022-01-10")
    
    G.add_edge("emp2", "emp1", type="REPORTS_TO")
    G.add_edge("emp5", "emp1", type="REPORTS_TO")
    G.add_edge("emp3", "emp4", type="REPORTS_TO")
    
    G.add_edge("proj1", "dept1", type="OWNED_BY")
    G.add_edge("proj2", "dept2", type="OWNED_BY")
    G.add_edge("proj3", "dept1", type="OWNED_BY")
    
    return G


def main():
    """Test inference with an existing trained model."""
    
    print("\n" + "=" * 60)
    print("🔍 INFERENCE TEST - Using Trained Model")
    print("=" * 60)
    
    # Configuration
    model_path = "./models/company-cypher-model"  # Path to trained model
    db_path = "./inference_test_kuzu_db"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Original model name for format detection
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.info("Please run the company_graph_example.py first to train a model")
        return
    
    logger.info(f"✅ Found trained model at: {model_path}")
    
    # Setup Kuzu database with company graph
    logger.info("\n📊 Setting up test database...")
    from auto_graph_rag.ingestion.networkx_loader import NetworkXLoader
    
    # Create fresh database
    kuzu_adapter = KuzuAdapter(Path(db_path))
    nx_loader = NetworkXLoader()
    
    # Load company graph
    graph = create_company_graph()
    node_labels = {node: data["type"] for node, data in graph.nodes(data=True)}
    edge_labels = {(u, v): data["type"] for u, v, data in graph.edges(data=True)}
    
    processed_graph = nx_loader.process_graph(graph, node_labels, edge_labels)
    stats = kuzu_adapter.create_from_networkx(processed_graph, "company")
    
    logger.info(f"  Loaded {stats['total_nodes']} nodes and {stats['total_edges']} edges")
    
    # Initialize query engine with the trained model
    logger.info("\n🤖 Loading fine-tuned model...")
    query_engine = QueryEngine(
        model_path=model_path,
        kuzu_adapter=kuzu_adapter,
        model_name=model_name
    )
    
    # Test queries - organized by complexity
    test_queries = {
        "Simple Lookups": [
            "List all employees",
            "Show all departments",
            "Find all projects"
        ],
        "Filtered Queries": [
            "Who works in Engineering?",
            "Find employees with salary over 100000",
            "Which projects are active?",
            "Show Senior level employees"
        ],
        "Relationship Queries": [
            "Who works on project Alpha?",
            "Which employees report to Alice Johnson?",
            "What projects are owned by Engineering?",
            "Who belongs to the Marketing department?"
        ],
        "Aggregation Queries": [
            "What is the average salary in Engineering?",
            "Count the number of active projects",
            "How many employees are in each department?",
            "What's the total budget for all projects?"
        ],
        "Complex Queries": [
            "Find employees in Engineering who work on active projects",
            "Show the management hierarchy",
            "Which department has the highest average salary?",
            "List employees and their project count"
        ]
    }
    
    # Run inference tests
    print("\n" + "=" * 60)
    print("🚀 RUNNING INFERENCE TESTS")
    print("=" * 60)
    
    total_queries = 0
    successful_queries = 0
    
    for category, queries in test_queries.items():
        print(f"\n📝 {category}:")
        print("-" * 40)
        
        for question in queries:
            total_queries += 1
            print(f"\n❓ Question: {question}")
            
            try:
                result = query_engine.query(
                    question=question,
                    max_retries=2,
                    temperature=0.1,
                    return_cypher=True,
                    format_results=True
                )
                
                if result.get("success"):
                    successful_queries += 1
                    cypher = result.get("cypher", "N/A")
                    # Clean up cypher for display
                    cypher_display = cypher.replace('\n', ' ').strip()
                    if len(cypher_display) > 100:
                        cypher_display = cypher_display[:100] + "..."
                    
                    print(f"✅ Cypher: {cypher_display}")
                    print(f"📊 Results: {result.get('count', 0)} rows found")
                    
                    # Show sample results if available
                    if result.get("results") and len(result["results"]) > 0:
                        sample = result["results"][0]
                        print(f"   Sample: {str(sample)[:80]}...")
                else:
                    error = result.get("error", "Unknown error")
                    print(f"⚠️  Failed: {error[:100]}...")
                    
                    # Show the generated Cypher that failed
                    if result.get("cypher"):
                        print(f"   Attempted: {result['cypher'][:100]}...")
                        
            except Exception as e:
                print(f"❌ Error: {str(e)[:100]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total queries tested: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Success rate: {(successful_queries/total_queries)*100:.1f}%")
    
    if successful_queries < total_queries:
        print("\n💡 Tips for improving success rate:")
        print("  - Train for more epochs (10-20)")
        print("  - Use more training examples (500+)")
        print("  - Include more diverse query patterns in training")
        print("  - Fine-tune the temperature parameter")
    
    print("\n✨ Inference test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()