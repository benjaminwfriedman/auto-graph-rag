"""Schema extraction utilities that work without LLM."""

from typing import Dict, Any
from pathlib import Path
import json

from ..ingestion.kuzu_adapter import KuzuAdapter


def extract_basic_schema(db_path: Path) -> Dict[str, Any]:
    """Extract basic schema from Kuzu database without LLM analysis.
    
    This provides a fallback when LLM-based exploration isn't available.
    
    Args:
        db_path: Path to Kuzu database
        
    Returns:
        Basic schema dictionary
    """
    adapter = KuzuAdapter(db_path)
    db_schema = adapter.get_schema_info()
    
    # Convert to the expected schema format
    schema = {
        "summary": f"Graph with {len(db_schema['node_tables'])} node types and {len(db_schema['edge_tables'])} edge types",
        "nodes": {},
        "edges": {}
    }
    
    # Process node tables
    for table in db_schema["node_tables"]:
        # Extract node type from table name (remove prefix)
        node_type = table["name"].split("_", 1)[-1] if "_" in table["name"] else table["name"]
        
        schema["nodes"][node_type] = {
            "properties": [prop for prop in table["properties"] if prop not in ["id", "type"]],
            "description": f"{node_type} entity from database",
            "example_values": {}  # Would need sample data to populate
        }
    
    # Process edge tables
    for table in db_schema["edge_tables"]:
        # Extract edge type from table name
        parts = table["name"].split("_")
        if len(parts) >= 2:
            edge_type = parts[1]  # Assumes format: prefix_EDGETYPE_source_to_target
        else:
            edge_type = table["name"]
        
        if edge_type not in schema["edges"]:
            schema["edges"][edge_type] = {
                "source": table["from"],
                "target": table["to"], 
                "properties": [prop for prop in table["properties"] if prop not in ["id", "type"]],
                "description": f"{edge_type} relationship",
                "cardinality": "unknown"  # Would need analysis to determine
            }
    
    return schema


def enhance_schema_with_samples(schema: Dict[str, Any], db_path: Path, max_samples: int = 10) -> Dict[str, Any]:
    """Enhance basic schema with sample data from the database.
    
    Args:
        schema: Basic schema to enhance
        db_path: Path to Kuzu database
        max_samples: Maximum samples to retrieve per type
        
    Returns:
        Enhanced schema with example values
    """
    adapter = KuzuAdapter(db_path)
    
    # Sample data for each node type
    for node_type, node_info in schema["nodes"].items():
        try:
            # Try to query the table (assuming standard naming)
            # This is a simple heuristic - might need adjustment based on actual table names
            possible_table_names = [
                f"company_{node_type}",  # Standard format
                f"default_{node_type}",
                node_type.lower(),
                node_type
            ]
            
            for table_name in possible_table_names:
                try:
                    query = f"MATCH (n:{node_type}) RETURN n LIMIT {max_samples}"
                    results = adapter.execute_query(query)
                    
                    if results:
                        # Extract example values from first result
                        first_result = results[0]
                        example_values = {}
                        
                        for key, value in first_result.items():
                            if key.startswith("n.") and not key.endswith(".id"):
                                prop_name = key[2:]  # Remove "n." prefix
                                example_values[prop_name] = value
                        
                        node_info["example_values"] = example_values
                        break
                except:
                    continue
        except:
            pass  # Skip if can't sample this node type
    
    return schema