"""Kuzu database adapter for graph storage and querying."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import kuzu
from .networkx_loader import ProcessedGraph


class KuzuAdapter:
    """Adapter for Kuzu embedded graph database."""
    
    def __init__(self, db_path: Path):
        """Initialize Kuzu database connection.
        
        Args:
            db_path: Path to database directory
        """
        self.db_path = Path(db_path)
        
        # Create parent directory if needed, but not the database directory itself
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing database if it exists
        if self.db_path.exists():
            import shutil
            if self.db_path.is_dir():
                shutil.rmtree(self.db_path)
            else:
                self.db_path.unlink()  # Remove file
        
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        
        # Track created tables for schema introspection
        self.created_node_tables = []
        self.created_edge_tables = []
        
        # Track inserted nodes to avoid duplicates
        self.inserted_nodes = set()
        
    def create_from_networkx(
        self,
        processed_graph: ProcessedGraph,
        graph_name: str = "default"
    ) -> Dict[str, Any]:
        """Create Kuzu tables from processed NetworkX graph.
        
        Args:
            processed_graph: ProcessedGraph object
            graph_name: Name prefix for tables
            
        Returns:
            Statistics about created tables
        """
        # Clear tracking for fresh ingestion
        self.created_node_tables = []
        self.created_edge_tables = []
        self.inserted_nodes = set()
        
        stats = {
            "node_tables": [],
            "edge_tables": [],
            "total_nodes": 0,
            "total_edges": 0
        }
        
        # Create node tables
        for node_type, properties in processed_graph.node_properties.items():
            table_name = f"{graph_name}_{node_type}"
            
            # Get nodes of this type for type inference
            nodes_of_type = [
                n for n in processed_graph.nodes
                if n["type"] == node_type
            ]
            
            # Create table with type inference
            self._create_node_table(table_name, properties, nodes_of_type)
            stats["node_tables"].append(table_name)
            
            # Track for schema introspection
            self.created_node_tables.append({
                "name": table_name,
                "type": node_type,
                "properties": list(properties)
            })
            
            # Insert nodes
            self._insert_nodes(table_name, nodes_of_type)
            stats["total_nodes"] += len(nodes_of_type)
        
        # Create edge tables - one per edge type with unique source-target combinations
        created_edge_tables_set = set()
        for edge_type in processed_graph.edge_types:
            # Get unique source-target type combinations
            type_combos = set(processed_graph.edge_types[edge_type])
            
            for source_type, target_type in type_combos:
                # Create unique table name for each source-target combination
                table_name = f"{graph_name}_{edge_type}_{source_type}_to_{target_type}"
                
                # Skip if already created
                if table_name in created_edge_tables_set:
                    continue
                created_edge_tables_set.add(table_name)
                
                source_table = f"{graph_name}_{source_type}"
                target_table = f"{graph_name}_{target_type}"
                
                # Get edges of this type for type inference
                edges_of_type = [
                    e for e in processed_graph.edges
                    if e["type"] == edge_type
                    and e["source_type"] == source_type
                    and e["target_type"] == target_type
                ]
                
                properties = processed_graph.edge_properties.get(edge_type, set())
                self._create_edge_table(
                    table_name, source_table, target_table, properties, edges_of_type
                )
                stats["edge_tables"].append(table_name)
                
                # Track for schema introspection
                self.created_edge_tables.append({
                    "name": table_name,
                    "type": edge_type,
                    "source": source_type,
                    "target": target_type,
                    "properties": list(properties)
                })
                
                # Insert edges (edges_of_type already retrieved above)
                self._insert_edges(table_name, edges_of_type, source_table, target_table)
                stats["total_edges"] += len(edges_of_type)
        
        return stats
    
    def _create_node_table(self, table_name: str, properties: set, nodes_sample: list = None):
        """Create a node table with properties."""
        # Drop table if it exists to ensure clean slate
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass
        
        # Build property schema with type inference
        prop_schema = ["id STRING PRIMARY KEY"]
        prop_types = {}
        
        # Infer types from sample nodes if provided
        if nodes_sample:
            for node in nodes_sample[:10]:  # Sample first 10 nodes
                for key, value in node.items():
                    if key not in ["id", "type"] and key not in prop_types:
                        # Infer type based on value
                        if isinstance(value, (int, float)):
                            prop_types[key] = "DOUBLE"
                        elif isinstance(value, bool):
                            prop_types[key] = "BOOLEAN"
                        else:
                            prop_types[key] = "STRING"
        
        for prop in properties:
            if prop not in ["id", "type"]:  # Skip reserved
                # Use inferred type or default to STRING
                data_type = prop_types.get(prop, "STRING")
                prop_schema.append(f"{prop} {data_type}")
        
        create_query = f"""
        CREATE NODE TABLE {table_name} (
            {', '.join(prop_schema)}
        )
        """
        
        self.conn.execute(create_query)
    
    def _create_edge_table(
        self,
        table_name: str,
        source_table: str,
        target_table: str,
        properties: set,
        edges_sample: list = None
    ):
        """Create an edge table with properties."""
        # Drop table if it exists to ensure clean slate
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass
        
        # Build property schema with type inference
        prop_schema = []
        prop_types = {}
        
        # Infer types from sample edges if provided
        if edges_sample:
            for edge in edges_sample[:10]:
                for key, value in edge.items():
                    if key not in ["source", "target", "type", "source_type", "target_type"] and key not in prop_types:
                        if isinstance(value, (int, float)):
                            prop_types[key] = "DOUBLE"
                        elif isinstance(value, bool):
                            prop_types[key] = "BOOLEAN" 
                        else:
                            prop_types[key] = "STRING"
        
        for prop in properties:
            if prop not in ["source", "target", "type"]:  # Skip reserved
                data_type = prop_types.get(prop, "STRING")
                prop_schema.append(f"{prop} {data_type}")
        
        props_str = f", {', '.join(prop_schema)}" if prop_schema else ""
        
        create_query = f"""
        CREATE REL TABLE {table_name} (
            FROM {source_table} TO {target_table}{props_str}
        )
        """
        
        self.conn.execute(create_query)
    
    def _insert_nodes(self, table_name: str, nodes: List[Dict[str, Any]]):
        """Insert nodes into table."""
        for node in nodes:
            # Check if node already inserted (avoid duplicates)
            node_id = node.get("id")
            if node_id and f"{table_name}:{node_id}" in self.inserted_nodes:
                continue
            
            # Prepare values
            columns = []
            values = []
            for key, value in node.items():
                if key != "type":  # Skip type as it's in table name
                    columns.append(key)
                    # Format value based on type
                    if value is None:
                        values.append("NULL")
                    elif isinstance(value, (int, float)):
                        values.append(str(value))  # No quotes for numbers
                    elif isinstance(value, bool):
                        values.append("true" if value else "false")
                    else:
                        values.append(f"'{str(value)}'")
            
            insert_query = f"""
            CREATE (:{table_name} {{
                {', '.join(f'{col}: {val}' for col, val in zip(columns, values))}
            }})
            """
            self.conn.execute(insert_query)
            
            # Track inserted node
            if node_id:
                self.inserted_nodes.add(f"{table_name}:{node_id}")
    
    def _insert_edges(self, table_name: str, edges: List[Dict[str, Any]], source_table: str, target_table: str):
        """Insert edges into table."""
        for edge in edges:
            source_id = edge["source"]
            target_id = edge["target"]
            
            # Prepare properties
            props = []
            for key, value in edge.items():
                if key not in ["source", "target", "type", "source_type", "target_type"]:
                    # Format value based on type
                    if value is None:
                        formatted_value = "NULL"
                    elif isinstance(value, (int, float)):
                        formatted_value = str(value)  # No quotes for numbers
                    elif isinstance(value, bool):
                        formatted_value = "true" if value else "false"
                    else:
                        formatted_value = f"'{str(value)}'"
                    props.append(f"{key}: {formatted_value}")
            
            props_str = f" {{{', '.join(props)}}}" if props else ""
            
            insert_query = f"""
            MATCH (s:{source_table} {{id: '{source_id}'}}),
                  (t:{target_table} {{id: '{target_id}'}})
            CREATE (s)-[:{table_name}{props_str}]->(t)
            """
            
            self.conn.execute(insert_query)
    
    def execute_cypher(self, query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            
        Returns:
            Query results as list of dictionaries
        """
        result = self.conn.execute(query)
        
        # Convert to list of dictionaries
        results = []
        while result.has_next():
            row = result.get_next()
            results.append(dict(zip(result.get_column_names(), row)))
        
        return results
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about database schema.
        
        Returns:
            Schema information
        """
        # Use the tracked tables from ingestion
        schema = {
            "node_tables": [],
            "edge_tables": [],
            "statistics": {}
        }
        
        # Add node tables
        for table_info in self.created_node_tables:
            schema["node_tables"].append({
                "name": table_info["name"],
                "properties": table_info["properties"]
            })
        
        # Add edge tables  
        for table_info in self.created_edge_tables:
            schema["edge_tables"].append({
                "name": table_info["name"],
                "from": table_info["source"],
                "to": table_info["target"],
                "properties": table_info["properties"]
            })
        
        # Add basic statistics
        schema["statistics"] = {
            "num_node_tables": len(schema["node_tables"]),
            "num_edge_tables": len(schema["edge_tables"])
        }
        
        return schema