"""Schema learning and memory management."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class SchemaLearner:
    """Learn and store graph schema understanding."""
    
    def __init__(self, memory_path: Optional[Path] = None):
        """Initialize schema learner.
        
        Args:
            memory_path: Path to store learned schemas
        """
        self.memory_path = memory_path or Path("./schema_memory")
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.current_schema = None
    
    def save_schema(
        self,
        schema: Dict[str, Any],
        graph_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save learned schema to memory.
        
        Args:
            schema: Graph schema
            graph_name: Name of the graph
            metadata: Additional metadata
            
        Returns:
            Path to saved schema file
        """
        # Add metadata
        schema_with_meta = {
            "schema": schema,
            "graph_name": graph_name,
            "learned_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save to file
        file_path = self.memory_path / f"{graph_name}_schema.json"
        with open(file_path, 'w') as f:
            json.dump(schema_with_meta, f, indent=2)
        
        self.current_schema = schema
        return file_path
    
    def load_schema(self, graph_name: str) -> Optional[Dict[str, Any]]:
        """Load previously learned schema.
        
        Args:
            graph_name: Name of the graph
            
        Returns:
            Schema if found, None otherwise
        """
        file_path = self.memory_path / f"{graph_name}_schema.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.current_schema = data["schema"]
                return data["schema"]
        
        return None
    
    def update_schema(
        self,
        graph_name: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing schema with new information.
        
        Args:
            graph_name: Name of the graph
            updates: Updates to apply
            
        Returns:
            Updated schema
        """
        schema = self.load_schema(graph_name) or {"nodes": {}, "edges": {}}
        
        # Merge updates
        if "nodes" in updates:
            schema["nodes"].update(updates["nodes"])
        if "edges" in updates:
            schema["edges"].update(updates["edges"])
        if "summary" in updates:
            schema["summary"] = updates["summary"]
        
        # Save updated schema
        self.save_schema(schema, graph_name)
        
        return schema
    
    def get_statistics(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics about the schema.
        
        Args:
            schema: Graph schema
            
        Returns:
            Schema statistics
        """
        stats = {
            "num_node_types": len(schema.get("nodes", {})),
            "num_edge_types": len(schema.get("edges", {})),
            "total_properties": 0,
            "node_properties": {},
            "edge_properties": {},
            "relationship_patterns": []
        }
        
        # Node statistics
        for node_type, info in schema.get("nodes", {}).items():
            props = info.get("properties", [])
            stats["node_properties"][node_type] = len(props)
            stats["total_properties"] += len(props)
        
        # Edge statistics
        for edge_type, info in schema.get("edges", {}).items():
            props = info.get("properties", [])
            stats["edge_properties"][edge_type] = len(props)
            stats["total_properties"] += len(props)
            
            # Track relationship patterns
            pattern = f"{info.get('source')} -[{edge_type}]-> {info.get('target')}"
            stats["relationship_patterns"].append(pattern)
        
        return stats