"""Standalone module for building graphs in Kuzu database."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import networkx as nx
import json
import argparse
import logging

from .base import GraphBuilderInterface
from ..ingestion.networkx_loader import NetworkXLoader
from ..ingestion.kuzu_adapter import KuzuAdapter


logger = logging.getLogger(__name__)


class GraphBuilder(GraphBuilderInterface):
    """Build Kuzu graphs from various sources."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.nx_loader = NetworkXLoader()
        self.kuzu_adapter = None
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        if 'db_path' not in kwargs:
            raise ValueError("db_path is required")
        
        if 'graph' in kwargs:
            if not isinstance(kwargs['graph'], nx.Graph):
                raise ValueError("graph must be a NetworkX Graph")
        elif 'nodes' in kwargs and 'edges' in kwargs:
            if not isinstance(kwargs['nodes'], list) or not isinstance(kwargs['edges'], list):
                raise ValueError("nodes and edges must be lists")
        else:
            raise ValueError("Either 'graph' or 'nodes' and 'edges' are required")
        
        return True
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute graph building."""
        if 'graph' in kwargs:
            return self.build_from_networkx(**kwargs)
        else:
            return self.build_from_data(**kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            "name": "GraphBuilder",
            "version": "1.0.0",
            "description": "Build Kuzu graphs from NetworkX or raw data",
            "inputs": {
                "networkx": ["graph", "db_path", "graph_name", "node_labels", "edge_labels"],
                "raw_data": ["nodes", "edges", "db_path", "graph_name"]
            },
            "outputs": ["statistics", "db_path"]
        }
    
    def build_from_networkx(
        self,
        graph: nx.Graph,
        db_path: Path,
        graph_name: str = "default",
        node_labels: Optional[Dict[Any, str]] = None,
        edge_labels: Optional[Dict[Any, str]] = None
    ) -> Dict[str, Any]:
        """Build graph database from NetworkX graph."""
        db_path = Path(db_path)
        logger.info(f"Building graph '{graph_name}' at {db_path}")
        
        # Process the graph
        processed_graph = self.nx_loader.process_graph(
            graph, node_labels, edge_labels
        )
        
        # Create Kuzu database
        self.kuzu_adapter = KuzuAdapter(db_path)
        stats = self.kuzu_adapter.create_from_networkx(
            processed_graph, graph_name
        )
        
        stats['db_path'] = str(db_path)
        stats['graph_name'] = graph_name
        
        logger.info(f"Created {stats['total_nodes']} nodes and {stats['total_edges']} edges")
        return stats
    
    def build_from_data(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        db_path: Path,
        graph_name: str = "default"
    ) -> Dict[str, Any]:
        """Build graph database from raw data."""
        # Create NetworkX graph from raw data
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            node_id = node.pop('id')
            G.add_node(node_id, **node)
        
        # Add edges  
        for edge in edges:
            source = edge.pop('source')
            target = edge.pop('target')
            G.add_edge(source, target, **edge)
        
        # Extract labels if present
        node_labels = {n: data.get('type', 'Node') for n, data in G.nodes(data=True)}
        edge_labels = {(u, v): data.get('type', 'Edge') for u, v, data in G.edges(data=True)}
        
        return self.build_from_networkx(
            G, db_path, graph_name, node_labels, edge_labels
        )


def main():
    """CLI entry point for standalone graph building."""
    parser = argparse.ArgumentParser(description='Build Kuzu graph database')
    parser.add_argument('--input', required=True, help='Input file (JSON or Python with NetworkX)')
    parser.add_argument('--db-path', required=True, help='Output database path')
    parser.add_argument('--graph-name', default='default', help='Graph name')
    parser.add_argument('--format', choices=['json', 'networkx'], default='json',
                       help='Input format')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    builder = GraphBuilder()
    
    if args.format == 'json':
        # Load JSON data
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        stats = builder.build_from_data(
            nodes=data['nodes'],
            edges=data['edges'],
            db_path=Path(args.db_path),
            graph_name=args.graph_name
        )
    else:
        # Import NetworkX graph from Python file
        import importlib.util
        spec = importlib.util.spec_from_file_location("graph_module", args.input)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        graph = module.graph  # Assumes the file exports a 'graph' variable
        node_labels = getattr(module, 'node_labels', None)
        edge_labels = getattr(module, 'edge_labels', None)
        
        stats = builder.build_from_networkx(
            graph=graph,
            db_path=Path(args.db_path),
            graph_name=args.graph_name,
            node_labels=node_labels,
            edge_labels=edge_labels
        )
    
    print(f"✅ Graph built successfully!")
    print(f"📊 Statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()