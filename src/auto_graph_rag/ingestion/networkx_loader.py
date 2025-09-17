"""NetworkX graph loader and processor."""

from typing import Dict, Any, Optional, List, Tuple
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class ProcessedGraph:
    """Processed graph data ready for Kuzu ingestion."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    node_types: Dict[str, List[str]] = field(default_factory=dict)
    edge_types: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    node_properties: Dict[str, set] = field(default_factory=dict)
    edge_properties: Dict[str, set] = field(default_factory=dict)


class NetworkXLoader:
    """Load and process NetworkX graphs for Kuzu ingestion."""
    
    def process_graph(
        self,
        graph: nx.Graph,
        node_labels: Optional[Dict[Any, str]] = None,
        edge_labels: Optional[Dict[Tuple[Any, Any], str]] = None,
    ) -> ProcessedGraph:
        """Process NetworkX graph into structured format.
        
        Args:
            graph: NetworkX graph
            node_labels: Optional mapping of node IDs to type labels
            edge_labels: Optional mapping of edge tuples to type labels
            
        Returns:
            ProcessedGraph object
        """
        processed = ProcessedGraph()
        
        # Process nodes
        for node_id, node_data in graph.nodes(data=True):
            node_type = self._get_node_type(node_id, node_data, node_labels)
            
            node_entry = {
                "id": str(node_id),
                "type": node_type,
                **node_data
            }
            
            processed.nodes.append(node_entry)
            
            # Track node types and properties
            if node_type not in processed.node_types:
                processed.node_types[node_type] = []
                processed.node_properties[node_type] = set()
            
            processed.node_types[node_type].append(str(node_id))
            processed.node_properties[node_type].update(node_data.keys())
        
        # Process edges
        for source, target, edge_data in graph.edges(data=True):
            edge_type = self._get_edge_type(
                source, target, edge_data, edge_labels
            )
            
            source_type = self._get_node_type(
                source, graph.nodes[source], node_labels
            )
            target_type = self._get_node_type(
                target, graph.nodes[target], node_labels
            )
            
            edge_entry = {
                "source": str(source),
                "target": str(target),
                "source_type": source_type,
                "target_type": target_type,
                "type": edge_type,
                **edge_data
            }
            
            processed.edges.append(edge_entry)
            
            # Track edge types and properties
            if edge_type not in processed.edge_types:
                processed.edge_types[edge_type] = []
                processed.edge_properties[edge_type] = set()
            
            processed.edge_types[edge_type].append((source_type, target_type))
            processed.edge_properties[edge_type].update(edge_data.keys())
        
        return processed
    
    def _get_node_type(
        self,
        node_id: Any,
        node_data: Dict[str, Any],
        node_labels: Optional[Dict[Any, str]]
    ) -> str:
        """Determine node type from labels or data."""
        if node_labels and node_id in node_labels:
            return node_labels[node_id]
        
        # Try to infer from node data
        if "type" in node_data:
            return node_data["type"]
        if "label" in node_data:
            return node_data["label"]
        if "category" in node_data:
            return node_data["category"]
        
        return "Node"  # Default type
    
    def _get_edge_type(
        self,
        source: Any,
        target: Any,
        edge_data: Dict[str, Any],
        edge_labels: Optional[Dict[Tuple[Any, Any], str]]
    ) -> str:
        """Determine edge type from labels or data."""
        if edge_labels:
            if (source, target) in edge_labels:
                return edge_labels[(source, target)]
            if (target, source) in edge_labels:  # Check reverse for undirected
                return edge_labels[(target, source)]
        
        # Try to infer from edge data
        if "type" in edge_data:
            return edge_data["type"]
        if "label" in edge_data:
            return edge_data["label"]
        if "relationship" in edge_data:
            return edge_data["relationship"]
        
        return "CONNECTED_TO"  # Default type