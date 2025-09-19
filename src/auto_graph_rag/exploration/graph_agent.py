"""LLM-based graph exploration agent."""

from typing import Dict, Any, List, Optional
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class NodeTypeSchema(BaseModel):
    """Schema for a node type."""
    name: str = Field(description="Node type name")
    properties: List[str] = Field(description="List of property names")
    description: str = Field(description="Description of what this node represents")
    example_values: Dict[str, Any] = Field(description="Example property values")


class EdgeTypeSchema(BaseModel):
    """Schema for an edge type."""
    name: str = Field(description="Edge type name")
    source: str = Field(description="Source node type")
    target: str = Field(description="Target node type")
    properties: List[str] = Field(description="List of property names")
    description: str = Field(description="Description of the relationship")
    cardinality: str = Field(description="Relationship cardinality (one-to-one, one-to-many, many-to-many)")


class GraphSchema(BaseModel):
    """Complete graph schema."""
    nodes: List[NodeTypeSchema] = Field(description="Node type schemas")
    edges: List[EdgeTypeSchema] = Field(description="Edge type schemas")
    summary: str = Field(description="High-level summary of the graph domain")


class GraphExplorer:
    """Explore graph structure using LLM."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """Initialize graph explorer.
        
        Args:
            provider: LLM provider
            model: Model name
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.llm = ChatOpenAI(model=model, temperature=0)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.parser = PydanticOutputParser(pydantic_object=GraphSchema)
    
    def explore(
        self,
        kuzu_adapter,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Explore graph structure and learn schema using multi-step approach.
        
        Args:
            kuzu_adapter: KuzuAdapter instance
            max_samples: Maximum samples to analyze
            
        Returns:
            Learned graph schema
        """
        print("DEBUG: Starting multi-step schema exploration...")
        
        try:
            # Step 1: Get overview
            schema_overview = self._get_schema_overview(kuzu_adapter)
            print(f"DEBUG: Got schema overview with {len(schema_overview.get('node_tables', []))} node tables, {len(schema_overview.get('edge_tables', []))} edge tables")
            
            # Step 2: Analyze nodes incrementally
            node_schemas = self._analyze_nodes_incrementally(kuzu_adapter, schema_overview, max_samples)
            print(f"DEBUG: Analyzed {len(node_schemas)} node types")
            
            # Step 3: Analyze edges in batches
            edge_schemas = self._analyze_edges_incrementally(kuzu_adapter, schema_overview, node_schemas, max_samples)
            print(f"DEBUG: Analyzed {len(edge_schemas)} edge types")
            
            # Step 4: Generate final schema
            final_schema = self._synthesize_final_schema(schema_overview, node_schemas, edge_schemas)
            print(f"DEBUG: Final schema - nodes: {list(final_schema['nodes'].keys())}")
            print(f"DEBUG: Final schema - edges: {list(final_schema['edges'].keys())}")
            
            return final_schema
            
        except Exception as e:
            print(f"ERROR: Multi-step schema exploration failed: {e}")
            raise RuntimeError(f"Schema exploration failed: {e}") from e
    
    def _sample_data(
        self,
        kuzu_adapter,
        max_samples: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Sample data from each table.
        
        Args:
            kuzu_adapter: KuzuAdapter instance
            max_samples: Maximum samples per table
            
        Returns:
            Sample data by table
        """
        samples = {}
        schema = kuzu_adapter.get_schema_info()
        
        # Sample node tables
        for table in schema["node_tables"]:
            table_name = table["name"]
            # In Kuzu, the node label is the table name itself
            query = f"MATCH (n:{table_name}) RETURN n LIMIT {min(10, max_samples)}"
            
            try:
                results = kuzu_adapter.execute_cypher(query)
                samples[f"node_{table_name}"] = results[:min(10, max_samples)]
            except Exception as e:
                print(f"Failed to sample node table {table_name}: {e}")
                samples[f"node_{table_name}"] = []
        
        # Sample edge tables  
        for table in schema["edge_tables"]:
            table_name = table["name"]
            # In Kuzu, the relationship type is the table name itself
            query = f"MATCH ()-[r:{table_name}]->() RETURN r LIMIT {min(5, max_samples)}"
            
            try:
                results = kuzu_adapter.execute_cypher(query)
                samples[f"edge_{table_name}"] = results[:min(5, max_samples)]
            except Exception as e:
                print(f"Failed to sample edge table {table_name}: {e}")
                samples[f"edge_{table_name}"] = []
        
        return samples
    
    def _get_schema_overview(self, kuzu_adapter) -> Dict[str, Any]:
        """Get basic schema overview without detailed sampling."""
        schema = kuzu_adapter.get_schema_info()
        
        # Count total records for context
        total_nodes = 0
        total_edges = 0
        
        for table in schema.get("node_tables", []):
            try:
                count_query = f"MATCH (n:{table['name']}) RETURN count(n) as count"
                result = kuzu_adapter.execute_cypher(count_query)
                if result:
                    total_nodes += result[0].get('count', 0)
            except:
                pass
        
        for table in schema.get("edge_tables", []):
            try:
                count_query = f"MATCH ()-[r:{table['name']}]->() RETURN count(r) as count"
                result = kuzu_adapter.execute_cypher(count_query)
                if result:
                    total_edges += result[0].get('count', 0)
            except:
                pass
        
        return {
            "node_tables": schema.get("node_tables", []),
            "edge_tables": schema.get("edge_tables", []),
            "total_nodes": total_nodes,
            "total_edges": total_edges
        }
    
    def _analyze_nodes_incrementally(self, kuzu_adapter, schema_overview, max_samples) -> Dict[str, Any]:
        """Analyze node types in small batches."""
        node_schemas = {}
        node_tables = schema_overview.get("node_tables", [])
        
        # Process nodes in batches of 2-3
        batch_size = 2
        for i in range(0, len(node_tables), batch_size):
            batch = node_tables[i:i + batch_size]
            
            # Sample data for this batch
            batch_samples = {}
            for table in batch:
                table_name = table["name"]
                query = f"MATCH (n:{table_name}) RETURN n LIMIT {min(3, max_samples)}"
                try:
                    results = kuzu_adapter.execute_cypher(query)
                    batch_samples[table_name] = results[:3]
                except Exception as e:
                    print(f"Failed to sample {table_name}: {e}")
                    batch_samples[table_name] = []
            
            # Analyze this batch with LLM
            if batch_samples:
                batch_analysis = self._analyze_node_batch(batch, batch_samples)
                node_schemas.update(batch_analysis)
        
        return node_schemas
    
    def _analyze_node_batch(self, tables, samples) -> Dict[str, Any]:
        """Analyze a batch of node tables."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing node types in a graph database. For each node type, determine:
1. What it represents in the real world
2. Key properties and their meanings
3. Example values

Return JSON format: {{"NodeType": {{"description": "...", "properties": [...], "example_values": {{...}}}}}}"""),
            ("human", """Node Tables: {tables}
Sample Data: {samples}

Analyze these node types:""")
        ])
        
        llm = ChatOpenAI(model=self.model, temperature=0)
        
        result = llm.invoke(prompt.format_messages(
            tables=json.dumps([t["name"] for t in tables]),
            samples=json.dumps(samples, default=str)
        ))
        
        try:
            import re
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            raise RuntimeError(f"Failed to parse node batch analysis: {e}")
        
        raise RuntimeError("Could not parse LLM response for node analysis")
    
    def _analyze_edges_incrementally(self, kuzu_adapter, schema_overview, node_schemas, max_samples) -> Dict[str, Any]:
        """Analyze edge types in batches, grouping by semantic similarity."""
        edge_schemas = {}
        edge_tables = schema_overview.get("edge_tables", [])
        
        # Group edges by semantic categories for more efficient analysis
        edge_groups = self._group_edges_semantically([t["name"] for t in edge_tables])
        
        for group_name, edge_names in edge_groups.items():
            print(f"DEBUG: Analyzing {group_name} group with {len(edge_names)} edge types")
            
            # Sample data for this group
            group_samples = {}
            for edge_name in edge_names:
                query = f"MATCH (a)-[r:{edge_name}]->(b) RETURN '{edge_name}' as rel_type, labels(a)[0] as src, labels(b)[0] as tgt LIMIT {min(2, max_samples)}"
                try:
                    results = kuzu_adapter.execute_cypher(query)
                    group_samples[edge_name] = results[:2]
                except Exception as e:
                    print(f"Failed to sample {edge_name}: {e}")
                    group_samples[edge_name] = []
            
            # Analyze this group
            if group_samples:
                group_analysis = self._analyze_edge_group(group_name, edge_names, group_samples, node_schemas)
                edge_schemas.update(group_analysis)
        
        return edge_schemas
    
    def _group_edges_semantically(self, edge_names) -> Dict[str, List[str]]:
        """Group edges by semantic similarity to reduce context size."""
        groups = {
            "diplomatic": [],
            "economic": [], 
            "military": [],
            "cultural": [],
            "political": [],
            "geographic": [],
            "other": []
        }
        
        # Simple keyword-based grouping
        for edge_name in edge_names:
            name_lower = edge_name.lower()
            if any(word in name_lower for word in ["embassy", "diplomatic", "treaty", "accord"]):
                groups["diplomatic"].append(edge_name)
            elif any(word in name_lower for word in ["economic", "trade", "export", "aid", "embargo"]):
                groups["economic"].append(edge_name)
            elif any(word in name_lower for word in ["military", "alliance", "war", "conflict", "attack"]):
                groups["military"].append(edge_name)
            elif any(word in name_lower for word in ["cultural", "student", "book", "tourism", "emigrant"]):
                groups["cultural"].append(edge_name)
            elif any(word in name_lower for word in ["political", "bloc", "vote", "protest", "government"]):
                groups["political"].append(edge_name)
            elif any(word in name_lower for word in ["territory", "border", "geographic", "region"]):
                groups["geographic"].append(edge_name)
            else:
                groups["other"].append(edge_name)
        
        # Remove empty groups and limit group size
        filtered_groups = {}
        for k, v in groups.items():
            if v:
                # Split large groups into smaller batches
                if len(v) > 8:
                    for i in range(0, len(v), 8):
                        batch_name = f"{k}_batch_{i//8 + 1}"
                        filtered_groups[batch_name] = v[i:i+8]
                else:
                    filtered_groups[k] = v
        
        return filtered_groups
    
    def _analyze_edge_group(self, group_name, edge_names, samples, node_schemas) -> Dict[str, Any]:
        """Analyze a semantically grouped set of edges."""
        # Build the system prompt with known node types
        node_types_str = str(list(node_schemas.keys()))
        system_prompt = f"""You are analyzing {group_name} relationships in an international relations graph. 
Known node types: {node_types_str}

For each relationship, determine:
1. What it represents in international relations
2. Source and target node types
3. Cardinality (one-to-many, many-to-many, etc.)

Return JSON with actual relationship names as keys."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Relationship Group: {group_name}
Edge Types: {edge_names}
Sample Data: {samples}

Analyze these relationships:""")
        ])
        
        llm = ChatOpenAI(model=self.model, temperature=0)
        
        result = llm.invoke(prompt.format_messages(
            group_name=group_name,
            edge_names=edge_names,
            samples=json.dumps(samples, default=str)
        ))
        
        try:
            import re
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            raise RuntimeError(f"Failed to parse edge group analysis for {group_name}: {e}")
        
        raise RuntimeError(f"Could not parse LLM response for edge group {group_name}")
    
    def _synthesize_final_schema(self, schema_overview, node_schemas, edge_schemas) -> Dict[str, Any]:
        """Synthesize final comprehensive schema."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are creating a final comprehensive schema summary for an international relations knowledge graph.

Based on the analyzed components, provide a clear 2-3 sentence summary of:
1. The domain and purpose of this graph
2. What types of entities and relationships it contains
3. What kinds of questions it can answer

Be concise and focus on the key insights."""),
            ("human", """Schema Overview: {overview}
Number of node types: {num_nodes}
Number of edge types: {num_edges}

Key node types: {node_types}
Key relationship categories: {edge_categories}

Create a summary of this international relations knowledge graph:""")
        ])
        
        llm = ChatOpenAI(model=self.model, temperature=0)
        
        result = llm.invoke(prompt.format_messages(
            overview=f"Total nodes: {schema_overview.get('total_nodes', 0)}, Total edges: {schema_overview.get('total_edges', 0)}",
            num_nodes=len(node_schemas),
            num_edges=len(edge_schemas),
            node_types=list(node_schemas.keys()),
            edge_categories=list(set(edge_name.split('_')[0] for edge_name in edge_schemas.keys()))
        ))
        
        # Extract summary from result
        summary = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "summary": summary.strip(),
            "nodes": node_schemas,
            "edges": edge_schemas
        }