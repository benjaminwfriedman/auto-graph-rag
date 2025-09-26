"""LLM-based graph exploration agent."""

from typing import Dict, Any, List
import json
import logging
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from ..llm.providers import LLMProviderFactory

logger = logging.getLogger(__name__)


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
    
    def __init__(self, provider: str = "auto", model: str = None):
        """Initialize graph explorer.
        
        Args:
            provider: LLM provider ("openai", "local", or "auto" for automatic selection)
            model: Model name (auto-selected if None)
        """
        # Use the unified provider system
        self.llm_provider = LLMProviderFactory.create(
            provider=provider,
            model=model,
            temperature=0.1  # Lower temperature for more consistent schema analysis
        )
        
        logger.info(f"Initialized GraphExplorer with {self.llm_provider.get_model_name()}")
        
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
1. What it represents in the domain
2. Key properties and their meanings
3. Example values from the data

Return JSON format: {{"NodeType": {{"description": "...", "properties": [...], "example_values": {{...}}}}}}"""),
            ("human", """Node Tables: {tables}
Sample Data: {samples}

Analyze these node types:""")
        ])
        
        # Convert to dict format for provider
        messages = []
        for msg in prompt.format_messages(
            tables=json.dumps([t["name"] for t in tables]),
            samples=json.dumps(samples, default=str)
        ):
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "system" if msg.type == "system" else "user"
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append(msg)
        
        # Use provider's structured chat
        result = self.llm_provider.chat_structured(messages)
        
        try:
            # Result should already be parsed JSON from structured chat
            if isinstance(result, dict):
                return result
            elif isinstance(result, list) and result:
                return result[0] if isinstance(result[0], dict) else {}
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return {}
        except Exception as e:
            logger.error(f"Failed to process node batch analysis: {e}")
            return {}
    
    def _analyze_edges_incrementally(self, kuzu_adapter, schema_overview, node_schemas, max_samples) -> Dict[str, Any]:
        """Analyze edge types in batches."""
        edge_schemas = {}
        edge_tables = schema_overview.get("edge_tables", [])
        
        # Process edges in batches
        batch_size = 8  # Optimal batch size for context management
        for i in range(0, len(edge_tables), batch_size):
            batch = edge_tables[i:i + batch_size]
            
            # Sample data for this batch
            batch_samples = {}
            for table in batch:
                edge_name = table["name"]
                # Get actual relationship instances with node names
                query = f"MATCH (a)-[r:{edge_name}]->(b) RETURN a.name as source_name, '{edge_name}' as rel_type, b.name as target_name, r as rel_data LIMIT {min(3, max_samples)}"
                try:
                    results = kuzu_adapter.execute_cypher(query)
                    if results:
                        # Format examples for readability
                        examples = []
                        for result in results:
                            source = result.get('source_name', 'Unknown')
                            target = result.get('target_name', 'Unknown')
                            rel_data = result.get('rel_data', {})
                            # Extract any properties from the relationship data
                            if rel_data and isinstance(rel_data, dict) and len(rel_data) > 0:
                                # Filter out internal fields if any
                                prop_items = {k: v for k, v in rel_data.items() if not k.startswith('_')}
                                if prop_items:
                                    prop_str = f" {prop_items}"
                                    examples.append(f"{source} --[{edge_name}{prop_str}]--> {target}")
                                else:
                                    examples.append(f"{source} --[{edge_name}]--> {target}")
                            else:
                                examples.append(f"{source} --[{edge_name}]--> {target}")
                        batch_samples[edge_name] = {
                            "type": edge_name,
                            "count": len(results),
                            "examples": examples,
                            "sample_relationships": results[:3]  # Include raw data for LLM analysis
                        }
                    else:
                        batch_samples[edge_name] = {"type": edge_name, "count": 0, "examples": []}
                except Exception as e:
                    print(f"Failed to sample {edge_name}: {e}")
                    batch_samples[edge_name] = {"type": edge_name, "count": 0, "examples": [], "error": str(e)}
            
            # Analyze this batch with LLM
            if batch_samples:
                batch_analysis = self._analyze_edge_batch(batch, batch_samples, node_schemas)
                edge_schemas.update(batch_analysis)
        
        return edge_schemas
    
    def _analyze_edge_batch(self, tables, samples, node_schemas) -> Dict[str, Any]:
        """Analyze a batch of edge types."""
        # Build the system prompt with known node types
        node_types_str = str(list(node_schemas.keys()))
        system_prompt = f"""You are analyzing relationship types in a graph database. 
Known node types: {node_types_str}

For each relationship, determine:
1. What it represents in the domain - use context of the domain + examples to elaborate
2. Source and target node types
3. Cardinality (one-to-many, many-to-many, etc.)
4. Properties (if any)
5. Concrete examples from the sample data provided

IMPORTANT: 
- Always include the "examples" field with actual relationship instances from the sample data
- Return ONLY valid JSON format - no text before or after
- Use double quotes for all strings
- Include all required fields for each relationship

Return a JSON object where each key is a relationship name and the value contains: description, source, target, cardinality, properties (array), examples (array of strings)."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Edge Types: {edge_names}
Sample Data: {samples}

Analyze these relationships:""")
        ])
        
        # Extract edge names from the tables
        edge_names = [t["name"] for t in tables]
        
        # Convert to dict format for provider
        messages = []
        for msg in prompt.format_messages(
            edge_names=json.dumps(edge_names),
            samples=json.dumps(samples, default=str)
        ):
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "system" if msg.type == "system" else "user"
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append(msg)
        
        # Use provider's structured chat
        result = self.llm_provider.chat_structured(messages)
        
        try:
            # Result should already be parsed JSON from structured chat
            if isinstance(result, dict):
                logger.debug(f"Edge batch analysis returned {len(result)} edge types")
                return result
            elif isinstance(result, list) and result:
                return result[0] if isinstance(result[0], dict) else {}
            else:
                logger.warning(f"Unexpected edge result type: {type(result)}")
                return {}
        except Exception as e:
            logger.error(f"Failed to process edge batch analysis: {e}")
            return {}
    
    def _synthesize_final_schema(self, schema_overview, node_schemas, edge_schemas) -> Dict[str, Any]:
        """Synthesize final comprehensive schema."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are creating a final comprehensive schema summary for a knowledge graph.

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

Create a summary of this knowledge graph:""")
        ])
        
        # Convert to dict format for provider
        messages = []
        for msg in prompt.format_messages(
            overview=f"Total nodes: {schema_overview.get('total_nodes', 0)}, Total edges: {schema_overview.get('total_edges', 0)}",
            num_nodes=len(node_schemas),
            num_edges=len(edge_schemas),
            node_types=list(node_schemas.keys()),
            edge_categories=list(set(edge_name.split('_')[0] for edge_name in edge_schemas.keys()))
        ):
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "system" if msg.type == "system" else "user"
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append(msg)
        
        # Use regular chat for summary (not structured)
        summary = self.llm_provider.chat(messages)
        
        return {
            "summary": summary.strip(),
            "nodes": node_schemas,
            "edges": edge_schemas
        }