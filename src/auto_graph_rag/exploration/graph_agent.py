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
        """Explore graph structure and learn schema.
        
        Args:
            kuzu_adapter: KuzuAdapter instance
            max_samples: Maximum samples to analyze
            
        Returns:
            Learned graph schema
        """
        # Get basic schema info from database
        db_schema = kuzu_adapter.get_schema_info()
        print(f"DEBUG: DB schema has {len(db_schema.get('node_tables', []))} node tables, {len(db_schema.get('edge_tables', []))} edge tables")
        
        # Sample data from each table
        samples = self._sample_data(kuzu_adapter, max_samples)
        print(f"DEBUG: Sampled {len(samples)} table groups")
        
        # Check if we have any meaningful data
        total_sample_items = sum(len(sample_list) for sample_list in samples.values())
        print(f"DEBUG: Total sample items: {total_sample_items}")
        
        if total_sample_items == 0:
            print("DEBUG: No sample data found - this may cause empty schema")
        
        if not db_schema.get('node_tables') and not db_schema.get('edge_tables'):
            print("DEBUG: No schema info found - this may cause empty schema")
        
        # Use LLM to analyze and understand the graph
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a graph database expert. Analyze the provided graph schema and data samples to understand:
1. What each node type represents
2. What properties are important for each node type
3. What each edge type represents and its cardinality
4. The overall domain of the graph

{format_instructions}"""),
            ("human", """Database Schema:
{db_schema}

Sample Data:
{samples}

Please analyze this graph and provide a comprehensive schema understanding.""")
        ])
        
        chain = prompt | self.llm | self.parser
        
        result = chain.invoke({
            "db_schema": json.dumps(db_schema, indent=2),
            "samples": json.dumps(samples, indent=2),
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Convert to dictionary format  
        schema_dict = {
            "nodes": {},
            "edges": {},
            "summary": result.summary
        }
        
        for node in result.nodes:
            schema_dict["nodes"][node.name] = {
                "properties": node.properties,
                "description": node.description,
                "examples": node.example_values
            }
        
        for edge in result.edges:
            schema_dict["edges"][edge.name] = {
                "source": edge.source,
                "target": edge.target,
                "properties": edge.properties,
                "description": edge.description,
                "cardinality": edge.cardinality
            }
        
        # DEBUG: Print what we're returning
        print(f"DEBUG: Schema conversion - nodes: {list(schema_dict['nodes'].keys())}")
        print(f"DEBUG: Schema conversion - edges: {list(schema_dict['edges'].keys())}")
        
        return schema_dict
    
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