"""Generate natural language questions for training data using LLM."""

from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from ..llm.providers import LLMProviderFactory

logger = logging.getLogger(__name__)


class QuestionCypherPair(BaseModel):
    """A question-cypher training pair."""
    question: str = Field(description="Natural language question")
    cypher: str = Field(description="Corresponding Cypher query")
    complexity: int = Field(description="Query complexity (1-5)")
    intent: str = Field(description="Query intent (lookup, filter, traverse, aggregate, path)")
    

class QuestionBatch(BaseModel):
    """Batch of question-cypher pairs."""
    pairs: List[QuestionCypherPair]


class QuestionGenerator:
    """Generate diverse questions using LLM based on graph schema."""
    
    def __init__(self, llm_provider: str = "auto", llm_model: Optional[str] = None, kuzu_adapter=None):
        """Initialize question generator.
        
        Args:
            llm_provider: LLM provider ("openai", "local", or "auto")
            llm_model: Model to use (optional, auto-selected if None)
            kuzu_adapter: Optional Kuzu adapter for validation
        """
        # Use the unified provider system
        self.provider = LLMProviderFactory.create(
            provider=llm_provider,
            model=llm_model,
            temperature=0.8  # Higher temperature for diversity
        )
        
        self.kuzu_adapter = kuzu_adapter
        
        logger.info(f"Initialized QuestionGenerator with {self.provider.get_model_name()}")
    
    def generate(
        self,
        schema: Dict[str, Any],
        num_examples: int,
        complexity_distribution: Dict[int, float],
        kuzu_adapter=None,
        validate_queries: bool = True,
        include_results: bool = True,
        batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Generate diverse questions based on schema.
        
        Args:
            schema: Graph schema from exploration
            num_examples: Number of examples to generate
            complexity_distribution: Distribution of complexities
            kuzu_adapter: KuzuAdapter instance for query validation
            validate_queries: Whether to validate queries work
            include_results: Whether to include query results in training data
            batch_size: Number of examples to generate per LLM call (default: 20)
            
        Returns:
            List of generated question-cypher pairs with results
        """
        self.kuzu_adapter = kuzu_adapter
        all_pairs = []
        
        # Calculate examples per complexity
        complexity_counts = {
            level: int(num_examples * prob)
            for level, prob in complexity_distribution.items()
        }
        
        # Generate for each complexity level
        for complexity, count in complexity_counts.items():
            print(f"DEBUG: Generating {count} examples for complexity {complexity}")
            remaining = count
            attempts = 0
            max_attempts = count * 3  # Allow retries for failed queries
            
            while remaining > 0 and attempts < max_attempts:
                current_batch = min(batch_size, remaining)
                print(f"DEBUG: Generating batch of {current_batch}, remaining={remaining}, attempts={attempts}")
                
                batch = self._generate_batch(
                    schema, 
                    complexity, 
                    current_batch,
                    validate_queries,
                    include_results
                )
                
                print(f"DEBUG: Generated batch with {len(batch)} examples")
                
                # Filter out invalid queries if validation is enabled
                if validate_queries:
                    valid_batch = [p for p in batch if p.get("is_valid", True)]
                    print(f"DEBUG: {len(valid_batch)} examples passed validation")
                    all_pairs.extend(valid_batch)
                    remaining -= len(valid_batch)
                else:
                    all_pairs.extend(batch)
                    remaining -= len(batch)
                
                attempts += current_batch
                
                if len(batch) == 0:
                    print(f"DEBUG: Empty batch generated, breaking to avoid infinite loop")
                    break
        
        return all_pairs
    
    def _generate_batch(
        self,
        schema: Dict[str, Any],
        complexity: int,
        batch_size: int,
        validate: bool = True,
        include_results: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate a batch of question-cypher pairs.
        
        Args:
            schema: Graph schema
            complexity: Target complexity level
            batch_size: Number of pairs to generate
            validate: Whether to validate queries
            include_results: Whether to include query results
            
        Returns:
            List of question-cypher pairs with validation status
        """
        complexity_descriptions = {
            1: """Simple lookups and basic queries:
               - Finding all nodes of a type
               - Getting specific nodes by ID
               - Basic property retrieval""",
            
            2: """Filtered queries with conditions:
               - Finding nodes with specific property values
               - Using WHERE clauses with comparisons (=, >, <, !=)
               - String matching and CONTAINS operations
               - Combining multiple conditions with AND/OR""",
            
            3: """Relationship traversals:
               - Following edges between nodes
               - Finding connected nodes
               - Multi-hop relationships
               - Optional relationships
               - Bidirectional queries""",
            
            4: """Aggregations and analytics:
               - COUNT, SUM, AVG, MIN, MAX operations
               - GROUP BY clauses
               - DISTINCT operations
               - ORDER BY and LIMIT
               - Statistical queries""",
            
            5: """Complex path queries and advanced patterns:
               - Shortest path algorithms
               - Variable-length paths
               - Pattern matching with multiple relationships
               - Subqueries and WITH clauses
               - Complex filtering on paths
               - Combining multiple patterns"""
        }
        
        # First, get sample data from the actual graph if available
        sample_data = ""
        if self.kuzu_adapter and validate:
            samples = self._get_sample_data(schema)
            sample_data = f"""
Sample data from the actual graph:
{json.dumps(samples, indent=2)}

Use these ACTUAL values in your queries to ensure they return real results."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at generating diverse, realistic question-cypher query pairs for training a model.
            
Your task is to generate {batch_size} DIVERSE question-cypher pairs at complexity level {complexity}.

{complexity_description}

IMPORTANT GUIDELINES:
1. Questions should be natural and varied - use different phrasings, styles, and vocabulary
2. Include colloquial language, abbreviations, and different question formats
3. Mix formal and informal tones
4. Use the actual node types, edge types, and properties from the schema
5. Ensure Cypher queries are syntactically correct and would work with the given schema
6. Make questions realistic - what users would actually ask about this graph
7. Vary the intent: some exploratory, some specific, some analytical
8. Include edge cases and interesting query patterns
9. When sample data is provided, use ACTUAL values from the samples to ensure queries return results

Return the pairs in JSON format with fields: question, cypher, complexity, intent"""),
            
            ("human", """Graph Schema:
{schema}

{sample_data}

Generate {batch_size} diverse question-cypher pairs at complexity level {complexity}.

Make sure to:
- Use actual node types from the schema: {node_types}
- Use actual edge types from the schema: {edge_types}
- Use actual properties from the schema nodes and edges
- Create realistic, natural questions a user would ask
- Ensure all Cypher queries are valid for this schema
- When filtering or searching, use actual values from the sample data when provided

Output as JSON array.""")
        ])
        
        # Extract node and edge types for the prompt
        node_types = list(schema.get("nodes", {}).keys())
        edge_types = list(schema.get("edges", {}).keys())
        
        
        formatted_messages = prompt.format_messages(
            batch_size=batch_size,
            complexity=complexity,
            complexity_description=complexity_descriptions[complexity],
            schema=json.dumps(schema, indent=2),
            sample_data=sample_data,
            node_types=", ".join(node_types) if node_types else "No node types defined",
            edge_types=", ".join(edge_types) if edge_types else "No edge types defined"
        )
        
        
        # Convert messages to dict format for provider
        messages = []
        for msg in formatted_messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain message format
                role = "system" if msg.type == "system" else "user"
                messages.append({"role": role, "content": msg.content})
            else:
                # Already in dict format
                messages.append(msg)
        
        # Use the provider's structured chat for JSON output
        try:
            pairs_data = self.provider.chat_structured(messages)
            
            # Ensure we have a list
            if isinstance(pairs_data, dict):
                pairs_data = pairs_data.get("pairs", [pairs_data])
            
            content = json.dumps(pairs_data)  # For fallback parsing
            
            # Try to find JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                pairs_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire content
                pairs_data = json.loads(content)
            
            # Convert to our format and validate if requested
            pairs = []
            for item in pairs_data:
                pair = {
                    "question": item.get("question", ""),
                    "cypher": item.get("cypher", ""),
                    "complexity": item.get("complexity", complexity),
                    "intent": item.get("intent", "unknown"),
                    "patterns": self._extract_patterns(item.get("cypher", ""))
                }
                
                # Validate and get results if requested
                if validate and self.kuzu_adapter:
                    is_valid, results, error = self._validate_query(
                        pair["cypher"]
                    )
                    pair["is_valid"] = is_valid
                    if is_valid and include_results:
                        pair["results"] = results
                        pair["result_count"] = len(results) if results else 0
                    if error:
                        pair["validation_error"] = error
                
                pairs.append(pair)
            
            return pairs
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            return []
    
    def _get_sample_data(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get sample data from the actual graph.
        
        Args:
            schema: Graph schema
            
        Returns:
            Sample data dictionary
        """
        samples = {}
        
        # Sample nodes for each type
        for node_type in schema.get("nodes", {}).keys():
            try:
                query = f"MATCH (n:{node_type}) RETURN n LIMIT 5"
                results = self.kuzu_adapter.execute_cypher(query)
                if results:
                    samples[f"sample_{node_type}"] = results
            except:
                pass
        
        # Sample relationships
        for edge_type, edge_info in schema.get("edges", {}).items():
            try:
                source = edge_info.get("source", "Node")
                target = edge_info.get("target", "Node")
                query = f"""
                MATCH (s:{source})-[r:{edge_type}]->(t:{target})
                RETURN s, r, t LIMIT 3
                """
                results = self.kuzu_adapter.execute_cypher(query)
                if results:
                    samples[f"sample_{edge_type}_relationships"] = results
            except:
                pass
        
        return samples
    
    def _validate_query(self, cypher: str) -> Tuple[bool, Optional[List], Optional[str]]:
        """Validate a Cypher query by executing it.
        
        Args:
            cypher: Cypher query to validate
            
        Returns:
            Tuple of (is_valid, results, error_message)
        """
        if not self.kuzu_adapter:
            return True, None, None  # Can't validate without adapter
        
        try:
            # Execute the query
            results = self.kuzu_adapter.execute_cypher(cypher)
            
            # Query is valid if it executes without error
            print(f"DEBUG: Valid query: {cypher[:100]}...")
            return True, results, None
            
        except Exception as e:
            # Query failed - return error details
            error_msg = str(e)
            print(f"DEBUG: Invalid query: {cypher[:100]}...")
            print(f"DEBUG: Error: git{error_msg}")
            return False, None, error_msg
    
    def _extract_patterns(self, cypher: str) -> List[str]:
        """Extract query patterns from Cypher query.
        
        Args:
            cypher: Cypher query string
            
        Returns:
            List of identified patterns
        """
        patterns = []
        cypher_upper = cypher.upper()
        
        # Check for various Cypher patterns
        if "MATCH" in cypher_upper:
            patterns.append("match")
        if "WHERE" in cypher_upper:
            patterns.append("filter")
        if "COUNT" in cypher_upper or "SUM" in cypher_upper or "AVG" in cypher_upper:
            patterns.append("aggregation")
        if "ORDER BY" in cypher_upper:
            patterns.append("sort")
        if "LIMIT" in cypher_upper:
            patterns.append("limit")
        if "SHORTEST" in cypher_upper and "PATH" in cypher_upper:
            patterns.append("shortest_path")
        if "*" in cypher and ".." in cypher:
            patterns.append("variable_length_path")
        if "GROUP BY" in cypher_upper:
            patterns.append("group_by")
        if "DISTINCT" in cypher_upper:
            patterns.append("distinct")
        if "OPTIONAL" in cypher_upper:
            patterns.append("optional")
        if "WITH" in cypher_upper:
            patterns.append("with_clause")
        if "UNION" in cypher_upper:
            patterns.append("union")
        if "CREATE" in cypher_upper or "MERGE" in cypher_upper:
            patterns.append("write")
        if "DELETE" in cypher_upper:
            patterns.append("delete")
        if "-[" in cypher or "]-" in cypher:
            patterns.append("relationship")
            
        return patterns if patterns else ["simple"]
    
    def _group_edges_semantically(self, edge_names: List[str]) -> Dict[str, List[str]]:
        """Group edges into batches to reduce context size.
        
        Uses simple batching strategy to work with any domain.
        """
        batch_size = 10  # Optimal batch size for question generation
        filtered_groups = {}
        
        # Sort edge names for consistent processing
        sorted_edges = sorted(edge_names)
        
        # Create batches
        for i in range(0, len(sorted_edges), batch_size):
            batch = sorted_edges[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(sorted_edges) + batch_size - 1) // batch_size
            
            # Use descriptive batch names
            if total_batches == 1:
                batch_name = "relationships"
            else:
                batch_name = f"relationships_batch_{batch_num}_of_{total_batches}"
            
            filtered_groups[batch_name] = batch
        
        return filtered_groups
    
    def _create_batch_schema(self, schema: Dict[str, Any], edge_group: List[str]) -> Dict[str, Any]:
        """Create a smaller schema focused on specific edge types."""
        batch_schema = {
            "summary": schema.get("summary", ""),
            "nodes": schema.get("nodes", {}),
            "edges": {}
        }
        
        # Only include edges from this group
        all_edges = schema.get("edges", {})
        for edge_name in edge_group:
            if edge_name in all_edges:
                batch_schema["edges"][edge_name] = all_edges[edge_name]
        
        return batch_schema
    
    def generate_batched(
        self,
        schema: Dict[str, Any],
        num_examples: int,
        complexity_distribution: Dict[int, float],
        kuzu_adapter=None,
        validate_queries: bool = True,
        include_results: bool = True,
        batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Generate questions using batched approach for large schemas."""
        print("DEBUG: Using batched question generation approach")

        ## TODO this is the crucial method to improve
        
        # Group edge types semantically
        edge_names = list(schema.get("edges", {}).keys())
        edge_groups = self._group_edges_semantically(edge_names)
        
        print(f"DEBUG: Grouped {len(edge_names)} edge types into {len(edge_groups)} semantic groups")
        for group_name, edges in edge_groups.items():
            print(f"  - {group_name}: {len(edges)} edge types")
        
        all_questions = []
        
        # Calculate examples per group (weighted by group size)
        total_edges = len(edge_names)
        
        for group_name, group_edges in edge_groups.items():
            # Calculate examples for this group based on its proportion
            group_weight = len(group_edges) / total_edges if total_edges > 0 else 1.0 / len(edge_groups)
            group_examples = max(5, int(num_examples * group_weight))  # At least 5 examples per group
            
            print(f"DEBUG: Generating {group_examples} examples for {group_name} group")
            
            # Create schema subset for this group
            batch_schema = self._create_batch_schema(schema, group_edges)
            
            # Generate questions for this batch
            try:
                batch_questions = self.generate(
                    schema=batch_schema,
                    num_examples=group_examples,
                    complexity_distribution=complexity_distribution,
                    kuzu_adapter=kuzu_adapter,
                    validate_queries=validate_queries,
                    include_results=include_results,
                    batch_size=batch_size
                )
                
                all_questions.extend(batch_questions)
                print(f"DEBUG: Successfully generated {len(batch_questions)} questions for {group_name}")
                
            except Exception as e:
                print(f"WARNING: Failed to generate questions for {group_name}: {e}")
                continue
        
        # Shuffle and limit to requested number
        import random
        random.shuffle(all_questions)
        
        final_questions = all_questions[:num_examples]
        print(f"DEBUG: Batched generation complete: {len(final_questions)} total questions")
        
        return final_questions