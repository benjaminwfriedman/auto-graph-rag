# Auto-Graph-RAG: Teaching Small LLMs to Query Knowledge Graphs Like Their Big Brothers

Graph RAG is becoming a standard operating procedure for information retrieval in chat and agentic systems. When compared to vector-based retrieval, Graph RAG enables more comprehensive knowledge access with reduced hallucination risk. However, Graph RAG typically requires significantly more upfront engineering to craft prompts that provide sufficient schema context and query examples, enabling models to generate syntactically correct queries that align with the knowledge graph's structure and query language.

Additionally, since an LLM pass must be computed to generate a query to retrieve information from the graph, Graph RAG is computationally expensive. On large knowledge graphs, context can quickly approach the 8K+ token limit of common LLMs and cost over a cent per question. Agentic systems often need to make tens to hundreds of retrievals during a session, making cost management a critical consideration.

## The Problem That Started This

I've been working on a project involving scene graph queries—essentially generating hundreds to thousands of graph queries per session to understand spatial and object relationships in visual scenes. The API costs were getting painful, but more frustrating was the repetition: my scene graph structure wasn't changing much between queries, yet I was burning expensive context tokens every single time to "teach" GPT-4 about the same node types, relationship patterns, and query syntax.

It felt wasteful. Why was I repeatedly paying to explain that scenes contain objects, objects have spatial relationships, and here's how to write a Cypher query to find "all chairs near tables" when the underlying graph schema was fundamentally stable?

This led to a simple insight: what if I could teach a small model about my specific graph structure once, then use that specialized model for all subsequent queries without any context overhead?

## Our Approach

Our core insight is simple but valuable: we decompose Graph RAG into two distinct phases—contextual learning and inference—and optimize each with purpose-built models. Crucially, we've designed this to be easily applicable to any knowledge graph with minimal manual intervention.

**Contextual learning** focuses on understanding the graph's syntax, topology, and attributes. Traditional Graph RAG relies on human engineers to manually explore graph structures and craft rich contextual prompts—a time-intensive process. We automate this through an agentic exploration system that strategically samples the knowledge graph to discover its schema, relationships, and query patterns. The system automatically adapts to different graph types and structures without requiring domain-specific configuration.

**Inference** focuses on translating natural language inputs into valid graph queries. Rather than burning tokens on schema descriptions for every query, we distill the learned graph knowledge into a fine-tuned small language model (Llama-3.2-1B) using automatically generated question-Cypher pairs from the exploration phase.

**The result:** a specialized 1B parameter model that performs graph query generation without requiring extensive context, dramatically reducing cost while maintaining query accuracy. For inference, we need only the lightweight SLM rather than expensive API calls to frontier models.

**Easy deployment:** The entire process—from graph exploration to model fine-tuning—is designed to work out-of-the-box with any Neo4j-compatible knowledge graph. Point it at your database, run the automated exploration, and get a specialized model trained on your specific graph structure.

## Initial Results

I tested Auto-Graph-RAG on a nations dataset containing international relations, treaties, and diplomatic relationships. The automated exploration phase generated question-Cypher training pairs, and I compared a fine-tuned Llama-3.2-1B (served locally on CPU) against GPT-4 on six test queries.

**Performance comparison:**

| Metric | OpenAI GPT-4 | Auto-Graph-RAG |
|--------|--------------|----------------|
| **Accuracy** | 100% success | 100% success |
| **Average Latency** | 2.16 seconds | 6.47 seconds |
| **Token Usage** | ~295 prompt + 39 completion | 0 tokens |
| **Cost per Query** | ~0.0003 | $0 |
| **Infrastructure** | API dependency | Local CPU only |
| **Setup Time** | Manual prompt engineering | Automated exploration |

Interestingly, the OpenAI API calls were faster in this test, likely due to their optimized inference infrastructure and the fact that our model was running on CPU. However, the cost difference is dramatic: in projects like mine where we run thousands of queries per session, the cost climbs to nearly a dollar per session with OpenAI. The 3x latency penalty becomes easily acceptable to eliminate these recurring costs and 
remove dependency on external services entirely.

## A Simple Interface: And Getting Simpler

The entire Auto-Graph-RAG pipeline is designed to be straightforward to use. Here's what implementing a complete Graph RAG system looks like with our approach, using real code from the Nations dataset demo:

```python
# Step 1: Convert your graph data to a Kuzu database
from auto_graph_rag.modules import GraphBuilder

builder = GraphBuilder()
stats = builder.build_from_networkx(
    graph=nations_graph,
    db_path="nations_db",
    graph_name="nations",
    node_labels=node_labels,
    edge_labels=edge_labels
)
# That's it - your graph is now queryable with Cypher

# Step 2: Automatically explore and understand your graph structure
from auto_graph_rag.modules import GraphExplorer

explorer = GraphExplorer(llm_provider="openai", llm_model="gpt-4")
schema = explorer.explore_from_db(
    db_path="nations_db",
    max_samples=20,
    save_schema_to="nations_schema.json"
)
# The system now understands your graph's semantics

# Step 3: Generate training data automatically
from auto_graph_rag.modules import DataGenerator

generator = DataGenerator(llm_provider="openai", llm_model="gpt-4")
dataset = generator.generate_from_schema(
    schema_path="nations_schema.json",
    num_examples=150,
    output_path="nations_dataset.jsonl",
    complexity_distribution={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
)
# 150 diverse question-Cypher pairs created without manual effort

# Step 4: Fine-tune a small model on your specific graph
from auto_graph_rag.modules import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_from_file(
    dataset_path="nations_dataset.jsonl",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    output_dir="nations_model",
    epochs=15
)
# Your specialized 1B parameter model is ready

# Step 5: Query your graph - zero context needed!
from auto_graph_rag.modules import QueryExecutor

executor = QueryExecutor()
result = executor.query_with_model(
    question="Which countries have embassy relationships?",
    model_path="nations_model",
    db_path="nations_db"
)
# Returns: {'success': True, 'results': [...], 'cypher': 'MATCH ...'}
```

Five module calls take you from raw graph data to a production-ready query system. No prompt engineering, no manual schema documentation, no context window management.

### Key Simplifications

1. **Automated Schema Discovery:** The system discovers your graph structure, relationships, and patterns without manual documentation.

2. **No Prompt Templates**: The exploration phase learns how to query your specific graph structure automatically.

3. **Zero Context Tokens:** Once trained, queries need zero context tokens. Just pass the question and get results.

5. **Modular by Design**: Use only what you need. Already have training data? Skip exploration. Want to use a different model? Just swap the trainer configuration.

The same interface is designed to work whether you have a 14-node demo graph or an enterprise knowledge graph with millions of nodes. The complexity is handled internally while keeping the API clean and costs minimal.

## The Context Window Problem Gets Worse

From my experience building Graph RAG systems, I know the context management challenge becomes significantly more complex as graphs grow. Real production knowledge graphs often require extensive schema descriptions, relationship examples, and query patterns to generate reliable Cypher queries.

In my scene graph work, I was constantly hitting context limits trying to describe all the possible object types, spatial relationships, and query patterns. Each session required careful context management to avoid token limits while ensuring query quality.

Consider a more realistic scenario with larger graphs requiring 4,000 input tokens and 100 output tokens per query:
- **Cost per query**: Jumps to ~$0.008 (nearly 16x increase)
- **Session cost at 1K queries**: $8.00 vs $0.87 for smaller contexts  
- **Annual projection (1 Session per Day)**: ~$3,000 vs $200 for simple graphs

For my high-volume use case, this would have been financially unsustainable while providing no additional value—I was paying premium prices to repeatedly teach the same graph structure.

## Why This Approach Makes Sense

The fundamental issue with current Graph RAG is that we're repeatedly paying to "teach" the model the same graph structure on every query. It's like hiring an expert consultant who forgets everything between meetings.

This became especially clear in my scene graph work: the core relationships (objects have positions, spatial relationships exist between objects, scenes have hierarchical structure) never changed, but I was paying to explain them thousands of times per session.

Auto-Graph-RAG inverts this: pay once during training to encode the graph knowledge into model weights, then inference becomes essentially free. The approach should scale particularly well for:

- **High-query-volume applications** like my scene graph project where costs compound rapidly
- **Complex enterprise graphs** with rich schemas that bloat context windows  
- **Agentic systems** making many sequential graph queries
- **Cost-sensitive deployments** where 3x latency is acceptable for zero marginal cost
- **Privacy-sensitive deployments** where local inference is required
- **Diverse graph domains** where manual schema engineering is prohibitive

The automated approach means teams can apply this to their existing knowledge graphs without becoming experts in prompt engineering or graph schema analysis. While local CPU inference is slower than optimized cloud APIs, the cost savings become compelling at scale.

## Limitations & Open Questions

This is early-stage work with several important limitations:

- **Single graph specialization**: The current approach trains on one graph at a time
- **Latency trade-off**: Local CPU inference is ~3x slower than OpenAI APIs
- **Schema evolution**: Unclear how well the model adapts as graph structures change (though my scene graphs were quite stable)
- **Query complexity limits**: Haven't tested on highly complex multi-hop queries
- **Graph size scaling**: Need to validate the automated exploration on very large graphs
- **Training time**: The automated exploration and fine-tuning process adds upfront overhead

The core hypothesis—that separating contextual learning from inference can dramatically improve Graph RAG economics—seems sound based on these initial results and my painful experience with scene graph API costs, but more extensive validation across different graph types and sizes is needed.

## Next Steps

I'm particularly interested in:
- Testing on larger, more complex graphs with diverse schema patterns
- Exploring multi-graph training approaches for broader applicability
- Optimizing local inference performance and exploring GPU acceleration
- Investigating hybrid approaches that combine fine-tuned query generation with other retrieval methods
- Measuring the exploration and training overhead against long-term savings
- Validating the automated approach across different graph domains (social networks, supply chains, etc.)

The potential for eliminating both the context window scaling problem and API costs in Graph RAG feels significant, especially when the solution can be easily applied to any knowledge graph. While there's a latency trade-off, the cost savings and operational independence make this compelling for many use cases, and definitely would have saved me from the API cost spiral that motivated this work.

**Code and initial results available at**: [https://github.com/benjaminwfriedman/auto-graph-rag]

*Has anyone else experimented with fine-tuning approaches for structured query generation? I'm curious about experiences balancing inference latency vs. API costs, especially from others who've hit similar scaling walls with high-volume graph queries.*