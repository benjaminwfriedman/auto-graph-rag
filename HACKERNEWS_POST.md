# Auto-Graph-RAG: Teaching Small LLMs to Query Knowledge Graphs Like Their Big Brothers

Graph RAG is becoming a standard operating procedure for information retrieval in chat and agentic systems. When compared to vector-based retrieval, Graph RAG enables more comprehensive knowledge access with reduced hallucination risk. However, Graph RAG typically requires significantly more upfront engineering to craft prompts that provide sufficient schema context and query examples, enabling models to generate syntactically correct queries that align with the knowledge graph's structure and query language.
Additionally, since an LLM pass must be computed to generate a query to retrieve information from the graph, Graph RAG is computationally expensive. On large knowledge graphs, context can quickly approach the 8K+ token limit of common LLMs and cost over a cent per question. Agentic systems often need to make tens to hundreds of retrievals during a session, making cost management a critical consideration.

# Our Approach

Our core insight is simple but valuable: we decompose Graph RAG into two distinct phases—contextual learning and inference—and optimize each with purpose-built models.

**Contextual learning** focuses on understanding the graph's syntax, topology, and attributes. Traditional Graph RAG relies on human engineers to manually explore graph structures and craft rich contextual prompts—a time-intensive process. We automate this through an agentic exploration system that strategically samples the knowledge graph to discover its schema, relationships, and query patterns.

**Inference** focuses on translating natural language inputs into valid graph queries. Rather than burning tokens on schema descriptions for every query, we distill the learned graph knowledge into a fine-tuned small language model (Llama-3.2-1B) using automatically generated question-Cypher pairs from the exploration phase.

**The result:** a specialized 1B parameter model that performs graph query generation without requiring extensive context, dramatically reducing both latency and cost while maintaining query accuracy. For inference, we need only the lightweight SLM rather than expensive API calls to frontier models.