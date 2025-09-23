I am starting a project to creat a system that fine tunes generative language models on the task of generating cypher queries to answer questions via a knowledge graph. The general pattern I want to follow is this. A user submits a graph - for now lets use networkx as input. The graph is then saved to a kuzu database. Then we use an LLM Agent to interogate the graph learning for its self what the nodes are, what features are on the nodes, what the edges are, what features are on the edges and how nodes and edges relate to each other - building up its understandin of how the graph works - it should save this understanding in its memory. Then I want to use that information to generate thousands of example pairs of how the graph could be queried to answer questions - this is to be training data for a smaller SLM or much smaller LLM to learn to generate queries from questions like: 

Question:
"Who worked on project A"  

Query:
Match(e:Employee)-[:WORKED_ON]-(p:Project) return DISTINCT e

Then I want the SLM or small LLM to be fine tuned on this generated training data. 

Then the SLM or Small LLM can be used for graph rag effectivly. 


This system can scale to work with larger graphs housed in Neo4j or other graph databases and fits well into agentic systems that leverage knowlege graph for structured memory an have need for high frequency recurrent inferance which becomes quickly expensive when leveraging the most popular LLMs. It fits well because SLMs can be served at a fraction of the price. There are also add on benifets for carbon consious firms, and firms participating in ESG initiatives because SLMs require far less energy to run and can be served using less energy than runing a personal computer. 