"""Command-line interface for modular Auto-Graph-RAG."""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Auto-Graph-RAG Modular CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  build       Build graph database from NetworkX or raw data
  explore     Explore graph schema using LLM
  generate    Generate training data from schema
  train       Fine-tune model on training data
  query       Execute queries with fine-tuned model
  pipeline    Run complete or partial pipeline

Examples:
  # Build graph from NetworkX
  auto-graph-rag build --input graph.py --db-path ./mydb
  
  # Explore schema
  auto-graph-rag explore --db-path ./mydb --output schema.json
  
  # Generate training data
  auto-graph-rag generate --schema schema.json --output dataset.jsonl --num-examples 500
  
  # Train model
  auto-graph-rag train --dataset dataset.jsonl --model meta-llama/Llama-3.2-1B-Instruct --output ./my-model
  
  # Query with trained model
  auto-graph-rag query --model ./my-model --db ./mydb --question "Find all employees"
  
  # Run complete pipeline
  auto-graph-rag pipeline --input-graph graph.py --output-model ./final-model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build graph database')
    build_parser.add_argument('--input', required=True, help='Input file (Python or JSON)')
    build_parser.add_argument('--db-path', required=True, help='Output database path')
    build_parser.add_argument('--graph-name', default='default', help='Graph name')
    build_parser.add_argument('--format', choices=['json', 'networkx'], default='json')
    build_parser.add_argument('--verbose', action='store_true')
    
    # Explore command
    explore_parser = subparsers.add_parser('explore', help='Explore graph schema')
    explore_parser.add_argument('--db-path', required=True, help='Database path')
    explore_parser.add_argument('--output', help='Output schema file')
    explore_parser.add_argument('--max-samples', type=int, default=100)
    explore_parser.add_argument('--llm-provider', default='openai')
    explore_parser.add_argument('--llm-model', default='gpt-4')
    explore_parser.add_argument('--verbose', action='store_true')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate training data')
    generate_parser.add_argument('--schema', required=True, help='Schema JSON file')
    generate_parser.add_argument('--output', required=True, help='Output dataset file')
    generate_parser.add_argument('--num-examples', type=int, default=100)
    generate_parser.add_argument('--db-path', help='Database path for validation')
    generate_parser.add_argument('--complexity', help='Complexity distribution JSON')
    generate_parser.add_argument('--llm-provider', default='openai')
    generate_parser.add_argument('--llm-model', default='gpt-4')
    generate_parser.add_argument('--stats', action='store_true')
    generate_parser.add_argument('--verbose', action='store_true')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--dataset', required=True, help='Dataset JSONL file')
    train_parser.add_argument('--model', required=True, help='Base model name')
    train_parser.add_argument('--output', required=True, help='Output model directory')
    train_parser.add_argument('--epochs', type=int, default=3)
    train_parser.add_argument('--learning-rate', type=float, default=2e-5)
    train_parser.add_argument('--batch-size', type=int, default=4)
    train_parser.add_argument('--lora-rank', type=int, default=16)
    train_parser.add_argument('--stats', action='store_true')
    train_parser.add_argument('--verbose', action='store_true')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Execute queries')
    query_parser.add_argument('--model', required=True, help='Model path')
    query_parser.add_argument('--db', required=True, help='Database path')
    query_parser.add_argument('--question', help='Single question')
    query_parser.add_argument('--questions-file', help='File with questions')
    query_parser.add_argument('--output', help='Output results file')
    query_parser.add_argument('--model-name', default='auto-detect')
    query_parser.add_argument('--temperature', type=float, default=0.1)
    query_parser.add_argument('--max-retries', type=int, default=2)
    query_parser.add_argument('--no-cypher', action='store_true')
    query_parser.add_argument('--no-format', action='store_true')
    query_parser.add_argument('--verbose', action='store_true')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run pipeline')
    pipeline_parser.add_argument('--config', help='Configuration file')
    pipeline_parser.add_argument('--save-config', help='Save config template')
    pipeline_parser.add_argument('--input-graph', help='Input NetworkX graph')
    pipeline_parser.add_argument('--input-nodes', help='Input nodes JSON')
    pipeline_parser.add_argument('--input-edges', help='Input edges JSON')
    pipeline_parser.add_argument('--db-path', default='./pipeline_db')
    pipeline_parser.add_argument('--output-model', help='Output model directory')
    pipeline_parser.add_argument('--skip-build', action='store_true')
    pipeline_parser.add_argument('--skip-explore', action='store_true')
    pipeline_parser.add_argument('--skip-generate', action='store_true')
    pipeline_parser.add_argument('--skip-train', action='store_true')
    pipeline_parser.add_argument('--skip-test', action='store_true')
    pipeline_parser.add_argument('--num-examples', type=int, default=100)
    pipeline_parser.add_argument('--epochs', type=int, default=3)
    pipeline_parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Import and run the appropriate module
    try:
        if args.command == 'build':
            from .modules.graph_builder import main as build_main
            sys.argv = ['build'] + [f'--{k.replace("_", "-")}' if len(k) > 1 else f'-{k}' 
                                   for k, v in vars(args).items() 
                                   if v is not None and k != 'command' and v is not False] + \
                      [str(v) for k, v in vars(args).items() 
                       if v is not None and k != 'command' and v is not False and not isinstance(v, bool)]
            return build_main()
            
        elif args.command == 'explore':
            from .modules.graph_explorer import main as explore_main
            sys.argv = ['explore'] + [f'--{k.replace("_", "-")}' if len(k) > 1 else f'-{k}' 
                                     for k, v in vars(args).items() 
                                     if v is not None and k != 'command' and v is not False] + \
                      [str(v) for k, v in vars(args).items() 
                       if v is not None and k != 'command' and v is not False and not isinstance(v, bool)]
            return explore_main()
            
        elif args.command == 'generate':
            from .modules.data_generator import main as generate_main
            sys.argv = ['generate'] + [f'--{k.replace("_", "-")}' if len(k) > 1 else f'-{k}' 
                                      for k, v in vars(args).items() 
                                      if v is not None and k != 'command' and v is not False] + \
                      [str(v) for k, v in vars(args).items() 
                       if v is not None and k != 'command' and v is not False and not isinstance(v, bool)]
            return generate_main()
            
        elif args.command == 'train':
            from .modules.model_trainer import main as train_main
            sys.argv = ['train'] + [f'--{k.replace("_", "-")}' if len(k) > 1 else f'-{k}' 
                                   for k, v in vars(args).items() 
                                   if v is not None and k != 'command' and v is not False] + \
                      [str(v) for k, v in vars(args).items() 
                       if v is not None and k != 'command' and v is not False and not isinstance(v, bool)]
            return train_main()
            
        elif args.command == 'query':
            from .modules.query_executor import main as query_main
            sys.argv = ['query'] + [f'--{k.replace("_", "-")}' if len(k) > 1 else f'-{k}' 
                                   for k, v in vars(args).items() 
                                   if v is not None and k != 'command' and v is not False] + \
                      [str(v) for k, v in vars(args).items() 
                       if v is not None and k != 'command' and v is not False and not isinstance(v, bool)]
            return query_main()
            
        elif args.command == 'pipeline':
            from .modules.pipeline import main as pipeline_main
            sys.argv = ['pipeline'] + [f'--{k.replace("_", "-")}' if len(k) > 1 else f'-{k}' 
                                      for k, v in vars(args).items() 
                                      if v is not None and k != 'command' and v is not False] + \
                      [str(v) for k, v in vars(args).items() 
                       if v is not None and k != 'command' and v is not False and not isinstance(v, bool)]
            return pipeline_main()
            
    except Exception as e:
        print(f"❌ Error executing {args.command}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())