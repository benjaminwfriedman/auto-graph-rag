"""Format query results for presentation."""

from typing import Dict, Any, List, Optional
import json
import pandas as pd
from rich.table import Table
from rich.console import Console


class ResultFormatter:
    """Format query results in various formats."""
    
    def __init__(self):
        """Initialize result formatter."""
        self.console = Console()
    
    def format_results(
        self,
        results: List[Dict[str, Any]],
        question: Optional[str] = None,
        cypher: Optional[str] = None,
        format_type: str = "table"
    ) -> str:
        """Format query results.
        
        Args:
            results: Query results
            question: Original question
            cypher: Generated Cypher query
            format_type: Output format (table, json, markdown, text)
            
        Returns:
            Formatted results string
        """
        if format_type == "table":
            return self._format_as_table(results, question, cypher)
        elif format_type == "json":
            return self._format_as_json(results, question, cypher)
        elif format_type == "markdown":
            return self._format_as_markdown(results, question, cypher)
        else:  # text
            return self._format_as_text(results, question, cypher)
    
    def _format_as_table(
        self,
        results: List[Dict[str, Any]],
        question: Optional[str],
        cypher: Optional[str]
    ) -> str:
        """Format as rich table.
        
        Args:
            results: Query results
            question: Original question
            cypher: Generated query
            
        Returns:
            Table string
        """
        if not results:
            return "No results found."
        
        # Create table
        table = Table(title=question or "Query Results")
        
        # Add columns
        if results:
            for key in results[0].keys():
                table.add_column(str(key))
            
            # Add rows
            for row in results:
                table.add_row(*[str(v) for v in row.values()])
        
        # Convert to string
        from io import StringIO
        string_io = StringIO()
        console = Console(file=string_io, force_terminal=False)
        console.print(table)
        
        output = string_io.getvalue()
        
        if cypher:
            output = f"Query: {cypher}\n\n{output}"
        
        return output
    
    def _format_as_json(
        self,
        results: List[Dict[str, Any]],
        question: Optional[str],
        cypher: Optional[str]
    ) -> str:
        """Format as JSON.
        
        Args:
            results: Query results
            question: Original question
            cypher: Generated query
            
        Returns:
            JSON string
        """
        output = {
            "question": question,
            "cypher": cypher,
            "results": results,
            "count": len(results)
        }
        
        return json.dumps(output, indent=2, default=str)
    
    def _format_as_markdown(
        self,
        results: List[Dict[str, Any]],
        question: Optional[str],
        cypher: Optional[str]
    ) -> str:
        """Format as Markdown table.
        
        Args:
            results: Query results
            question: Original question
            cypher: Generated query
            
        Returns:
            Markdown string
        """
        lines = []
        
        if question:
            lines.append(f"## {question}\n")
        
        if cypher:
            lines.append(f"**Query:** `{cypher}`\n")
        
        if not results:
            lines.append("*No results found.*")
        else:
            # Convert to DataFrame for easy markdown conversion
            df = pd.DataFrame(results)
            lines.append(df.to_markdown(index=False))
            lines.append(f"\n*{len(results)} results*")
        
        return "\n".join(lines)
    
    def _format_as_text(
        self,
        results: List[Dict[str, Any]],
        question: Optional[str],
        cypher: Optional[str]
    ) -> str:
        """Format as plain text.
        
        Args:
            results: Query results
            question: Original question
            cypher: Generated query
            
        Returns:
            Plain text string
        """
        lines = []
        
        if question:
            lines.append(f"Question: {question}")
            lines.append("-" * 50)
        
        if cypher:
            lines.append(f"Query: {cypher}")
            lines.append("-" * 50)
        
        if not results:
            lines.append("No results found.")
        else:
            for i, result in enumerate(results, 1):
                lines.append(f"Result {i}:")
                for key, value in result.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")
            
            lines.append(f"Total: {len(results)} results")
        
        return "\n".join(lines)
    
    def summarize_results(
        self,
        results: List[Dict[str, Any]],
        question: str,
        max_items: int = 5
    ) -> str:
        """Create a natural language summary of results.
        
        Args:
            results: Query results
            question: Original question
            max_items: Maximum items to include in summary
            
        Returns:
            Natural language summary
        """
        if not results:
            return f"I couldn't find any results for: {question}"
        
        count = len(results)
        
        # Build summary based on question type
        question_lower = question.lower()
        
        if "count" in question_lower or "how many" in question_lower:
            return f"The count is {count}."
        
        if "average" in question_lower or "avg" in question_lower:
            # Try to find numeric column
            for key in results[0].keys():
                if isinstance(results[0][key], (int, float)):
                    values = [r[key] for r in results]
                    avg = sum(values) / len(values)
                    return f"The average {key} is {avg:.2f}."
        
        # List results
        if count == 1:
            items = ", ".join(f"{k}: {v}" for k, v in results[0].items())
            return f"Found 1 result: {items}"
        else:
            summary = f"Found {count} results. "
            
            if count <= max_items:
                # Show all
                items = []
                for r in results:
                    # Get first non-null value as display
                    display = next((str(v) for v in r.values() if v), "Result")
                    items.append(display)
                summary += "They are: " + ", ".join(items)
            else:
                # Show sample
                items = []
                for r in results[:max_items]:
                    display = next((str(v) for v in r.values() if v), "Result")
                    items.append(display)
                summary += f"The first {max_items} are: " + ", ".join(items)
                summary += f", and {count - max_items} more."
        
        return summary