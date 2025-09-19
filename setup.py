from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto-graph-rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tune language models for Cypher query generation on knowledge graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/auto-graph-rag",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "networkx>=3.0",
        "kuzu>=0.6.0",
        "openai>=1.0",
        "transformers>=4.40",
        "peft>=0.10",
        "datasets>=2.0",
        "accelerate>=0.30",
        "bitsandbytes>=0.40",
        "pandas>=2.0",
        "pydantic>=2.0",
        "typer>=0.12",
        "rich>=13.0",
        "torch>=2.0",
        "langchain>=0.2.0",
        "langchain-openai>=0.1.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "black>=24.0",
            "ruff>=0.4",
            "mypy>=1.10",
            "pre-commit>=3.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "auto-graph-rag=auto_graph_rag.cli:main",
        ],
    },
)