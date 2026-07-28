#!/usr/bin/env python3
"""
adapters/cli/cli.py — Click CLI for the Madagascar RAG system.

Entry point: python -m src.adapters.cli.cli
             or the `rag` command if installed via pyproject.toml.
"""

from __future__ import annotations

import json
import os
import sys

import click
import openai
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.application.index_usecase import BuildIndexUseCase
from src.application.query_usecase import QueryUseCase
from src.infrastructure.chunker import WikipediaChunker
from src.infrastructure.embedder import OpenAIEmbedder
from src.infrastructure.generator import OpenAIGenerator, OpenAITranslator
from src.infrastructure.hybrid_retriever import HybridRetriever
from src.infrastructure.loader import WikipediaLoader

INDEX_PATH = os.environ.get("INDEX_PATH", "index")


def _make_container():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        click.echo("Error: OPENAI_API_KEY not set. Add it to .env or your environment.", err=True)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    retriever = HybridRetriever(index_path=INDEX_PATH)
    embedder = OpenAIEmbedder(client)

    return {
        "retriever": retriever,
        "build_uc": BuildIndexUseCase(
            loader=WikipediaLoader(),
            chunker=WikipediaChunker(),
            embedder=embedder,
            retriever=retriever,
            index_path=INDEX_PATH,
        ),
        "query_uc": QueryUseCase(
            retriever=retriever,
            embedder=embedder,
            generator=OpenAIGenerator(client),
            translator=OpenAITranslator(client),
        ),
    }


@click.group()
def cli():
    """🇲🇬 Madagascar RAG — ask questions about the Wikipedia article."""


@cli.command("build")
@click.option("--force", is_flag=True, help="Force re-fetch + re-embed even if index exists")
def build(force: bool):
    """Build (or reload) the retrieval index."""
    c = _make_container()
    stats = c["build_uc"].execute(force_rebuild=force)
    click.echo(f"\n✅ Index ready — {stats.total_chunks} chunks "
               f"({stats.text_chunks} text, {stats.table_chunks} tables)")


@cli.command("query")
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, help="Chunks to retrieve")
@click.option("--json-output", is_flag=True, help="Output full result as JSON")
def query(question: str, top_k: int, json_output: bool):
    """Ask a question (English or French)."""
    c = _make_container()
    c["build_uc"].execute()  # no-op if index already exists

    result = c["query_uc"].execute(question, top_k=top_k)

    if json_output:
        click.echo(json.dumps(result.dict(), ensure_ascii=False, indent=2))
        return

    sep = "=" * 64
    click.echo(f"\n{sep}")
    click.echo(f"Q: {result.query}")
    if result.detected_language == "fr":
        click.echo(f"   [Translated: {result.search_query_used}]")
    click.echo(f"\nA: {result.answer}")
    if result.sources:
        click.echo("\nSources:")
        for s in result.sources:
            click.echo(f"  #{s.chunk_id:03d}  [{s.type.value:5s}]  {s.section}")
    click.echo(f"{sep}\n")


@cli.command("interactive")
@click.option("--top-k", default=5, show_default=True)
def interactive(top_k: int):
    """Start an interactive REPL session."""
    c = _make_container()
    c["build_uc"].execute()

    click.echo("\n🇲🇬  Madagascar RAG — Interactive Mode")
    click.echo("   Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            question = click.prompt("Question", prompt_suffix=" > ").strip()
        except (click.Abort, EOFError):
            click.echo("\nBye!")
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = c["query_uc"].execute(question, top_k=top_k)
        click.echo(f"\nAnswer: {result.answer}")
        sections = ", ".join({s.section for s in result.sources})
        click.echo(f"Sources: {sections}\n")


@cli.command("evaluate")
@click.option("--output", default="eval_results.json", show_default=True)
def evaluate(output: str):
    """Run the full evaluation suite (17 questions)."""
    # Delegate to the standalone evaluate.py script
    import subprocess
    subprocess.run([sys.executable, "evaluate.py", "--output", output], check=True)


if __name__ == "__main__":
    cli()
