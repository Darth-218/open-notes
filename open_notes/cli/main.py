from __future__ import annotations

import json
import sys

import click

from open_notes import OpenNotes
from open_notes.config import Config


@click.group()
@click.option("--config", type=click.Path(), help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """Main entry point for the OpenNotes CLI.

    Initializes the application context with the specified configuration
    and sets up the OpenNotes application instance.

    Args:
        ctx: Click context object for passing data between commands.
        config: Optional path to a custom configuration file.

    Returns:
        None. Initializes ctx.obj with config and app instances.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(config) if config else Config.load()
    ctx.obj["app"] = OpenNotes(ctx.obj["config"])


@cli.command()
@click.pass_context
def index(ctx: click.Context) -> None:
    """Index all notes in the knowledge base.

    Scans the configured knowledge base directory and indexes all notes
    for semantic search. The index is used by search and query commands.

    Returns:
        None. Prints JSON result of indexing operation to stdout.
    """
    app: OpenNotes = ctx.obj["app"]
    result = app.index_all()
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int) -> None:
    """Search notes using semantic similarity.

    Performs a semantic search across all indexed notes using the configured
    embedding model and returns the most similar results.

    Args:
        query: The search query string.

    Returns:
        None. Prints search results with scores to stdout.

    Options:
        -k, --top-k: Number of results to return (default: 5).
    """
    app: OpenNotes = ctx.obj["app"]
    results = app.search(query, top_k=top_k)

    if not results:
        click.echo("No results found")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"[{i}] {r.heading_path or 'Note'}")
        click.echo(f"    {r.content[:200]}...")
        click.echo(f"    Score: {r.score:.3f} ({r.source})")
        click.echo()


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Number of context chunks")
@click.pass_context
def query(ctx: click.Context, query: str, top_k: int) -> None:
    """Query notes using RAG with an LLM.

    Performs retrieval-augmented generation by finding relevant notes and
    generating an answer using the configured LLM.

    Args:
        query: The question to ask about the notes.

    Returns:
        None. Prints the generated answer and sources to stdout.

    Options:
        -k, --top-k: Number of context chunks to retrieve (default: 5).
    """
    app: OpenNotes = ctx.obj["app"]
    result = app.query(query, top_k=top_k)

    click.echo("=" * 60)
    click.echo(result["answer"])
    click.echo("=" * 60)

    if result["sources"]:
        click.echo("\nSources:")
        for i, s in enumerate(result["sources"], 1):
            heading = s.get("heading_path") or s.get("note_id", "Note")
            click.echo(f"[{i}] {heading}")
            click.echo(f"    {s['content'][:150]}...")


@cli.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the MCP server for remote note queries.

    Initializes and runs the Model Context Protocol server, allowing
    remote clients to query notes via stdio or HTTP transport.

    Returns:
        None. Server runs until terminated.

    Raises:
        SystemExit: If transport type is invalid.
    """
    config: Config = ctx.obj["config"]
    app: OpenNotes = ctx.obj["app"]

    from open_notes.mcp.server import init_mcp, mcp

    init_mcp(
        config=config,
        query_engine=app.query_engine,
        rag_pipeline=app.rag_pipeline,
        note_storage=app.note_storage,
    )

    transport = config.mcp_transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=config.mcp_host, port=config.mcp_port)


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch for file changes and auto-index notes.

    Starts a file system watcher that monitors the knowledge base
    directory for changes and automatically re-indexes modified notes.

    Returns:
        None. Runs indefinitely until interrupted.
    """
    app: OpenNotes = ctx.obj["app"]
    app.watch()


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Display the current configuration.

    Shows all configuration values including paths, models, and settings
    that are currently in use.

    Returns:
        None. Prints configuration values to stdout.
    """
    cfg: Config = ctx.obj["config"]
    click.echo("Configuration:")
    click.echo(f"  knowledge_base_path: {cfg.knowledge_base_path}")
    click.echo(f"  embedding_model: {cfg.embedding_model}")
    click.echo(f"  embedding_dimension: {cfg.embedding_dimension}")
    click.echo(f"  vector_db_path: {cfg.vector_db_path}")
    click.echo(f"  keyword_db_path: {cfg.keyword_db_path}")
    click.echo(f"  llm_provider: {cfg.llm_provider}")
    click.echo(f"  llm_model_path: {cfg.llm_model_path}")
    click.echo(f"  search_mode: {cfg.search_mode}")
    click.echo(f"  search_top_k: {cfg.search_top_k}")
    click.echo(f"  mcp_host: {cfg.mcp_host}")
    click.echo(f"  mcp_port: {cfg.mcp_port}")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Display statistics about the indexed notes.

    Shows statistics including the total number of notes, chunks, and
    embeddings in the knowledge base.

    Returns:
        None. Prints JSON statistics to stdout.
    """
    app: OpenNotes = ctx.obj["app"]
    result = app.get_stats()
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()