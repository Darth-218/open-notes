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
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(config) if config else Config.load()
    ctx.obj["app"] = OpenNotes(ctx.obj["config"])


@cli.command()
@click.pass_context
def index(ctx: click.Context) -> None:
    app: OpenNotes = ctx.obj["app"]
    result = app.index_all()
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int) -> None:
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
    app: OpenNotes = ctx.obj["app"]
    app.watch()


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
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
    app: OpenNotes = ctx.obj["app"]
    result = app.get_stats()
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
