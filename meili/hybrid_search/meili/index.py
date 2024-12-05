import typer
import meilisearch
import os
from dotenv import load_dotenv
from hybrid_search.meili.rag import *
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from hybrid_search.meili.rag import *
load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

app = typer.Typer()



@app.command()
def add_documents(
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    documents = [
        Document(id=1, name='test', description='test'),
        Document(id=2, name='test', description='test'),
    ]
    
    count = client.add_documents('test', documents)
    typer.echo(f"Added {count} documents to the 'test' index.")

@app.command()
def test_query(
    query: str = typer.Argument("test", help="Search query"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    results = client.search('test', query)
    
    typer.echo(f"Search results for '{query}':")
    for hit in results.hits:
        typer.echo(f"ID: {hit.id}, Name: {hit.name}, Description: {hit.description}")

@app.command()
def delete_index(
    index_name: str = typer.Option("test", help="Name of the index to delete"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    try:
        client.delete_index(index_name)
        typer.echo(f"Successfully deleted the '{index_name}' index.")
    except Exception as e:
        typer.echo(f"An error occurred while deleting the index: {e}")

@app.command()
def add_index(
    index_name: str = typer.Option("test", help="Name of the index to create"),
    primary_key: str = typer.Option("id", help="Primary key field name"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port"),
    model_name: str = typer.Option("Alibaba-NLP/gte-en-mlm-large", help="Model name"),
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    try:
        index = client.create_index(index_name, primary_key, model_name)
        typer.echo(f"Successfully created index '{index_name}' with primary key '{primary_key}'")
    except Exception as e:
        typer.echo(f"An error occurred while creating the index: {e}")

if __name__ == "__main__":
    app()