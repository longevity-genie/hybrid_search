import typer
import meilisearch
import os
from dotenv import load_dotenv

load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY")

app = typer.Typer()

@app.command()
def add_documents(
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    key = os.getenv("MEILI_MASTER_KEY")
    client = meilisearch.Client(f'http://{host}:{port}', key)
    
    # Try to get the index, if it doesn't exist, create it
    try:
        index = client.get_index('test')
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e)}")
        print("Attempting to create index...")
        index = client.create_index('test', {'primaryKey': 'id'})
    
    documents = [
        {'id': 1, 'name': 'test', 'description': 'test'},
        {'id': 2, 'name': 'test', 'description': 'test'},
    ]
    
    index.add_documents(documents)
    typer.echo(f"Added {len(documents)} documents to the 'test' index.")

@app.command()
def test_query(
    query: str = typer.Argument("test", help="Search query"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    client = meilisearch.Client(f'http://{host}:{port}', key)
    index = client.index('test')

    
    results = index.search(query)
    index.search
    
    typer.echo(f"Search results for '{query}':")
    for hit in results['hits']:
        typer.echo(f"ID: {hit['id']}, Name: {hit['name']}, Description: {hit['description']}")

@app.command()
def delete_index(
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    client = meilisearch.Client(f'http://{host}:{port}', key)
    
    try:
        client.delete_index('test')
        typer.echo("Successfully deleted the 'test' index.")
    except Exception as e:
        typer.echo(f"An error occurred while deleting the index: {e}")

@app.command()
def add_index(
    index_name: str = typer.Argument(..., help="Name of the index to create"),
    primary_key: str = typer.Option("id", help="Primary key field name"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port"),
    model_name: str = typer.Option("BAAI/bge-m3", help="Model name"),
):
    client = meilisearch.Client(f'http://{host}:{port}', key)
    embedder = {'name': 'test', 'source': 'huggingFace', 'vector_name': 'test', 'model_name': 'BAAI/bge-m3', 'new_line_replacement': '\\n'}

    try:
        index = client.create_index(index_name, {'primaryKey': primary_key})
        typer.echo(f"Successfully created index '{index_name}' with primary key '{primary_key}'")
        index.update_embedders(
            )
    except Exception as e:
        typer.echo(f"An error occurred while creating the index: {e}")

if __name__ == "__main__":
    app()