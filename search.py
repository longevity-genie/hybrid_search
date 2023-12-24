import click
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from hybrid_search.opensearch_hybrid_search import HYBRID_SEARCH
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch

@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(search)

@app.command("search")
@click.option('--url', default='http://localhost:9200', help='URL of the OpenSearch instance')
@click.option('--index', default='index_test_rsids_10k', help='Name of the index in OpenSearch')
@click.option('--device', default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
@click.option('--model', default='BAAI/bge-base-en-v1.5', help='Name of the model to use')
@click.option('--query', default='What is ageing?', help='The query to search for')
@click.option('--k', default=10, help='Number of search results to return')
@click.option('--verbose', default=False, help='How much to print')
def search(url: str, index: str, device: str, model: str, query: str, k: int, verbose: bool):
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    docsearch = OpenSearchHybridSearch.create(url, index, embeddings)

    # Example functionality: Performing a search and printing results
    results = docsearch.similarity_search(query, k=k, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")

    print("Search IDS:")
    for result in results:
        print(result.metadata["page_id"])
        if verbose:
            print(result)


@app.command("test_apoe")
@click.pass_context
def test_apoe(ctx):
    ctx.invoke(search, query="What and how APOE rsids influence on aging?")

@app.command("test_rsids")
@click.pass_context
def test_rsids(ctx):
    ctx.invoke(search, query="rs123456789 and rs123456788")


@app.command("test_heroes")
@click.pass_context
def test_heroes(ctx):
    ctx.invoke(search, query="comic superheroes")


if __name__ == '__main__':
    app()