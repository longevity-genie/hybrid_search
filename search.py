from typing import Optional

import click
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from hybrid_search.novel_embeddings import BgeM3Embeddings
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch, rerank_results, rerank_extend_results


@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(search)

@app.command("search")
@click.option('--url', default='https://localhost:9200', help='URL of the OpenSearch instance')
@click.option('--index', default='index-gte-test_rsids_10k', help='Name of the index in OpenSearch') #index-bge-en-icl_rsids_10k
@click.option('--device', default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
@click.option('--embedding', default='Alibaba-NLP/gte-large-en-v1.5', help='Name of the model to use') #can be BAAI/bge-large-en-v1.5 or BAAI/bge-en-icl
@click.option('--query', default='What is ageing?', help='The query to search for')
@click.option('--k', default=10, help='Number of search results to return')
@click.option('--threshold', default=None, help='Threshold to cut out results')
@click.option('--verbose', default=False, help='How much to print')
def search(url: str, index: str, device: str, embedding: str, query: str, k: int, threshold: Optional[float], verbose: bool):
    print(f"searching in INDEX: {index}, \nQUERY: {query}")
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    if "bge-m3" in embedding:
        embeddings = BgeM3Embeddings(use_fp16=True, device=device)
    else:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    docsearch: OpenSearchHybridSearch = OpenSearchHybridSearch.create(url, index, embeddings)

    # Example functionality: Performing a search and printing results
    #results = docsearch.similarity_search_with_score(query, k, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")
    found = docsearch.hybrid_search(query, k, search_pipeline = "norm-pipeline", threshold=threshold)
    print(f"len results {len(found)}")
    results = rerank_extend_results(query, found)
    print("Search IDS:")
    for (result, f, f2) in results:
        if "page_id" in result.metadata:
            print(result.metadata["page_id"], f, f2)
        if verbose:
            print(result)


@app.command("test_apoe")
@click.pass_context
def test_apoe(ctx):
    ctx.invoke(search, query="What and how APOE rsids influence on aging?")

@app.command("test_rsids")
@click.pass_context
def test_rsids(ctx):

    """
    :param ctx:
    :return:
    In particular for rs123456789 and rs123456788 as well as similar but misspelled rsids are added to the documents:
        * 10.txt contains both two times
        * 11.txt contains both one time
        * 12.txt and 13 contain only one rsid
        * 20.txt contains both wrong rsids two times
        * 21.txt contains both wrong rsids one time
        * 22.txt and 23 contain only one wrong rsid
    """
    ctx.invoke(search, query="rs123456789 and rs123456788")


@app.command("test_heroes")
@click.pass_context
def test_heroes(ctx):
    ctx.invoke(search, query="comic superheroes")


if __name__ == '__main__':
    app()