
import click
from hybrid_search.opensearch_hybrid_search import HYBRID_SEARCH
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from langchain.embeddings import HuggingFaceBgeEmbeddings


@click.command()
@click.option('--url', default='http://localhost:9200', help='URL of the OpenSearch instance')
@click.option('--index', default='index_test_rsids_10k', help='Name of the index in OpenSearch')
@click.option('--device', default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
@click.option('--model', default='BAAI/bge-base-en-v1.5', help='Name of the model to use')
@click.option('--query', default='What is ageing?', help='The query to search for')
@click.option('--k', default=10, help='Number of search results to return')
def main(url: str, index: str, device: str, model: str, query: str, k: int):
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

    print("Search Results:")
    for result in results:
        print(result)

#"What and how APOE rsids influence on aging?"

if __name__ == '__main__':
    main()