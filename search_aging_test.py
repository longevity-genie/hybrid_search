
import click
from langchain.embeddings import HuggingFaceBgeEmbeddings
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from hybrid_search.opensearch_hybrid_search import HYBRID_SEARCH
import time

@click.command()
@click.option('--opensearch_url', default='http://localhost:9200', help='URL of the OpenSearch instance')
@click.option('--query', default="What and how APOE rsids influence on aging?", help='query to search for')
@click.option('--device', default='cpu', help='Device to use (e.g., cpu, gpu).')
@click.option('--model_name', default='BAAI/bge-base-en-v1.5', help='Name of the model to use.')
@click.option('--k', default=10, help='Number of search results to return')
def main(opensearch_url: str, query: str, device: str, model_name: str, k: int):
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    docsearch = OpenSearchHybridSearch(
        opensearch_url=opensearch_url,
        index_name = "bge_base_en_v1.5_aging_5",
        embedding_function = embeddings,
        http_auth=("admin", "admin"),
        use_ssl = True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False)


    start = time.time()
    docs = docsearch.similarity_search(query, k=k, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")
    print(query, "opensearch size:", len(docs), " time:", time.time() - start)
    for doc in docs:
        print(doc)


    print("Finish")

if __name__ == '__main__':
    main()