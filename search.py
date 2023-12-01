
import click
from hybrid_search.opensearch_hybrid_search import HYBRID_SEARCH
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from langchain.embeddings import HuggingFaceBgeEmbeddings


@click.command()
@click.option('--opensearch_url', default='http://localhost:9200', help='URL of the OpenSearch instance')
@click.option('--index_name', default='index_test_rsids_10k', help='Name of the index in OpenSearch')
@click.option('--device', default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
@click.option('--model_name', default='BAAI/bge-base-en-v1.5', help='Name of the model to use')
@click.option('--query', default='What is ageing?', help='The query to search for')
@click.option('--k', default=10, help='Number of search results to return')
def main(opensearch_url: str, index_name: str, device: str, model_name: str, query: str, k: int):
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    docsearch = OpenSearchHybridSearch(
        opensearch_url,
        index_name,
        embeddings
    )

    # Example functionality: Performing a search and printing results
    results = docsearch.similarity_search(query, k=k, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")

    print("Search Results:")
    for result in results:
        print(result)

if __name__ == '__main__':
    main()

'''
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from hybrid_search.opensearch_hybrid_search import HYBRID_SEARCH
import time
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

docsearch = OpenSearchHybridSearch(
    opensearch_url="https://localhost:9200",
    index_name = "index_test_rsids_10k",
    embedding_function = embeddings,
    http_auth=("admin", "admin"),
    use_ssl = True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False)

query = "What do rs123456789 and rs123456788?"
# Documents contain these rsids with such numbers:
# 10 contains both two times
# 11 contains both one time
# 12 and 13 contain only one rsid

# 20 contains both wrong rsids two times
# 21 contains both wrong rsids one time
# 22 and 23 contain only one wrong rsid

start = time.time()
docs = docsearch.similarity_search(query, k=10, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")
print(query, "opensearch size:", len(docs), " time:", time.time() - start)
for doc in docs:
    print(doc.metadata["page_id"])

query = "Comics superheroes"
# Only 114 document has text about superheroes.
# Text did not contain words 'comics' or 'superheroes'

start = time.time()
docs = docsearch.similarity_search(query, k=10, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")
print(query, "opensearch size:", len(docs), " time:", time.time() - start)
for doc in docs:
    print(doc.metadata["page_id"])


print("Finish")
'''