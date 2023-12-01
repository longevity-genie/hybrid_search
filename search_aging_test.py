from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from hybrid_search.opensearch_hybrid_search import HYBRID_SEARCH
from resolvers import resolve_embeddings
from resolvers import EmbeddingType
import time

embedding = resolve_embeddings(EmbeddingType.HuggingFaceBGE, model="BAAI/bge-base-en-v1.5")

docsearch = OpenSearchHybridSearch(
    opensearch_url="https://localhost:9200",
    index_name = "bge_base_en_v1.5_aging_5",
    embedding_function = embedding,
    http_auth=("admin", "admin"),
    use_ssl = True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False)

query = "What and how APOE rsids influence on aging?"

start = time.time()
docs = docsearch.similarity_search(query, k=10, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")
print(query, "opensearch size:", len(docs), " time:", time.time() - start)
for doc in docs:
    print(doc)


print("Finish")