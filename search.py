from langchain.document_loaders import DirectoryLoader
from opensearch_hybrid_search import OpenSearchHybridSearch
from opensearch_hybrid_search import HYBRID_SEARCH
from resolvers import resolve_embeddings
from resolvers import EmbeddingType
import time

loader = DirectoryLoader('./tacutopapers_test_rsids_10k/', glob="*.txt")
docs = loader.load()
for i, doc in enumerate(docs):
    doc.metadata["page_id"] = doc.metadata["source"].split("\\")[-1].split(".")[0]

embeddings = resolve_embeddings(EmbeddingType.HuggingFaceBGE)

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

query = "Comics superherows"
# Only 114 document has text about superherows.
# Text did not contain words 'comics' or 'superherow'

start = time.time()
docs = docsearch.similarity_search(query, k=10, search_type = HYBRID_SEARCH, search_pipeline = "norm-pipeline")
print(query, "opensearch size:", len(docs), " time:", time.time() - start)
for doc in docs:
    print(doc.metadata["page_id"])


print("Finish")