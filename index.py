from langchain.document_loaders import DirectoryLoader
from opensearch_hybrid_search import OpenSearchHybridSearch
from resolvers import resolve_embeddings
from resolvers import EmbeddingType

loader = DirectoryLoader('./tacutopapers_test_rsids_10k/', glob="*.txt")
docs = loader.load()
for i, doc in enumerate(docs):
    doc.metadata["page_id"] = doc.metadata["source"].split("\\")[-1].split(".")[0]

embeddings = resolve_embeddings(EmbeddingType.HuggingFaceBGE)

docsearch = OpenSearchHybridSearch.from_documents(
    docs, embeddings, opensearch_url="https://localhost:9200",
    index_name = "index_test_rsids_10k",
    http_auth=("admin", "admin"),
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

docsearch.prepare_pipeline("https://localhost:9200", ("admin", "admin"), "norm-pipeline")

print("Finish")
