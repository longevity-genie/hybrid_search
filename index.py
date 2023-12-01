import click
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings

from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch

# Main CLI command
@click.command()
@click.option('--data-path', default='data/tacutopapers_test_rsids_10k/', help='Path to the data directory.')
@click.option('--glob-pattern', default='*.txt', help='Glob pattern for files.')
@click.option('--embedding', default='BAAI/bge-base-en-v1.5', help='Type of embedding to use.')
@click.option('--pipeline-url', default='http://localhost:9200', help='URL for the pipeline.')
@click.option('--pipeline-user', default='admin', help='Username for the pipeline.')
@click.option('--pipeline-password', default='admin', help='Password for the pipeline.')
@click.option('--pipeline-name', default='norm-pipeline', help='Name of the pipeline.')
@click.option('--device', default='cpu', help='Device to use')
def main(data_path: str, glob_pattern: str, embedding: str, pipeline_url: str, pipeline_user: str, pipeline_password: str, pipeline_name: str, device: str):
    loader = DirectoryLoader(data_path, glob=glob_pattern)
    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata['page_id'] = doc.metadata['source'].split('/')[-1].split('.')[0]

    #model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    docsearch = OpenSearchHybridSearch.from_documents(
        docs, embeddings, opensearch_url=pipeline_url,
        index_name = "index_test_rsids_10k",
        http_auth=(pipeline_user, pipeline_password),
        use_ssl = True,
        verify_certs = False,
        ssl_assert_hostname = False,
        ssl_show_warn = False,
    )
    # Prepare the pipeline with configurable arguments
    docsearch.prepare_pipeline(pipeline_url, (pipeline_user, pipeline_password), pipeline_name)

    # Rest of the script logic
    # Add any additional processing or logic here

if __name__ == '__main__':
    main()