from typing import List

import click
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch

# Main CLI command
@click.command()
@click.option('--data-path', default='data/tacutopapers_test_rsids_10k/', help='Path to the data directory.')
@click.option('--glob-pattern', default='*.txt', help='Glob pattern for files.')
@click.option('--embedding', default='BAAI/bge-base-en-v1.5', help='Type of embedding to use.')
@click.option('--url', default='https://localhost:9200', help='URL for the pipeline.')
@click.option('--user', default='admin', help='Username for the pipeline.')
@click.option('--password', default='admin', help='Password for the pipeline.')
@click.option('--pipeline-name', default='norm-pipeline', help='Name of the pipeline.')
@click.option('--index_name', default='index_test_rsids_10k', help='Name of index')
@click.option('--device', default='cpu', help='Device to use')
def main(data_path: str, glob_pattern: str, embedding: str, url: str, user: str, password: str, pipeline_name: str, index_name: str, device: str):
    loader = DirectoryLoader(data_path, glob=glob_pattern, loader_cls=TextLoader)
    docs: list[Document] = loader.load()

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
    docsearch = OpenSearchHybridSearch.create(url, index_name, embeddings, documents=docs)
    # Prepare the pipeline with configurable arguments
    docsearch.prepare_pipeline(url)

if __name__ == '__main__':
    main()