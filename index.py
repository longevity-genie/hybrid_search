from typing import List

import click
from click import Context
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from pycomfort.logging import timing

from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    if ctx.invoked_subcommand is None:
        click.echo('Running the default command...')
        main()
    pass


@timing("indexing")
def index_function(data_path: str, glob_pattern: str, embedding: str, url: str, user: str, password: str, pipeline_name: str, index_name: str, device: str):
    loader = DirectoryLoader(data_path, glob=glob_pattern, loader_cls=TextLoader)
    docs: list[Document] = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata['page_id'] = doc.metadata['source'].split('/')[-1].split('.')[0]

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    if "bge" in embedding:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    docsearch: OpenSearchHybridSearch = OpenSearchHybridSearch.create(url, index_name, embeddings, login=user, password=password, pipeline_name=pipeline_name, documents=docs)
    # Prepare the pipeline with configurable arguments
    if not docsearch.check_pipeline_exists():
        print(f"hybrid search pipeline does not exist, creating it for {url}")
        docsearch.create_pipeline(url)
    print(f"Finished indexing with index name {index_name} and embedding {embedding} of data-path {data_path}")

# Main CLI command
@app.command("main")
@click.option('--data-path', show_default=True, default='data/tacutopapers_test_rsids_10k/', help='Path to the data directory.')
@click.option('--glob-pattern', show_default=True, default='*.txt', help='Glob pattern for files.')
@click.option('--embedding', show_default=True, default='BAAI/bge-base-en-v1.5', help='Type of embedding to use.') #can also be allenai/specter2_aug2023refresh
@click.option('--url', show_default=True, default='https://localhost:9200', help='URL for the pipeline.')
@click.option('--user', show_default=True, default='admin', help='Username for the pipeline.')
@click.option('--password', show_default=True, default='admin', help='Password for the pipeline.')
@click.option('--pipeline-name', show_default=True, default='norm-pipeline', help='Name of the pipeline.')
@click.option('--index_name', show_default=True, default='index-test_rsids_10k', help='Name of index')
@click.option('--device', show_default=True, default='cpu', help='Device to use')
def main(data_path: str, glob_pattern: str, embedding: str, url: str, user: str, password: str, pipeline_name: str, index_name: str, device: str):
    index_function(data_path, glob_pattern, embedding, url, user, password, pipeline_name, index_name, device)


@app.command("bge")
@click.pass_context
def bge_command(ctx, *args, **kwargs):
    # You can set default values for any option you want to override
    kwargs['embedding'] = 'BAAI/bge-base-en-v1.5'
    if 'index_name' not in kwargs:
        index_name = "index-bge-test_rsids_10k"
        print(f"no index name set, setting up default as {index_name}")
        kwargs['index_name'] = index_name

    # Call the main command with the new defaults
    ctx.invoke(main, *args, **kwargs)


@app.command("specter2")
@click.pass_context
def specter_command(ctx, *args, **kwargs):
    # You can set default values for any option you want to override
    kwargs['embedding'] = 'allenai/specter2_base' #'allenai/specter2_aug2023refresh'
    if 'index_name' not in kwargs:
        index_name = "index-specter2-test_rsids_10k"
        print(f"no index name set, setting up default as {index_name}")
        kwargs['index_name'] = index_name

    # Call the main command with the new defaults
    ctx.invoke(main, *args, **kwargs)


if __name__ == '__main__':
    app()