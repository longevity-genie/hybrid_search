from typing import List

import click
import loguru
from click import Context
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from opensearchpy import OpenSearch, RequestsHttpConnection
from pycomfort.config import configure_logger, LogLevel, LOG_LEVELS, load_environment_keys
from pycomfort.logging import timing
from typing import Optional
from FlagEmbedding import BGEM3FlagModel
from hybrid_search.novel_embeddings import BgeM3Embeddings
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
import logging
from opensearchpy import OpenSearch, exceptions
from opensearchpy import OpenSearch, RequestsHttpConnection

@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx: Context):
    if ctx.invoked_subcommand is None:
        click.echo('Running the default command...')
        main()
    pass


@timing("indexing")
def index_function(data_path: str, glob_pattern: str, embedding: str, url: str, user: str, password: str, pipeline_name: str, index_name: str, device: str, space: str, logger: Optional["loguru.Logger"] = None):
    logger.info(f"indexing from {data_path} using pattern: {glob_pattern} \n using {embedding} with URL {url} \n  USER: {user}  PASSWORD: {password} \n index_name {index_name} ")
    loader = DirectoryLoader(data_path, glob=glob_pattern, loader_cls=TextLoader)
    docs: list[Document] = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata['page_id'] = doc.metadata['source'].split('/')[-1].split('.')[0]

    model_kwargs = {"device": device, "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}

    if "bge-m3" in embedding:
        embeddings = BgeM3Embeddings(use_fp16=True, device=device)
    elif "bge" in embedding:
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
    logger.info("starting to indexing")
    docsearch: OpenSearchHybridSearch = OpenSearchHybridSearch.create(url, index_name, embeddings,
                                                                      login=user, password=password,
                                                                      pipeline_name=pipeline_name,
                                                                      documents=docs,
                                                                      space_type=space
                                                                      )
    # Prepare the pipeline with configurable arguments
    if not docsearch.check_pipeline_exists():
        logger.info(f"hybrid search pipeline does not exist, creating it for {url}")
        docsearch.create_pipeline(url)
    logger.info(f"Finished indexing with index name {index_name} and embedding {embedding} of data-path {data_path}")


# Main CLI command
@app.command("main")
@click.option('--data-path', show_default=True, default='data/tacutopapers_test_rsids_10k/', help='Path to the data directory.')
@click.option('--glob-pattern', show_default=True, default='*.txt', help='Glob pattern for files.')
@click.option('--embedding', show_default=True, default='Alibaba-NLP/gte-large-en-v1.5', help='Type of embedding to use.') #can also use BAAI/bge-en-icl , BAAI/bge-large-en-v1.5 and also be allenai/specter2_aug2023refresh
@click.option('--url', show_default=True, default='https://localhost:9200', help='URL for the pipeline.')
@click.option('--user', show_default=True, default='admin', help='Username for the pipeline.')
@click.option('--password', show_default=True, default='admin', help='Password for the pipeline.')
@click.option('--pipeline-name', show_default=True, default='norm-pipeline', help='Name of the pipeline.')
@click.option('--index_name', show_default=True, default='index-gte-test_rsids_10k', help='Name of index')
@click.option('--device', show_default=True, default='cpu', help='Device to use')
@click.option('--space', type=click.Choice(["cosinesimil", "l2", "innerproduct", "l1", "linf"], False), default='l2', help='Space to use for OpenSearch')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def main(data_path: str, glob_pattern: str, embedding: str, url: str, user: str, password: str, pipeline_name: str, index_name: str, device: str, space: str, log_level: str):
    logger = configure_logger(log_level)
    logger.add("./logs/hybrid_index_{time}.log")
    load_environment_keys(usecwd=True)
    return index_function(data_path, glob_pattern, embedding, url, user, password, pipeline_name, index_name, device, space, logger)


@app.command("test_connection")
@click.option('--url', default='https://localhost:9200', help='URL of the OpenSearch cluster')
@click.option('--username', default='admin', help='Username for the OpenSearch cluster')
@click.option('--password', default='admin', help='Password for the OpenSearch cluster')
@click.option('--use-ssl', default=True, type=bool, help='Use SSL for connection')
@click.option('--ssl-show-warn', default=False, type=bool, help='Show SSL warnings')
def test_opensearch(url: str, username: str, password: str, use_ssl: bool, ssl_show_warn: bool):
    """ Connects to OpenSearch and adds a test index with test data. """
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('opensearchpy').setLevel(logging.DEBUG)
    """ Connects to OpenSearch and adds a test index with test data. """
    try:
        # Initialize the OpenSearch client
        client = OpenSearch(
            hosts=[url],
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=ssl_show_warn,
            connection_class=RequestsHttpConnection,
            trust_env=True
        )

        # Create a test index
        index_name = 'test_index'
        client.indices.create(index=index_name, ignore=400)

        # Add test data
        test_data = {"name": "Test Name", "description": "This is a test entry."}
        response = client.index(index=index_name, body=test_data)

        # Print success message
        if response.get('result') in ['created', 'updated']:
            click.echo("Test data added successfully!")
        else:
            click.echo("Failed to add test data.")

    except exceptions.OpenSearchException as e:
        click.echo(f"Error interacting with OpenSearch: {e}")

    @app.command("delete_index")
    @click.option('--url', default='https://localhost:9200', help='URL of the OpenSearch cluster')
    @click.option('--username', default='admin', help='Username for the OpenSearch cluster')
    @click.option('--password', default='admin', help='Password for the OpenSearch cluster')
    @click.option('--use-ssl', default=True, type=bool, help='Use SSL for connection')
    @click.option('--ssl-show-warn', default=False, type=bool, help='Show SSL warnings')
    @click.option('--index-name', default='test_index', help='Name of the index to be deleted')
    def delete_index(url: str, username: str, password: str, use_ssl: bool, ssl_show_warn: bool, index_name: str):
        """ Deletes a specified index from the OpenSearch cluster. """
        try:
            # Initialize the OpenSearch client
            client = OpenSearch(
                hosts=[url],
                http_auth=(username, password),
                use_ssl=use_ssl,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=ssl_show_warn,
                connection_class=RequestsHttpConnection,
                trust_env=True
            )

            # Delete the specified index
            response = client.indices.delete(index=index_name, ignore=[400, 404])

            # Print success message
            if response.get('acknowledged', False):
                click.echo(f"Index '{index_name}' deleted successfully.")
            else:
                click.echo(f"Index '{index_name}' was not found or could not be deleted.")

        except exceptions.OpenSearchException as e:
            click.echo(f"Error interacting with OpenSearch: {e}")

@app.command("bge")
@click.pass_context
def bge_command(ctx, *args, **kwargs):
    # You can set default values for any option you want to override
    kwargs['embedding'] = 'BAAI/bge-large-en-v1.5' #'BAAI/bge-base-en-v1.5'
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