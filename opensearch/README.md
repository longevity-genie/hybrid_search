# Hybrid Search Testing Project
A project to test and implement hybrid search capabilities using different search engines and embedding models. Currently focused on OpenSearch implementation, following this article: https://opensearch.org/blog/hybrid-search/

## Installation:

With Poetry:
```bash
poetry install
poetry shell
```

For the OpenSearch itself there are several installation options. 

### From docker-compose

This repository goes with a test two nodes open-search cluster together with a dashboard.

Optional: change OPENSEARCH_JAVA_OPTS=-Xms2512m -Xmx2512m according to your RAM availability, usually it is recommended to have them equal in side.
Start docker-compose:
```bash
docker compose up
```
Open http://localhost:5601/ to explore the dashboard, "admin" is used both as user and passport by default.

For additional OpenSearch setup options, please refer to the OpenSearch submodule documentation.

## About
This project provides tools to test and compare different hybrid search implementations. Hybrid search combines traditional keyword-based search with semantic (embedding-based) search to provide more accurate and contextually relevant results.

Currently implemented:
- OpenSearch-based hybrid search
- Support for multiple embedding models (BGE, Specter2)
- Test suite with controlled test cases

## Usage:
- Launch open-search either with docker-compose or java
- Launch index.py for the initial indexing test dataset. It creates an index and pipeline for hybrid search.
- Activate environment
```bash
poetry shell  # to activate environment
```
- Create .env file, you can use .env.template
- Launch search to perform test search.
```bash
python index.py #to index
python search.py # to search, uses default query
```
You can also tune index.py parameters. For example:
```
python index.py main --url https://agingkills.eu:9200 --user admin --password admin --index_name index-bge-test_rsids_10k --embedding BAAI/bge-base-en-v1.5

```

If you want to use another embedding, for example specter2, try:
```bash
python index.py specter2
```

## Tests
The project includes carefully crafted test datasets to evaluate hybrid search performance:

### RSID test
Tests the system's ability to handle both exact and fuzzy matching of RSIDs (genetic reference SNP cluster IDs).

There are text pieces deliberately incorporated into tacutu papers data ( /data/tacutopapers_test_rsids_10k )
In particular for rs123456789 and rs123456788 as well as similar but misspelled rsids are added to the documents:
* 10.txt contains both two times
* 11.txt contains both one time
* 12.txt and 13 contain only one rsid
* 20.txt contains both wrong rsids two times
* 21.txt contains both wrong rsids one time
* 22.txt and 23 contain only one wrong rsid

You can test them by:
```
python search.py test_rsids
```

### Comics superheroes test
Tests semantic search capabilities by finding contextually relevant documents without exact keyword matches.

Also, similar test for "Comics superheroes" that will test embeddings:
* Only 114 document has text about superheroes, but text did not contain words 'comics' or 'superheroes'

You can test them by:
```
python search.py test_heroes
```

These tests help evaluate:
- Exact match capabilities
- Fuzzy matching performance
- Semantic understanding
- Hybrid ranking effectiveness

Right now testing is not automated and you have to call CLI to test


## Troubleshooting

If something is not working with OpenSearch, read log messages carefully. For example, if you have small disk space it can block writing (watermark issue) that will cause failing with different final error message.