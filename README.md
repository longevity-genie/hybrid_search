# hybrid search instructions
Created following this article https://opensearch.org/blog/hybrid-search/

## Installation:

With conda or micromamba setup the environment:
```
micromamba create -f environment.yaml
micromamba activate hybrid_search
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

### Manual installation

- Go to https://opensearch.org/downloads.html and download OpenSearch choose the installation variant you like. OpenSearch Dashboards is a convenient tool but not mandatory.
- Install the latest Java
- For Windows unpack the archive. In opensearch_folder/config/opensearch.yml make sure plugins.security.ssl.http.enabled: true. Because it works correctly only with ssl on, despite some functionality still being available with http. Launch opensearch-windows-install.bat, despite the name it is not an installer but a main launcher.
- For Linux use docker or follow instructions in the documentation.

## Usage:
- Launch open-search either with docker-compose or java
- Launch index.py for the initial indexing test dataset. It creates an index and pipeline for hybrid search.
- Activate environment
```bash
micromamba activate hybrid_search #to activate environment
pip install -e . #[optional] install current package locally
```
- Launch search to perform test search.
```bash
python index.py #to index
python search.py # to search, uses default query
```
If you want to use another embedding, for example specter2, try:
```bash
python index.py
```

## Tests

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
python search test_rsids
```

Also, similar test for "Comics superheroes" that will test embeddings:
* Only 114 document has text about superheroes, but text did not contain words 'comics' or 'superheroes'

You can test them by:
```
python search test_heroes
```

Right now testing is not automated and you have to call CLI to test


## Troubleshooting

If something is not working with OpenSearch, read log messages carefully. For example, if you have small disk space it can block writing (watermark issue) that will cause failing with different final error message.
