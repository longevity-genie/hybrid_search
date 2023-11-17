# hybrid search instructions
Created following this article https://opensearch.org/blog/hybrid-search/

## Installation:
- Go to https://opensearch.org/downloads.html and download OpenSearch choose the installation variant you like. OpenSearch Dashboards is a convenient tool but not mandatory.
- Install the latest Java
- For Windows unpack the archive. In opensearch_folder/config/opensearch.yml make sure plugins.security.ssl.http.enabled: true. Because it works correctly only with ssl on, despite some functionality still being available with http. Launch opensearch-windows-install.bat, despite the name it is not an installer but a main launcher.
- For Linux use docker or follow instructions in the documentation.

## Usage:
- Launch index.py for the initial indexing test dataset. It creates an index and pipeline for hybrid search.
- Launch search to perform test search.
