# hybrid search instructions
Created following this article https://opensearch.org/blog/hybrid-search/

## Installation:

With conda or micromamba setup the environment:
```
micromamba create -f environment.yaml
microammba activate hybrid seach
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
- Launch index.py for the initial indexing test dataset. It creates an index and pipeline for hybrid search.
- Launch search to perform test search.

## Troubleshooting

If there is crazy libssl1_1 error, try to install it manually. It is the fault of unstructured.PaddleOCR dependency which directory loader of langchain is using.
If something is not working with OpenSearch, read log messages carefully. For example, if you have small disk space it can block writing (watermark issue) that will cause failing with different final error message.
