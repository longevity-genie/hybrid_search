# implementing article https://opensearch.org/blog/hybrid-search/
import os
import pprint

from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.vectorstores.opensearch_vector_search import SCRIPT_SCORING_SEARCH
from langchain_community.vectorstores.opensearch_vector_search import _approximate_search_query_with_boolean_filter
from langchain_community.vectorstores.opensearch_vector_search import _approximate_search_query_with_efficient_filter
from langchain_community.vectorstores.opensearch_vector_search import _default_approximate_search_query
from langchain_community.vectorstores.opensearch_vector_search import MATCH_ALL_QUERY
from langchain_community.vectorstores.opensearch_vector_search import _default_script_query
from langchain_community.vectorstores.opensearch_vector_search import PAINLESS_SCRIPTING_SEARCH
from langchain_community.vectorstores.opensearch_vector_search import _default_painless_scripting_query

from FlagEmbedding import FlagReranker
from typing import Any, List, Optional, Tuple
import warnings
import requests

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests import Response

HYBRID_SEARCH = "hybrid_search"

def rerank_results(query: str, documents: list[Document], reranker: Optional[FlagReranker] = None) -> list[(str, Document)]:
    """reranks the resulting documents"""
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) if reranker is None else reranker
    sentences = [(query, d.page_content) for d in documents]
    ranks = reranker.compute_score(sentences)
    return list(zip(documents, ranks))

def rerank_extend_results(query: str, documents: list[(Document, float)], reranker: Optional[FlagReranker] = None) -> list[(str, Document)]:
    """reranks the resulting documents"""
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) if reranker is None else reranker
    sentences = [(query, d[0].page_content) for d in documents]
    ranks = reranker.compute_score(sentences)
    return [(documents[i][0], documents[i][1], ranks[i]) for i in range(0, len(documents))]


class OpenSearchHybridSearch(OpenSearchVectorSearch):

    opensearch_url: str
    login: str
    password: str
    client: OpenSearch
    with_reranker: bool = False
    reranker: Optional[FlagReranker] = None

    def delete_index(self, index_name: str):
        return self.client.indices.delete(index=index_name, ignore=[400, 404])

    # Function to check if the pipeline exists
    def check_pipeline_exists(self, pipeline_id: str = "norm-pipeline"):
        try:
            # Attempt to get the pipeline
            response = self.client.ingest.get_pipeline(id=pipeline_id)
            if response:
                print(f"Pipeline '{pipeline_id}' exists.")
                return True
        except Exception as e:
            # If the pipeline does not exist, an exception is thrown
            print(f"Pipeline '{pipeline_id}' does not exist. Exception was {str(e) if e is not None else ''}.")
            return False

    @classmethod
    def create(
            cls,
            opensearch_url: str,
            index_name: str,
            embeddings: Embeddings,
            login: Optional[str] = None,
            password: Optional[str] = None,
            use_ssl: bool = True,
            verify_certs: bool = False,
            ssl_assert_hostname: bool = False,
            ssl_show_warn: bool = False,
            documents: Optional[list[Document]] = None,
            space_type: str = "cosinesimil", #"l2"
            engine: str = "nmslib",
            with_reranker: bool = False,
            timeout: int = 120,
            **kwargs: Any,
    ) -> 'OpenSearchHybridSearch':
        login = os.getenv("OPENSEARCH_USER", "admin") if login is None else login
        password = os.getenv("OPENSEARCH_PASSWORD", "admin") if password is None else password
        http_auth: tuple[str, str] = (login, password)
        if documents is None or len(documents) == 0:
            result = cls(opensearch_url, index_name, embeddings,
                       http_auth=http_auth,
                       use_ssl=use_ssl,
                       verify_certs=verify_certs,
                       ssl_assert_hostname=ssl_assert_hostname,
                       ssl_show_warn=ssl_show_warn,
                       trust_env=True,
                       connection_class=RequestsHttpConnection,
                       space_type=space_type,
                       engine = engine,
                       timeout = timeout
                       **kwargs)
        else:
            result = OpenSearchHybridSearch.from_documents(
                documents, embeddings,
                opensearch_url=opensearch_url,
                index_name = index_name,
                http_auth=http_auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                ssl_assert_hostname=ssl_assert_hostname,
                ssl_show_warn=ssl_show_warn,
                trust_env=True,
                connection_class=RequestsHttpConnection,
                space_type=space_type,
                engine = engine,
                timeout = timeout, #lets make it large
                **kwargs
            )
        result.opensearch_url = opensearch_url
        result.login = login
        result.password = password
        result.with_reranker = with_reranker
        if result.with_reranker:
            result.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        return result

    @staticmethod
    def _hybrid_search_query(query_vector: List[float], k: int = 4, vector_field: str = "vector_field", query: str = ""):
        return {
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "text": {
                                    "query": query
                                }
                            }
                        },
                        {
                            "knn": {vector_field: {"vector": query_vector, "k": k}}
                        }
                    ]
                }
            }
        }
    """
    """
    @staticmethod
    def create_pipeline(url: str = "https://localhost:9200",
                        login: Optional[str] = None,
                        password: Optional[str] = None,
                        search_pipeline: str = "norm-pipeline",
                        normalization: str = "min_max",
                        combination: str = "arithmetic_mean",
                        verify: bool =False) -> Response:
        login = os.getenv("OPENSEARCH_USER", "admin") if login is None else login
        password = os.getenv("OPENSEARCH_PASSWORD", "admin") if password is None else password
        auth: tuple[str, str] = (login, password)

        # Making a PUT request
        r: Response = requests.request(method="PUT", url=f"{url}/_search/pipeline/{search_pipeline}", auth=auth, verify=verify, json={
                          "description": "Post-processor for hybrid search",
                          "phase_results_processors": [
                            {
                              "normalization-processor": {
                                "normalization": {
                                  "technique": normalization
                                },
                                "combination": {
                                  "technique": combination
                                }
                              }
                            }
                          ]
                        })

        # check status code for response received
        # success code - 200
        # print(r)

        # print content of request
        # print(r.content)
        return r


    def hybrid_search(self, query: str, k: int = 8,
                      search_pipeline: str = "norm-pipeline",
                      vector_field: str = "vector_field",
                      text_field: str = "text",
                      metadata_field: str = "metadata",
                      threshold: Optional[float] = None,
                      **kwargs: Any
                      ) -> list[tuple[Document, float]]:
        """
        Method that already has appropriate pipeline for the search
        :param query:
        :param k:
        :return:
        """

        results: list[tuple[Document, float]] = self.similarity_search_with_score(query, k=k,
                                      search_type = HYBRID_SEARCH,
                                      search_pipeline = search_pipeline,
                                      vector_field = vector_field,
                                      text_field = text_field,
                                      metadata_field = metadata_field,
                                      **kwargs
                                      )
        upd_results = results if threshold is None else [(r, f) for (r, f) in results if f >= threshold]
        return upd_results

    def _raw_similarity_search_with_score(
        self, query: str, k: int = 8, **kwargs: Any
    ) -> List[dict]:
        """Return raw opensearch documents (dict) including vectors,
        scores most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of dict with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        embedding = self.embedding_function.embed_query(query)
        search_type = kwargs.get("search_type", "approximate_search")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        filter = kwargs.get("filter", {})

        if (
            self.is_aoss
            and search_type != "approximate_search"
            and search_type != SCRIPT_SCORING_SEARCH
        ):
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search` and `script_scoring`"
            )

        if search_type == "approximate_search":
            boolean_filter = kwargs.get("boolean_filter", {})
            subquery_clause = kwargs.get("subquery_clause", "must")
            efficient_filter = kwargs.get("efficient_filter", {})
            # `lucene_filter` is deprecated, added for Backwards Compatibility
            lucene_filter = kwargs.get("lucene_filter", {})

            if boolean_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `boolean_filter` and `efficient_filter` are provided which "
                    "is invalid"
                )

            if lucene_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `lucene_filter` and `efficient_filter` are provided which "
                    "is invalid. `lucene_filter` is deprecated"
                )

            if lucene_filter != {} and boolean_filter != {}:
                raise ValueError(
                    "Both `lucene_filter` and `boolean_filter` are provided which "
                    "is invalid. `lucene_filter` is deprecated"
                )

            if (
                efficient_filter == {}
                and boolean_filter == {}
                and lucene_filter == {}
                and filter != {}
            ):
                if self.engine in ["faiss", "lucene"]:
                    efficient_filter = filter
                else:
                    boolean_filter = filter

            if boolean_filter != {}:
                search_query = _approximate_search_query_with_boolean_filter(
                    embedding,
                    boolean_filter,
                    k=k,
                    vector_field=vector_field,
                    subquery_clause=subquery_clause,
                )
            elif efficient_filter != {}:
                search_query = _approximate_search_query_with_efficient_filter(
                    embedding, efficient_filter, k=k, vector_field=vector_field
                )
            elif lucene_filter != {}:
                warnings.warn(
                    "`lucene_filter` is deprecated. Please use the keyword argument"
                    " `efficient_filter`"
                )
                search_query = _approximate_search_query_with_efficient_filter(
                    embedding, lucene_filter, k=k, vector_field=vector_field
                )
            else:
                search_query = _default_approximate_search_query(
                    embedding, k=k, vector_field=vector_field
                )
        elif search_type == SCRIPT_SCORING_SEARCH:
            space_type = kwargs.get("space_type", "l2")
            pre_filter = kwargs.get("pre_filter", MATCH_ALL_QUERY)
            search_query = _default_script_query(
                embedding, k, space_type, pre_filter, vector_field
            )
        elif search_type == PAINLESS_SCRIPTING_SEARCH:
            space_type = kwargs.get("space_type", "l2Squared")
            pre_filter = kwargs.get("pre_filter", MATCH_ALL_QUERY)
            search_query = _default_painless_scripting_query(
                embedding, k, space_type, pre_filter, vector_field
            )
        elif search_type == HYBRID_SEARCH:
            search_query = self._hybrid_search_query(embedding, k=k, vector_field=vector_field, query=query)
        else:
            raise ValueError("Invalid `search_type` provided as an argument")

        search_pipeline = kwargs.get("search_pipeline", None)
        params = {}
        if search_pipeline is not None:
            params["search_pipeline"] = search_pipeline

        response = self.client.search(index=index_name, body=search_query, params=params)
        #['total', 'max_score', 'hits']
        return [hit for hit in response["hits"]["hits"]]