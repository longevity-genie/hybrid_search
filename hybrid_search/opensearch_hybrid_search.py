# implementing article https://opensearch.org/blog/hybrid-search/
import os

from langchain.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores.opensearch_vector_search import SCRIPT_SCORING_SEARCH
from langchain_community.vectorstores.opensearch_vector_search import _approximate_search_query_with_boolean_filter
from langchain_community.vectorstores.opensearch_vector_search import _approximate_search_query_with_efficient_filter
from langchain_community.vectorstores.opensearch_vector_search import _default_approximate_search_query
from langchain_community.vectorstores.opensearch_vector_search import MATCH_ALL_QUERY
from langchain_community.vectorstores.opensearch_vector_search import _default_script_query
from langchain_community.vectorstores.opensearch_vector_search import PAINLESS_SCRIPTING_SEARCH
from langchain_community.vectorstores.opensearch_vector_search import _default_painless_scripting_query
from typing import Any, List, Optional
import warnings

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from requests import Response

HYBRID_SEARCH = "hybrid_search"


class OpenSearchHybridSearch(OpenSearchVectorSearch):

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
            **kwargs: Any,
    ) -> 'OpenSearchHybridSearch':
        login = os.getenv("OPENSEARCH_USER", "admin") if login is None else login
        password = os.getenv("OPENSEARCH_PASSWORD", "admin") if password is None else password
        http_auth: tuple[str, str] = (login, password)
        if documents is None or len(documents) == 0:
            return cls(opensearch_url, index_name, embeddings,
                       http_auth=http_auth,
                       use_ssl=use_ssl,
                       verify_certs=verify_certs,
                       ssl_assert_hostname=ssl_assert_hostname,
                       ssl_show_warn=ssl_show_warn,
                       **kwargs)
        else:
            return OpenSearchHybridSearch.from_documents(
                documents, embeddings,
                opensearch_url=opensearch_url,
                index_name = index_name,
                http_auth=http_auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                ssl_assert_hostname=ssl_assert_hostname,
                ssl_show_warn=ssl_show_warn,
                **kwargs
            )

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

    @staticmethod
    def prepare_pipeline(url: str = "https://localhost:9200",
                         login: Optional[str] = None,
                         password: Optional[str] = None,
                         search_pipeline: str = "norm-pipeline",
                         normalization: str = "min_max",
                         combination: str = "arithmetic_mean",
                         verify: bool =False) -> Response:
        import requests
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

    def _raw_similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
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
        index_name = kwargs.get("index_name", self.index_name) #DO WE NEED IT?
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

        response = self.client.search(index=self.index_name, body=search_query, params=params)

        return [hit for hit in response["hits"]["hits"]]
