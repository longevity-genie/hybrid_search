from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
from FlagEmbedding import BGEM3FlagModel
from pydantic import ConfigDict


class BgeM3Embeddings(BaseModel, Embeddings):

    model_name: str = "BAAI/bge-m3"
    """Model name to use."""
    model: Any | BGEM3FlagModel
    vector_name: str = 'dense_vecs'
    new_line_replacement: str = " "
    vector_name: str = 'dense_vecs'
    normalize_embeddings: bool = True
    use_fp16: bool = True
    # Setting use_fp16 to True speeds up computation with a slight performance degradation

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=self.use_fp16, normalize_embeddings=self.normalize_embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", self.new_line_replacement) for t in texts]
        return_sparse = "sparse" in self.vector_name
        return_colbert = "colbert" in self.vector_name
        embeddings = self.model.encode(texts, return_dense=True, return_sparse=return_sparse, return_colbert_vecs=return_colbert)[self.vector_name]
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", self.new_line_replacement)
        return_sparse = "sparse" in self.vector_name
        return_colbert = "colbert" in self.vector_name
        embedding = self.model.encode(text, return_dense=True, return_sparse=return_sparse, return_colbert_vecs=return_colbert)[self.vector_name]
        return embedding.tolist()