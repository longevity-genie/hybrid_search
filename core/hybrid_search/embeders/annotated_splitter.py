from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from hybrid_search.core.text_splitter import split_text_semantically, SemanticSplitter
# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.92


class AnnotatedSplitter(SemanticSplitter):

    def __init__(self, model: SentenceTransformer):
        super().__init__(model)

    # Override or extend methods from SemanticSplitter as needed
    # For example, if you need to add annotations to the split text
    def split(self, text: str, source: str, abstract: str | None = None, title: str | None = None, max_seq_length: int | None = None, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD, **kwargs) -> List[Document]:
        # Call the parent class's split method
        return self.split_text_semantically_annotated(
            text,
            source,
            max_seq_length,
            similarity_threshold,
            title,
            abstract
        )
    

    def split_text_semantically_annotated(
        self,
        text: str, 
        source: str,
        max_chunk_size: int | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        title: str | None = None,
        abstract: str | None = None
        ) -> List[Document]:
            # Use model's max sequence length if max_chunk_size is None
            if max_chunk_size is None:
                max_chunk_size = self.model.max_seq_length

            # Calculate adjusted chunk size using Document's method
            adjusted_max_chunk_size = Document.calculate_adjusted_chunk_size(
                self.model.tokenizer,
                max_chunk_size,
                title=title,
                abstract=abstract,
                source=source
            )
            
            # Get chunks using adjusted size
            chunks = self.split_text_semantically(text, adjusted_max_chunk_size, similarity_threshold)
            
            # Create Document objects for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    title=title,
                    abstract=abstract,
                    content=chunk,
                    source=source,
                    fragment_num=i + 1,
                    total_fragments=len(chunks)
                )
                documents.append(doc)
            
            return documents