from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from hybrid_search.core.text_splitter import split_text_semantically
# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.92

    
def split_text_semantically_annotated(
    model: SentenceTransformer,
    text: str, 
    source: str,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    mention_splits: bool = True,
    title: str | None = None,
    abstract: str | None = None
) -> List[Document]:
    # Use model's max sequence length if max_chunk_size is None
    if max_chunk_size is None:
        max_chunk_size = model.max_seq_length

    # Calculate adjusted chunk size using Document's method
    adjusted_max_chunk_size = Document.calculate_adjusted_chunk_size(
        model.tokenizer,
        max_chunk_size,
        title=title,
        abstract=abstract,
        source=source
    )
    
    # Get chunks using adjusted size
    chunks = split_text_semantically(model, text, adjusted_max_chunk_size, similarity_threshold)
    
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


def split_text_file_semantically_annotated(
    file_path: Path | str,
    model: SentenceTransformer,
    source: str | None = None,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    mention_splits: bool = True,
    title: str | None = None,
    abstract: str | None = None
) -> List[Document]:
    """
    Read text from a file and split it into semantically coherent chunks with annotations.
    
    Args:
        file_path: Path to the text file to process
        model: SentenceTransformer model for generating embeddings
        source: Source identifier to be added to each chunk (defaults to filename if None)
        max_chunk_size: Maximum token length for each chunk. If None, uses model's max_seq_length
        similarity_threshold: Threshold for combining sentences based on similarity
        mention_splits: Whether to include split numbers in annotations (defaults to True)
        title: Optional title to include in annotations
        abstract: Optional abstract to include in annotations
    
    Returns:
        List of semantically coherent text chunks with annotations
    """
    # Convert string path to Path object if necessary
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Read the text file
    text = file_path.read_text()
    
    # If no source is provided, use the filename
    if source is None:
        source = file_path.name
    
    # Use existing function to split text semantically with annotations
    return split_text_semantically_annotated(
        model, 
        text, 
        source,
        max_chunk_size, 
        similarity_threshold,
        mention_splits,
        title,
        abstract
    )