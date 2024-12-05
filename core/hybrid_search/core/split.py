from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.93

def split_text_by_tokens(model: SentenceTransformer, text: str, max_seq_length: int | None = None) -> List[str]:
    # Get the tokenizer from the model
    tokenizer = model.tokenizer

    # If max_seq_length is not provided, use the model's max sequence length
    if max_seq_length is None:
        max_seq_length = model.max_seq_length

    # Tokenize the entire text
    tokens = tokenizer.tokenize(text)

    # Split tokens into chunks
    token_chunks = [tokens[i:i + max_seq_length] for i in range(0, len(tokens), max_seq_length)]

    # Convert token chunks back to text
    text_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]

    return text_chunks

def split_and_embed_text(
    model: SentenceTransformer, 
    text: str, 
    max_seq_length: int | None = None
) -> Tuple[List[str], np.ndarray]:
    """
    Split text into chunks and create embeddings for each chunk.
    
    Returns:
        Tuple containing list of text chunks and their corresponding embeddings
    """
    # Split the text into chunks
    text_chunks = split_text_by_tokens(model, text, max_seq_length)
    
    # Generate embeddings for all chunks
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    
    return text_chunks, embeddings


def split_text_semantically(
    model: SentenceTransformer,
    text: str,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Split text into semantically coherent chunks.
    
    Args:
        model: SentenceTransformer model for generating embeddings
        text: Input text to split
        max_chunk_size: Maximum token length for each chunk. If None, uses model's max_seq_length
        similarity_threshold: Threshold for combining sentences based on similarity
    
    Returns:
        List of semantically coherent text chunks
    """
    # Use model's max sequence length if max_chunk_size is None
    if max_chunk_size is None:
        max_chunk_size = model.max_seq_length

    # First, replace hyphenated line breaks with the complete word
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Replace simple split with regex-based splitting that preserves gene IDs
    sentence_pattern = r'(?<![A-Za-z0-9])[.!?](?=\s+[A-Z]|$)'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() + "." for s in sentences if s.strip()]
    
    # Get embeddings for all sentences
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    
    # Initialize chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        # Get token count for current sentence
        sentence_tokens = len(model.tokenizer.tokenize(sentence))
        
        if current_length + sentence_tokens > max_chunk_size and current_chunk:
            # Save current chunk and start new one
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # If current chunk is empty, add sentence directly
        if not current_chunk:
            current_chunk.append(sentence)
            current_length += sentence_tokens
            continue
            
        # Check semantic similarity with the last sentence in current chunk
        current_embedding = sentence_embeddings[i]
        last_embedding = sentence_embeddings[i-1]
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1), 
            last_embedding.reshape(1, -1)
        )[0][0]
        
        # If similar enough and within size limit, add to current chunk
        if similarity >= similarity_threshold and current_length + sentence_tokens <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_tokens
        else:
            # Save current chunk and start new one with current sentence
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def split_and_embed_text_semantically(
    model: SentenceTransformer,
    text: str,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> Tuple[List[str], np.ndarray]:
    """
    Split text semantically into chunks and create embeddings for each chunk.
    
    Returns:
        Tuple containing list of text chunks and their corresponding embeddings
    """
    # Split the text into semantic chunks
    text_chunks = split_text_semantically(model, text, max_chunk_size, similarity_threshold)
    
    # Generate embeddings for all chunks
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    
    return text_chunks, embeddings

def split_text_file_semantically(
    file_path: Path | str,
    model: SentenceTransformer,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Read text from a file and split it into semantically coherent chunks.
    
    Args:
        model: SentenceTransformer model for generating embeddings
        file_path: Path to the text file to process
        max_chunk_size: Maximum token length for each chunk. If None, uses model's max_seq_length
        similarity_threshold: Threshold for combining sentences based on similarity
    
    Returns:
        List of semantically coherent text chunks
    """
    # Convert string path to Path object if necessary
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Read the text file
    text = file_path.read_text()
    
    # Use existing function to split text semantically
    return split_text_semantically(model, text, max_chunk_size, similarity_threshold)

def split_text_semantically_annotated(
    model: SentenceTransformer,
    text: str, 
    source: str,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    mention_splits: bool = True,
    title: str | None = None,
    abstract: str | None = None
) -> List[str]:
    # Use model's max sequence length if max_chunk_size is None
    if max_chunk_size is None:
        max_chunk_size = model.max_seq_length

    # Build prefix text with optional title and abstract
    prefix_text = ""
    if title:
        prefix_text += f"TITLE: {title}\n"
    if abstract:
        prefix_text += f"ABSTRACT: {abstract}\n\n"
    
    # Get base chunks using existing function
    chunks = split_text_semantically(model, text, max_chunk_size, similarity_threshold)
    
    # Determine if we need to include fragment information
    has_multiple_chunks = len(chunks) > 1
    
    if has_multiple_chunks:
        prefix_text += "TEXT_FRAGMENT: "
    
    # Add source and fragment placeholder after the text
    suffix_text = f"\n\nSOURCE: {source}"
    if mention_splits and has_multiple_chunks:
        suffix_text += "\tFRAGMENT: 999/999"
    
    # Calculate tokens for prefix and suffix text
    total_metadata_tokens = len(model.tokenizer.tokenize(prefix_text + suffix_text))
    
    # If we have multiple chunks, adjust max_chunk_size and resplit
    if has_multiple_chunks:
        adjusted_max_chunk_size = max_chunk_size - total_metadata_tokens
        chunks = split_text_semantically(model, text, adjusted_max_chunk_size, similarity_threshold)
    
    # Build full annotation for each chunk
    annotated_chunks = []
    for i, chunk in enumerate(chunks):
        prefix = ""
        if title:
            prefix += f"TITLE: {title}\n"
        if abstract:
            prefix += f"ABSTRACT: {abstract}\n"
        if has_multiple_chunks:
            prefix += "\nTEXT_FRAGMENT: "
        
        suffix = f"\n\nSOURCE: {source}"
        if mention_splits and has_multiple_chunks:
            suffix += f"\tFRAGMENT: {i+1}/{len(chunks)}"
            
        annotated_chunks.append(f"{prefix}{chunk}{suffix}\n")
    
    return annotated_chunks

def split_text_file_semantically_annotated(
    file_path: Path | str,
    model: SentenceTransformer,
    source: str | None = None,
    max_chunk_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    mention_splits: bool = True,
    title: str | None = None,
    abstract: str | None = None
) -> List[str]:
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