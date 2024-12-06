from pathlib import Path
import re
from pydantic import BaseModel, Field, ConfigDict
from abc import ABC, abstractmethod
import numpy as np


class Document(BaseModel):
    content: str
    vectors: np.ndarray = Field(default=np.array([]), alias="_vectors") #we use underscore as many databases want _vectors
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_defaults=True
    )

    def save_to_yaml(self, path: Path) -> Path:
        """Save document to a YAML file
        
        Args:
            path: Path where the YAML file should be saved
        """ 
        import yaml
        import numpy as np
        
        # Convert to dict and ensure path parent exists
        data = self.model_dump()
        
        # Simplify numpy array serialization
        if isinstance(data.get('vectors'), np.ndarray):
            data['vectors'] = data['vectors'].tolist()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML file
        with path.open('w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)
        return path


class ArticleDocument(Document):

    """Represents a document or document fragment with its metadata"""
    title: str | None
    abstract: str | None
    source: str
    fragment_num: int
    total_fragments: int
    vectors: np.ndarray | None = Field(default=None, alias="_vectors")  # Use same name for serialization


    def set_vectors(self, vectors: np.ndarray) -> None:
        """Set the document's vector embeddings"""
        self._vectors = vectors

    def get_vectors(self) -> np.ndarray | None:
        """Get the document's vector embeddings"""
        return self._vectors

    def to_formatted_string(self, mention_splits: bool = True) -> str:
        """
        Convert the document to a formatted string representation.
        
        Args:
            mention_splits: Whether to include fragment information
        
        Returns:
            Formatted string with metadata and content
        """
        parts = []
        
        if self.title:
            parts.append(f"TITLE: {self.title}\n")
        if self.abstract:
            parts.append(f"ABSTRACT: {self.abstract}\n")
            
        has_multiple_fragments = self.total_fragments > 1
        if has_multiple_fragments:
            parts.append("TEXT_FRAGMENT: ")
        
        parts.append(self.content)
        
        parts.append(f"\n\nSOURCE: {self.source}")
        if mention_splits and has_multiple_fragments:
            parts.append(f"\tFRAGMENT: {self.fragment_num}/{self.total_fragments}")
        
        parts.append("\n")
        
        return "\n".join(parts)

    @staticmethod
    def calculate_adjusted_chunk_size(
        tokenizer,
        max_chunk_size: int,
        title: str | None = None,
        abstract: str | None = None,
        source: str | None = None
    ) -> int:
        """
        Calculate the adjusted chunk size accounting for metadata tokens.
        
        Args:
            tokenizer: The tokenizer to use for token counting
            max_chunk_size: Original maximum chunk size
            title: Optional title text
            abstract: Optional abstract text
            source: Optional source identifier
            
        Returns:
            Adjusted maximum chunk size accounting for metadata
        """
        # Build sample metadata text
        metadata_text = ""
        if title:
            metadata_text += f"TITLE: {title}\n"
        if abstract:
            metadata_text += f"ABSTRACT: {abstract}\n"
        if source:
            metadata_text += f"\n\nSOURCE: {source}"
        metadata_text += "\tFRAGMENT: 999/999\n"  # Account for worst-case fragment notation
        
        # Calculate tokens for metadata
        metadata_tokens = len(tokenizer.tokenize(metadata_text))
        
        # Return adjusted size
        return max_chunk_size - metadata_tokens