
import click
from typing import List, Dict, Any, Tuple
from FlagEmbedding import BGEM3FlagModel

@click.command()
@click.option('--use-fp16', default=False, is_flag=True, help='Whether to use fp16 precision.')
@click.option('--passage', prompt='Enter the passage to encode',
              help='The passage to encode using the BGE M3 model.')
def main(use_fp16: bool, passage: str):
    """A CLI tool for encoding passages using the BGE M3 model."""
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=use_fp16)
    passage_embeddings = model.encode([passage], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    print_embeddings(passage_embeddings)


def print_embeddings(embeddings: Dict[str, Any]) -> None:
    """Prints the embeddings in a readable format."""
    if 'colbert_vecs' in embeddings:
        print(f"ColBERT Vectors Shape: {embeddings['colbert_vecs'][0].shape}")
    if 'dense' in embeddings:
        print(f"Dense Embedding Shape: {embeddings['dense'][0].shape}")
    if 'sparse' in embeddings:
        print("Sparse Embedding:", embeddings['sparse'][0])

if __name__ == '__main__':
    main()
