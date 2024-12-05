import typer
from sentence_transformers import SentenceTransformer
from pprint import pprint
from hybrid_search.core.models import *
from hybrid_search.core.explore import *
from pathlib import Path
from hybrid_search.core.split import *
app = typer.Typer()


from typing import Union


def load_auto(model_name_or_path: str):
    import torch
from transformers import AutoTokenizer, AutoModel


@app.command()
def main():
    """
    Main function.
    """
    #text: str = "For the differentiation of muscle PAX3 and PAX7 are important, but MYOD is the most important transcription factor"
    model: SentenceTransformer = load_gte_mlm_en()
    #model = load_bioembeddings()
    #pprint(see_tokens(text, gte))
    print("model: ", model)
    print("dimensions: ", model.get_sentence_embedding_dimension())
    print("max_seq_length: ", model.max_seq_length)
    file = Path("/home/antonkulaga/sources/hybrid_search/data/tacutopapers_test_rsids_10k/108.txt")
    splits = split_text_file_semantically_annotated(file, model, similarity_threshold=0.93, source="/home/antonkulaga/sources/hybrid_search/data/tacutopapers_test_rsids_10k/108.txt")
    for split in splits:
        print("=========================")
        print(split)

    #typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()
