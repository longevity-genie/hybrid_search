import typer
from sentence_transformers import SentenceTransformer
from pprint import pprint
from hybrid_search.core.embeddings import *
from hybrid_search.core.utils import *
from hybrid_search.core.text_splitter import TextSplitter
from pathlib import Path
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
    #simple_splitter = 
    #splits = split_text_file_semantically_annotated(file, model, similarity_threshold=0.92, source="/home/antonkulaga/sources/hybrid_search/data/tacutopapers_test_rsids_10k/108.txt")
    splitter = TextSplitter(model)
    documents = splitter.split_file(file)
    for document in documents:
        #print(f"=============SHAPE:======={document.vectors.shape}=========================")
        #pprint(document.vectors)
        print(document.content)
        print("===============================================")
        #print(document.save_to_yaml(Path(file.name).with_suffix(".yaml")))
    #typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()
