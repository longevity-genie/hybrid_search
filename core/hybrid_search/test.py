import typer
from sentence_transformers import SentenceTransformer
from pprint import pprint
from hybrid_search.core.models import *
from hybrid_search.core.explore import *

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
    model, tokenizer = load_medcpt_query_model_with_tokenizer()
    text: str = "For the differentiation of muscle PAX3 and PAX7 are important, but MYOD is the most important transcription factor"
    print(see_auto_tokens(text, model, tokenizer))
    #pprint(see_tokens(text, model))
    #typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()
