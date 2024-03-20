from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.12'
DESCRIPTION = 'Hybrid search with OpenSearch and Langchain'
LONG_DESCRIPTION = 'Package for the hybrid search with opensearch and langchain library'

# Setting up
setup(
    name="hybrid_search",
    version=VERSION,
    author="Alex Karmazin",
    author_email="<karmazinalex@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pycomfort>=0.0.15', 'click','transformers>=4.35.2', "sentence_transformers", "opensearch-py>=2.4.2", "langchain>=0.1.12", "langchain-community>=0.0.28"],
    keywords=['python', 'llm', 'science', 'review', 'hybrid search', 'semantic search'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)