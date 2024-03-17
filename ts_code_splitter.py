import logging
from langflow import CustomComponent
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter, Language, RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class TsCodeSplitter(CustomComponent):
  display_name: str = "Ts Code Splitter"

  def build(self, Documents: BaseLoader) -> TextSplitter:
    return self.splitting(Documents)
  
  def splitting(self, documents: List[Document]) -> List[Document]:
    logging.info('create splitting')
    sparators = [
        "\nenum ",
        "\ninterface ",
        "\nnamespace ",
        "\ntype ",
        # Split along class definitions
        "\nclass ",
        # Split along function definitions
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        # Split along control flow statements
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        # Split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
    ]

    ts_splitter = RecursiveCharacterTextSplitter(sparators, chunk_size=60, chunk_overlap=0)

    logging.info('start splitting')
    return ts_splitter.split_documents(documents)
