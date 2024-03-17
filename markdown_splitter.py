"""Functionality for splitting text."""
from __future__ import annotations

import logging
from typing import (
Iterable,
TypeVar,
)

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")

import logging

from langflow import CustomComponent
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document

from typing import List, Optional

class CharacterSplitter(CustomComponent):
    display_name: str = "Markdown Splitter"
    description: str = "Split text into chunks based on a separator."

    def build(self, Documents: BaseLoader) -> TextSplitter:
        result = self.split_documents(Documents)
        logging.info("CharacterSplitter result: " + str(result))
        return result
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        markdown_splitter = MarkdownTextSplitter()
        _metadatas = metadatas or [{}] * len(texts)
        documents = markdown_splitter.create_documents(texts, metadatas=_metadatas)
        return documents
        