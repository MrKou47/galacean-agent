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

from langchain.text_splitter import TextSplitter
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
import os
import zipfile

from typing import List, Optional

class MarkdownLoader(CustomComponent):
    display_name: str = "Markdown Loader"

    def build_config(self):
      return {
        "zip_file": { "display_name": "zip_file", "field_type": "file", "suffixes": [".zip"] },
        "code": { "show": True }
      }

    def build(self, zip_file: str, ) -> BaseLoader:
        docs = self.load(zip_file)
        return docs
    
    def load(self, file_path: str) -> List[Document]:
      logging.info("file_path : %s", file_path)
      try:
          with zipfile.ZipFile(file_path, 'r') as zip_ref:
              all_names = zip_ref.namelist()
              logging.info("all_names : %s", all_names)
              file_names = [name for name in all_names if '__MACOSX' not in name]
              zip_ref.extractall(".")
              extracted_file_paths = [os.path.join(".", file_name) for file_name in file_names]
      except Exception as e:
          return [Document(page_content=f"An error occurred: {e}", metadata={})]

      docs = []
      try:
          for file_path in extracted_file_paths:
              with open(file_path, "rb") as docx_file_obj:
                  markdown = docx_file_obj.read().encode('utf-8')
              # 存到本地，便于排查    
              md_file_path = "./md/" + os.path.basename(file_path).encode('cp437').decode('utf-8')
              md_dir = os.path.dirname(md_file_path)
              if not os.path.exists(md_dir):
                  # 如果目录不存在，则创建它
                  os.makedirs(md_dir)
              with open(md_file_path, "w", encoding="utf-8") as md_file_obj:
                  md_file_obj.write(markdown)
              # 放到 document 对象里    
              docs.append(Document(page_content=markdown, metadata={}))
      except Exception as e:
          return [Document(page_content=f"An error occurred: {e}", metadata={})]
      return docs
