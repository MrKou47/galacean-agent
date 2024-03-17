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

class ZipLoader(CustomComponent):
    display_name: str = "Zip Loader"
    description: str = "Load raw text files from a zip file."

    def build_config(self):
      return {
        "zip_file": { "display_name": "zip_file", "field_type": "file", "suffixes": [".zip"] },
        "file_ext": { "display_name": "file_ext", "field_type": "text" },
        "code": { "show": True }
      }
    
    def build(self, zip_file: str, file_ext: str) -> BaseLoader:
        target_fodler = os.path.basename(zip_file)
        docs = self.load(zip_file, target_fodler, file_ext)
        return docs

    def load(zip_file_path, dest_folder_path, file_ext):
      with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
          if os.path.splitext(member)[1] == file_ext:
              destination_path = os.path.join(dest_folder_path, member)
              destination_directory = os.path.dirname(destination_path)

              if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
                with zip_ref.open(member) as source, open(destination_path, "w") as target:
                  target.write(source.read().decode())