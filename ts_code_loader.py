import logging
from langflow import CustomComponent
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.git import GitLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter, Language, RecursiveCharacterTextSplitter
from langchain.schema import Document
import zipfile
from typing import List, Optional
import os

class TsCodeLoader(CustomComponent):
  display_name: str = "Ts Code Loader"
  description: str = "Load typescript code from a github repository."

  def build_config(self):
    return {
      "zipfile": { "display_name": "zip_file", "field_type": "file", "suffixes": [".zip"] },
      "code": { "show": True }
    }

  def build(self, zipfile: str) -> BaseLoader:
    self.unzip(zipfile, "./engine-sourcecode")
    loader = GenericLoader.from_filesystem( 
      "./engine-sourcecode",
      glob="**/*",
      suffixes=[".ts"],
      parser=LanguageParser(language="ts")
    )
    logging.info('ts_code_loader start loading')
    codes = loader.load()
    return codes
  
  def unzip(self, zip_file_path, dest_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
      for member in zip_ref.namelist():
        if os.path.splitext(member)[1] == '.ts' and '__MACOSX' not in member:
          destination_path = os.path.join(dest_folder_path, member)
          destination_directory = os.path.dirname(destination_path)

          if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
          with zip_ref.open(member) as source, open(destination_path, "w") as target:
            logging.info(f"member is: {member}")
            logging.info(f"File extracted: {destination_path}")
            target.write(source.read().decode('utf-8'))