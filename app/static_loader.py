from typing import List
from langchain_core.documents import Document

from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import TextSplitter, Language, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class StaticLoader:
  ts_loader = GenericLoader.from_filesystem( 
  "./dist/engine",
  glob="**/*",
  suffixes=[".ts"],
  parser=LanguageParser(language="ts")
)

  md_loader = DirectoryLoader(
    path="./dist/docs",
    glob="*.md",
    recursive=True,
  )

  ts_splitter: TextSplitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.TS,
    chunk_size=1000, chunk_overlap=200
  )

  md_splitter: TextSplitter = MarkdownHeaderTextSplitter(
    headers_to_split_on = [
      ("#", "Header 1"),
      ("##", "Header 2"),
      ("###", "Header 3"),
      ("####", "Header 4"),
    ],
  )

  def load(self) -> List[Document]:
    ts_documents = self.ts_loader.load()
    md_documents = self.md_loader.load()
    code_chunks = self.ts_splitter.split_documents(ts_documents)
    md_chunks: List[Document] = []
    for doc in md_documents:
      md_chunks.extend(self.md_splitter.split_text(doc.page_content))
      
    docs = code_chunks.extend(md_chunks)
    return docs
