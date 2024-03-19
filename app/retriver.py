import os
from typing import List, Callable
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

retriever_exists = os.path.exists("./dist/chroma")

def get_retriever(persist_directory: str, docs_loader: Callable[[], List[Document]]):
  embedding = OpenAIEmbeddings()

  db: Chroma = None
  if retriever_exists:
    db = Chroma(
      embedding_function=embedding,
      persist_directory=persist_directory,
    )
  else:
    db = Chroma.from_documents(
      documents=docs_loader(),
      embedding=embedding,
      persist_directory=persist_directory,
    )

  retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
  )
  return retriever