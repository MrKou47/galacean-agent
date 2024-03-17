from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()

from typing import List
from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.base import BaseLoader
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter, Language, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory.chat_memory import ChatMessageHistory

print("start")

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

ts_documents = ts_loader.load()
md_documents = md_loader.load()

print(
  len(ts_documents)
)

print(
  len(md_documents)
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

code_chunks = ts_splitter.split_documents(ts_documents)

md_chunks: List[Document] = []

for doc in md_documents:
  md_chunks.extend(md_splitter.split_text(doc.page_content))
  
docs = code_chunks.extend(md_chunks)

embedding = OpenAIEmbeddings()

db = None

print('start db')

if os.path.exists("./dist/chroma"):
  db = Chroma(
    embedding_function=embedding,
    persist_directory="./dist/chroma",
  )
else:
  db = Chroma.from_documents(
    documents=code_chunks,
    embedding=embedding,
    persist_directory="./dist/chroma",
  )

retriever = db.as_retriever(
  search_type="mmr",  # Also test "similarity"
  search_kwargs={"k": 8},
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

memory = ConversationSummaryMemory(
  llm=llm, memory_key="chat_history", return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

question = "How to add Physix ability to an entity?"
result = qa.invoke(question)

print(
  result["answer"],
)


# combine_docs_chain = create_stuff_documents_chain(
#   llm=llm,
#   memory=memory,
#   prompt=ChatPromptTemplate.from_messages(
#     [
#       ("system", "You are an AI assistant named {name}. Professional in Front end, OpenGL and WebGL. Now you are acting as a Technical Support Engineer for a WebGL game engine which called Galacean Engine. You havbe to help users solve the problem related to using Galacean Engine. Here are documents and engine source codes that might help you to solve the problem: {context}."),  
#       MessagesPlaceholder(variable_name="messages"),
#     ]
#   ),
# )

# chain = create_retrieval_chain(
#   retriever=retriever,
#   combine_docs_chain=combine_docs_chain,
# ).assign(name="Galacean Agent")

# chain.invoke()