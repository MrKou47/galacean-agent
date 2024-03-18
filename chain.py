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
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.sequential import SequentialChain
from langchain_core.runnables import RunnableSequence
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain import hub

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
  len(ts_documents), ts_documents[0].metadata
)

print(
  len(md_documents), md_documents[0].metadata
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

memory = ConversationBufferMemory(
  memory_key="chat_history",
  input_key="question",
  output_key='answer',
  return_messages=True
)

llm_chain = LLMChain(
  llm=llm,
  prompt=ChatPromptTemplate.from_messages(
    [
      ("system", "You are an AI assistant named Galacean Agent, proficient in front-end development, OpenGL, and WebGL. You are currently serving as a Technical Support Engineer for a WebGL game engine called Galacean Engine. Your role is to assist users in resolving issues related to using the Galacean Engine."),
      ("system", "Here are documents and engine source codes that might help you to solve the problem: {context}."),
      ("system", "don't make up messages."),
      ("system", "Here's the user's question: {question}"),
    ]
  ),
)

template = (
  "Combine the chat history and follow up question into "
  "a standalone question, and then translate the standalone querstion into English. If the standalone is already English, then you can directly return it."
  "If Chat History is empty or NOT_FOUND, then you can just handle the follow up question."
  "Chat History: {chat_history}"
  "Follow up question: {question}"
)
prompt = PromptTemplate.from_template(template)
question_generator_chain = LLMChain(
  llm=llm,
  prompt=prompt,
)

document_prompt = PromptTemplate(
  input_variables=["page_content"],
  template="{page_content}"
)
document_variable_name = "context"

combine_docs_chain = StuffDocumentsChain(
  llm_chain=llm_chain,
  document_prompt=document_prompt,
  document_variable_name=document_variable_name,
)

chain = ConversationalRetrievalChain(
  retriever=retriever,
  combine_docs_chain=combine_docs_chain,
  question_generator=question_generator_chain,
  return_source_documents=True,
  output_key="answer",
  memory=memory,
  response_if_no_docs_found="I'm sorry, I couldn't find any documents that match your query.",
  get_chat_history=lambda arguments : "NOT_FOUND"
)

result = chain.invoke("如何使用 Galacean Engine 中的物理系统?")

source_documents: List[Document] = result['source_documents']

"""
print the metadata about the source documents
"""
for doc in source_documents:
  print(doc.metadata)