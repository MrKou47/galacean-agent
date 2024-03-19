from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

from typing import List
from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import TextSplitter, Language, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

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
  search_type="mmr",
  search_kwargs={"k": 8},
)

llm = ChatOpenAI(temperature=0)

memory = ConversationBufferMemory(
  memory_key="chat_history",
  input_key="question",
  output_key='answer',
  return_messages=True
)

llm_stream = ChatOpenAI(
  model_name="gpt-4",
  temperature=0.2,
  streaming=True,
  callbacks=[
    StreamingStdOutCallbackHandler()
  ]
)

llm_chain = LLMChain(
  llm=llm_stream,
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
  "and then translate the question into English. If the question is already English, then you can directly return it."
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
  get_chat_history=lambda arguments : "NOT_FOUND",
)

