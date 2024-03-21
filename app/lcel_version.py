from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

from typing import List
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, Runnable, RunnableLambda, RunnableSequence
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain import hub

from static_loader import StaticLoader
from retriver import retriever_exists, get_retriever

static_loader = StaticLoader()

retriever = get_retriever("./dist/chroma", static_loader.load)

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
)

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are an AI assistant named Galacean Agent, proficient in front-end development, OpenGL, and WebGL. You are currently serving as a Technical Support Engineer for a WebGL game engine called Galacean Engine. Your role is to assist users in resolving issues related to using the Galacean Engine."),
    ("system", "Here are documents and engine source codes that might help you to solve the problem: {context}."),
    ("system", "don't make up messages."),
    ("system", "Here's the user's question: {input}"),
  ]
)

document_chain = create_stuff_documents_chain(llm_stream, prompt)

retriever_chain = create_retrieval_chain(retriever, document_chain)

def format_question(x: str) -> str:
  return { "input": x }

chain = RunnableSequence(first=RunnableLambda(format_question), last=retriever_chain)

def msg_handler(message: str, histroy):
  partial_message = ""
  metadata = []
  receiving_answer = False
  try:
    for chunk in chain.stream(input=message):
      print('chunk is', chunk)
      if "input" in chunk and chunk["input"]:
        partial_message = ""
      if "context" in chunk and chunk["context"]:
        for item in chunk["context"]:
          metadata.append(item.metadata)
      if "answer" in chunk:
        if chunk["answer"] == "" and receiving_answer:
            # Stream has ended if we are receiving answers and an empty answer appears
            source = ''
            for item in metadata:
              source = '- ' + item.get("source", "") + '\n\n'
            yield partial_message + '\n\n' + source
        elif chunk["answer"] != "":
            receiving_answer = True
            partial_message = partial_message + chunk["answer"]
            yield partial_message
  except StopIteration:
    print("The stream has ended.")