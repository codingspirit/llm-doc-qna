# @file agent.py
# @author Alex.D.Scofield (lizhiqin46783937@live.com)
# @date 2023-12-27
# @copyright Copyright (c) 2023

from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from pathlib import Path
import os
import sys

sys.stdin.reconfigure(encoding="utf-8")

model_name = "gpt-3.5-turbo"
# model_name = "anthropic.claude-v2"

if model_name == "anthropic.claude-v2":
    embeddings = BedrockEmbeddings()
    llm = BedrockChat(
        model_id=model_name,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={
            "temperature": 0.0,
            "max_tokens_to_sample": 4096,
        },
    )

else:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(
        model_name=model_name,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
    )

doc_path = input("Please input doc path \n")
doc_path = Path(doc_path)
doc_extension = doc_path.suffix
db_name = doc_path.stem + "_vdb"
db_path = sys.path[0] + "/" + model_name + "-vdb/" + db_name

if not os.path.exists(db_path):
    print("No existing VDB found. Reading the doc...")
    if doc_extension == ".txt":
        loader = TextLoader(doc_path)
    elif doc_extension == ".md":
        loader = UnstructuredMarkdownLoader(doc_path)
    elif doc_extension == ".pdf":
        loader = UnstructuredPDFLoader(doc_path)
    else:
        raise ValueError(f"Unsupported file type {doc_extension}")

    documents = loader.load()
    if doc_extension == ".md":
        text_splitter = MarkdownTextSplitter(chunk_size=2048, chunk_overlap=0)
    else:
        text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)

    text_chunks = text_splitter.split_documents(documents)
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(db_path)
    print("Finished")
else:
    print("Existing VDB found.")
    db = FAISS.load_local(db_path, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
)

while True:
    query = input('\nType your query, or type "exit" to terminate: \n\n')
    if query == "exit":
        break
    print("\nthinking...\n")
    qa.invoke(query)
    print("\n")
