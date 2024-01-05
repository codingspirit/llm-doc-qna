# @file agent.py
# @author Alex.D.Scofield (lizhiqin46783937@live.com)
# @date 2023-12-27
# @copyright Copyright (c) 2023

from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from pathlib import Path
import os
import sys

embeddings = OpenAIEmbeddings()

doc_path = input("Please input doc path \n")
db_name = Path(doc_path).stem + "_vdb"
db_path = sys.path[0] + "/" + db_name

if not os.path.exists(db_path):
    print("No existing VDB found. Reading the doc...")
    loader = TextLoader(doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_path)
    print("Finished")
else:
    print("Existing VDB found.")
    db = FAISS.load_local(db_path, embeddings)

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0,
)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

while True:
    query = input('\nType your query, or type "exit" to terminate: \n')
    if query == "exit":
        break
    print("\nthinking...\n")
    qa.run(query)
    print("\n")
