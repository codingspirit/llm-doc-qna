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

from pathlib import Path
import os

re_embedding = True

embeddings = OpenAIEmbeddings()

doc_path = input("Please input doc path \n")
base_path = os.path.dirname(doc_path)
db_name = Path(doc_path).stem


if not os.path.exists(base_path + db_name):
    print("No existing VDB found. Reading the doc...")
    loader = TextLoader(doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(base_path + db_name)
    print("Finished")
else:
    print("Existing VDB found.")
    db = FAISS.load_local(base_path + db_name, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), chain_type="stuff", retriever=db.as_retriever()
)

while True:
    query = input('Type your query, or type "exit" to terminate: \n')
    if query == "exit":
        break
    result = qa.run(query)
    print(result)
