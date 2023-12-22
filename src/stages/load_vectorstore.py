"""
This script is responsible for loading a Chroma vector database.
A similarity query can be carried out on this database.

Usage: script_name.py vector_database_filename(input) embedder_filename(input) query(input)
"""

from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import argparse
from dill import load
import time
import gc
from utils.data_helpers import copy_to_temp_folder, delete_folder

# env vars
load_dotenv()

# add positional arguments
parser = argparse.ArgumentParser()

parser.add_argument('vector_database_filename')
parser.add_argument('embedder_filename')
parser.add_argument('query')

# optional strategy argument
parser.add_argument('--strategy', default='similarity', choices=['similarity', 'selfquery'])

args = parser.parse_args()

# Access the arguments
vector_database_filename = args.vector_database_filename
embedder_filename = args.embedder_filename
query = args.query
strategy = args.strategy

# load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

# make a copy of the chroma folder
temp = copy_to_temp_folder(vector_database_filename)

# load chroma db from temporary folder
vectorstore = Chroma(persist_directory=temp, embedding_function=embedder)

if strategy == 'similarity':
    # query
    docs = vectorstore.similarity_search(query)

    print(f"Found {len(docs)} documents")
    for doc in docs:
        print(doc.metadata, doc.page_content)

elif strategy == 'selfquery':
    # advanced search
    from langchain.chains.query_constructor.base import AttributeInfo
    from langchain.llms.openai import OpenAI
    from langchain.retrievers.self_query.base import SelfQueryRetriever

    # Define our metadata
    metadata_field_info = [
        AttributeInfo(
            name="Modul",
            description="Name des Modul/Kurs/Space, auf den sich das Dokument bezieht",
            type="string",
        ),
        AttributeInfo(
            name="Modulkuerzel",
            description="3 bis 4-stellige Kurzbezeichnung des Modul/Kurs/Space, um das es in dem Dokument geht",
            type="string",
        )
    ]
    document_content_description = "Informationen aus Spaces, der Lernplattform des Studiengang Bachelor of Data Science an der FHNW"

    # Define self query retriever
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, verbose=True
    )

    docs = retriever.get_relevant_documents(query)

    print(f"Found {len(docs)} documents")
    for doc in docs:
        print(doc.metadata, doc.page_content)

# disconnect from chroma
del vectorstore
gc.collect()

#delete temporary folder
delete_folder(temp)

#wait a sec to avoid simulateous access to files
time.sleep(1)