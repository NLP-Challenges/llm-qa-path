"""
This script is responsible for dividing text into text blocks (corpus).

Usage: script_name.py text_filename(input) curpus_filename(output)
"""

import argparse
import os
from typing import Iterable
import time
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import utils as chromautils
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Add positional arguments
parser.add_argument('data_folder_name')
parser.add_argument('corpus_filename')

args = parser.parse_args()

# Access the arguments
data_folder_name = args.data_folder_name
corpus_filename = args.corpus_filename

# Helper functions
def save_docs_to_jsonl(array:Iterable[Document], file_path:str, mode:str='a') -> None:
    with open(file_path, mode=mode) as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

# Document collection
docs = []

# loop over all files in folder_name
for filename in tqdm(os.listdir(data_folder_name), 'Building Corpus'):
    if filename.endswith('.pdf'):
        # Handle PDFs
        loader = PyPDFLoader(f"{data_folder_name}/{filename}")
        new_docs = loader.load_and_split()
        docs.extend(new_docs)
    elif filename == 'spaces.parquet':
        # Handle parquet files
        df_spaces = pd.read_parquet(f"{data_folder_name}/{filename}")

        # Create a document for each row in the dataframe
        for index, space in df_spaces.iterrows():
            metadata = {
                'Modul': space['name'] if space['name'] else 'unknown',
                'Modulkuerzel': space['abbreviation'] if space['abbreviation'] else 'unknown',
                'ECTS': space['ects'] if space['ects'] else 'unknown',
                'Semester': space['semester'] if space['semester'] else 'unknown',
                'Tags': ', '.join(space['tag_values'].tolist()) if len(space['tag_values']) > 0 else 'none',
            }

            headers_to_split_on = [
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]

            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0) # 6.38 avg chars per word in german

            if space['portrait_content']:
                portrait_heading_splits = html_splitter.split_text(space['portrait_content'])
                portrait_splits = text_splitter.split_documents(portrait_heading_splits)
                for portrait_tab in portrait_splits:
                    portrait_tab.metadata = dict(**{ 'Quelle': f'Portrait des {space["name"]} Space' }, **portrait_tab.metadata, **metadata)
                docs.extend(portrait_splits)
            if space['exercise_content']:
                exercise_heading_splits = html_splitter.split_text(space['exercise_content'])
                exercise_splits = text_splitter.split_documents(exercise_heading_splits)
                for exercise_tab in exercise_splits:
                    exercise_tab.metadata = dict(**{ 'Quelle': f'Aufgaben im {space["name"]} Space' }, **exercise_tab.metadata, **metadata)
                docs.extend(exercise_splits)
            if space['sidebar_content']:
                sidebar_splits = html_splitter.split_text(space['sidebar_content'])
                for sidebar_tab in sidebar_splits:
                    sidebar_tab.metadata = dict(**{ 'Quelle': f'Sidebar des {space["name"]} Space' }, **sidebar_tab.metadata, **metadata)
                docs.extend(sidebar_splits)

# save docs to jsonlines file
docs = chromautils.filter_complex_metadata(docs)
save_docs_to_jsonl(docs, corpus_filename, mode='w')

# wait a sec to avoid simulateous access to files
time.sleep(1)
