from typing import Iterable
from langchain.docstore.document import Document
import json
import shutil
import tempfile
import os

def load_docs_from_jsonl(file_path) -> Iterable[Document]:
    docs = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            docs.append(obj)
    return docs

def save_docs_to_jsonl(array:Iterable[Document], file_path:str, mode:str='a') -> None:
    with open(file_path, mode=mode) as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def copy_to_temp_folder(source_folder:str):
    # create a temporary folder
    temp_folder = tempfile.mkdtemp()

    # copy files to temporary folder
    for item in os.listdir(source_folder):
        s = os.path.join(source_folder, item)
        d = os.path.join(temp_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    return temp_folder

def delete_folder(folder:str):
    try:
        # try delete folder
        shutil.rmtree(folder)

    except Exception as ex: 
        print(f"Temporaray folder could not be deleted: {ex}. The folder has to be deleted manually!")    
