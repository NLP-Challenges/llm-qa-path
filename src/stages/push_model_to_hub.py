
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('finetuned_path')
parser.add_argument("repo_name")

args = parser.parse_args()

# Access the arguments
finetuned_path = args.finetuned_path
repo_name = args.repo_name

#load hugging-face token
load_dotenv()
hf_token_write = os.environ["HF_ACCESS_TOKEN_WRITE"]

#push model, config and tokenizer to hub
HfApi().upload_folder(repo_id=repo_name, repo_type="model", folder_path=finetuned_path, token=hf_token_write, commit_message=f"Update of model components from run\n\nLink to WandB run:\nUpload happened isolated from training why link is not available anymore!")