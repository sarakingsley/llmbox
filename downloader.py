import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import huggingface_hub as hf
from huggingface_hub import (
    snapshot_download,
    HfApi,
    login)

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file. Add HF_TOKEN=hf_xxxxxxxx to your .env")

# Authenticate with Hugging Face
login(token=hf_token)

# Download AI Model to local device:
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./models/llms/meta/Llama-3.1-8B-Instruct",
    token=hf_token
)
