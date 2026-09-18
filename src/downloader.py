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

class ModelDownloader:

    def __init__(self) -> None:
        pass

    @staticmethod
    def downloader(tokenstring: str = "HF_TOKEN", repo: str = "meta-llama/Llama-3.1-8B-Instruct", localdir: str = "./models/llms/meta/Llama-3.1-8B-Instruct"):
        # Load environment variables from .env file
        load_dotenv()
        hf_token = os.getenv(tokenstring)
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file. Add HF_TOKEN=hf_xxxxxxxx to your .env")
        # Authenticate with Hugging Face
        login(token=hf_token)
        # Download AI Model to local device:
        snapshot_download(
            repo_id=repo,
            local_dir=localdir,
            token=hf_token
        )
