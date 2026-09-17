'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


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
