import csv    # CSV: standard library only.
import json
import os
import sys
import inspect
import pandas as pd
import huggingface_hub
import datasets

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_services import (
    DataTransformer,
    TokenizedChatDataset,
    CausalLMCollator,
    TrainingDataLoader

)

datatransformer = DataTransformer()

training_data = TrainingDataLoader(
    assistant_mask_strategy="template",
)
