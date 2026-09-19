'''

python3 -m demos.transform_file_into_traindata

'''

import csv    # CSV: standard library only.
import json
import os
import sys
import pandas as pd
import huggingface_hub
import datasets

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# LLMBOX Module imports:
from src.data_services import DataTransformer

# Initiating LLMBox Module Classes:
datautility = DataTransformer()

# Define the file_path of your Assignment 1 or Final Project Dataset:
ORIGINALDATA = "demos/lsat_style_questions2.csv"

# Transform the dataset into a format accepted by LLMBox:
with open(ORIGINALDATA , encoding="utf-8", newline="") as source:
    transformeddata = list(
        datautility.standardize_llm_dataset(
            csv.DictReader(source),
            text_columns=["StimulusPassage", "QuestionStem"],
            target_columns=["OptionA",
                "OptionB",
                "OptionC",
                "OptionD",
                "OptionE",
                "CorrectAnswer"],
        )
    )

# Save the transformed dataset to JSONL:
datautility.write_jsonl(examples=transformeddata, path="data/transformed_datasets/lsatexample.jsonl")

# Split the transformed dataset into a train and test set:
result = datautility.split_and_save_jsonl(
    transformeddata,
    output_directory=PROJECT_ROOT / "data" / "demos" / "lsat",
    train_fraction=0.8,
    seed=42,
)






######## additional options for working with Pandas DFs and Huggingface datasets:
'''
# Existing pandas DataFrame; pandas is not required by the function.
examples = list(
    datautility.standardize_llm_dataset(
        dataframe.to_dict(orient="records"),
        text_columns="text",
        target_columns="label",
    )
)

# Existing Hugging Face Dataset / IterableDataset.
# Select one split, not the whole DatasetDict.
examples = datautility.standardize_llm_dataset(
    dataset["train"],
    text_columns="question",
    target_columns="answer",
)
'''
