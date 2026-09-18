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

from src.data_utilities import DataUtility

datautility = DataUtility()

TRAINDATA = "demos/lsat_style_questions2.csv"

with open(TRAINDATA, encoding="utf-8", newline="") as source:
    examples = list(
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
print("FUCK YOU")

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
