'''

python3 -m demos.create_traindata_multi_input_targets

'''

import os
import sys
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utilities import DataUtility


datautility = DataUtility()

'''
rows = [
    {
        "title": "Package arrived damaged",
        "message": "The screen is cracked.",
        "category": "returns",
        "urgent": True,
    }
]

examples = list(
    datautility.standardize_llm_dataset(
        rows,
        text_columns=["title", "message"],
        target_columns=["category", "urgent"],
        instruction=(
            "Categorize the support request. "
            "Return JSON with keys category and urgent."
        ),
    )
)
'''

TRAINDATA = "/Users/skingsle/Documents/llmbox/llm_box/demos/lsat_style_questions.csv"

# StimulusPassage
# QuestionStem
examples = list(
    datautility.standardize_llm_dataset(
        rows,
        text_columns=["StimulusPassage", "QuestionStem"],
        target_columns=["OptionA",
            "OptionB",
            "OptionC",
            "OptionD",
            "OptionE",
            "CorrectAnswer"],
        instruction=(
            "Choose the correct option. "
            "Return JSON with keys CorrectAnswer."
        ),
    )
)

print(examples[0]["completion"])
