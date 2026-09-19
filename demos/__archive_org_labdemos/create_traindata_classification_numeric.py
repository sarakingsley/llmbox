'''

python3 -m demos.create_traindata_classification_numeric

'''


import os
import sys
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_services import DataTransformer


datautility = DataTransformer()

rows = [
    {"review": "The battery lasts all day.", "label": 1, "id": 101},
    {"review": "It stopped working immediately.", "label": 0, "id": 102},
]

examples = list(
    datautility.standardize_llm_dataset(
        rows,
        text_columns="review",
        target_columns="label",
        instruction=(
            "Classify the review sentiment. "
            "Respond with exactly positive or negative."
        ),
        label_maps={
            "label": {
                0: "negative",
                1: "positive",
            }
        },
    )
)
print(examples[0])
