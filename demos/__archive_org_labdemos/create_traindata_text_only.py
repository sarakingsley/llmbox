'''

python3 -m demos.create_traindata_text_only

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
    {"document": "Photosynthesis converts light energy into chemical energy."},
    {"document": "Mitochondria produce much of a cell's ATP."},
]

examples = list(
    datautility.standardize_llm_dataset(
        rows,
        text_columns="document",
    )
)

print(examples)
