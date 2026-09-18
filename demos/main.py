'''

python3 -m demos.main

MODES: CHAT:
python3 startllm.py model=phi4_instruct mode=chat model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/microsoft/phi-4-mini-instruct
python3 startllm.py model=gemma3_270m mode=chat model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/google/gemma-3-270m-it

MODES: GENERATE:
'''

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

from src.data_utilities import DataUtility
from src.downloader import ModelDownloader

datautility = DataUtility()
modeldownloader = ModelDownloader()

TRAINDATA = "demos/lsat_style_questions2.csv"
example_df = pd.read_csv(TRAINDATA)
example_df = pd.DataFrame(example_df)

# CSV FILE EXAMPLE
print("---- CSV FILE DEMO -----------")
print("   ")
startdemo = input("Start Demo? yes/no: ")
if startdemo == "yes":
    print("Demo CSV file columns:")
    print(example_df.columns)
    print("   ")
else: pass
proceed01 = input("Proceed? yes/no: ")
if proceed01 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("----STEPS:")
    print("   ")
    print("1. Define an object pointing to your dataset CSV file:")
    print("   ")
    print("TRAINDATA = demos/lsat_style_questions2.csv")
    print("   ")
proceed02 = input("Proceed? yes/no: ")
if proceed02 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("2. Use the 'standardize_llm_dataset function to format the dataset:")
    print("   ")
    std_llm_dataset_func = inspect.getsource(datautility.standardize_llm_dataset)
    print(std_llm_dataset_func)
    print("   ")
    print("FORMATTING DEMO DATASET")
else: pass
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
proceed03 = input("Proceed? yes/no: ")
if proceed03 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("--------INPUT-------------")
    print(examples[0]['prompt'])
    print("--------TARGETS-------------")
    print(examples[0]['completion'])
    print("   ")
else: pass
proceed04 = input("Proceed? yes/no: ")
if proceed04 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("3. Save the dataset as a JSONL file:")
    print("   ")
    print("creating JSONL file from examples")
else: pass
datautility.write_jsonl(examples, path="data/traindata/trainexample.json")
proceed05 = input("Proceed? yes/no: ")
if proceed05 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--------SAVING JSONL FILE TO DIRECTORY-------------")
else: pass
proceed06 = input("Proceed? yes/no: ")
result = datautility.split_and_save_jsonl(
    examples,
    output_directory=PROJECT_ROOT / "data" / "sentiment",
    train_fraction=0.8,
    seed=42,
)
if proceed06 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("4. Split the dataset into a train and test set:")
    print("   ")
    print(result)
else: pass
proceed07 = input("Proceed? yes/no: ")
if proceed07 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("--- How to start working with an LLM? ---------------")
    print("   ")
    print("---- LLM DEMO -----------")
    print("   ")
    print("---- Steps -----------")
    print("   ")
    print("1. Download an LLM from Canvas or Huggingface  ")
    print("   ")
else: pass
proceed08 = input("Proceed? yes/no: ")
if proceed08 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("--- HuggingFace ---------------")
    print("   ")
    print("Use downloader.py file to download LLMs from HuggingFace:")
    downloader_func = inspect.getsource(modeldownloader.downloader)
    print(downloader_func)
else: pass
proceed09 = input("Proceed? yes/no: ")
if proceed09 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("--- CANVAS ---------------")
    print("   ")
else: pass
proceed10 = input("Proceed? yes/no: ")
if proceed10 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("2. Create a 'model' directory within LLMBox app")
    print("   ")
else: pass
proceed11 = input("Proceed? yes/no: ")
if proceed11 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("3. Choose your MODE. Modes include:")
    print("   ")
else: pass
proceed12 = input("Proceed? yes/no: ")
if proceed12 == "yes":
    print("Chat")
    print("   ")
else: pass
proceed13 = input("Proceed? yes/no: ")
if proceed13 == "yes":
    print("Generate")
    print("   ")
else: pass
proceed14 = input("Proceed? yes/no: ")
if proceed14 == "yes":
    print("Generate + structured_output")
    print("   ")
else: pass
proceed15 = input("Proceed? yes/no: ")
if proceed15 == "yes":
    print("Train")
    print("   ")
else: pass
proceed16 = input("Proceed? yes/no: ")
if proceed16 == "yes":
    print("Finetune")
    print("   ")
else: pass
proceed17 = input("Proceed? yes/no: ")
if proceed17 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("3. Run the appropriate python script for your mode.")
    print("   ")
else: pass
proceed17 = input("Proceed? yes/no: ")
if proceed17 == "yes":
    print("   ")
    print("CHAT EXAMPLE 2: python3 startllm.py model=phi4_instruct mode=chat model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/microsoft/phi-4-mini-instruct")
    print("   ")
    print("   ")
else: pass
proceed18 = input("Proceed? yes/no: ")
if proceed18 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("CHAT EXAMPLE 2: python3 startllm.py model=gemma3_270m mode=chat model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/google/gemma-3-270m-it")
    print("   ")
    print("   ")
else: pass
proceed19 = input("Proceed? yes/no: ")
if proceed19 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print("GENERATE EXAMPLE 1: python3 startllm.py model=phi4_instruct mode=generate model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/microsoft/phi-4-mini-instruct prompt='Explain the tides'")
    print("   ")
    print("   ")
else: pass
proceed20 = input("Proceed? yes/no: ")
if proceed20 == "yes":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   ")
    print('''GENERATE + Structured_output: python3 startllm.py model=gemma3_270m model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/google/gemma-3-270m-it mode=structured_output structured_output.strict=false system_prompt="For each name in the user prompt use the structuredoutput format to print the name appropriately but make sure both names have a record" prompt="John Smith and Bob Whatever are going to the party" structured_output.enabled=true structured_output.schema_path=/Users/skingsle/Documents/llmbox/llm_box/data/pydantic_models/structuredoutput.json''')
    print("   ")
    print("   ")
else: pass
