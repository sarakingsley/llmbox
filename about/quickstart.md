# Add LLM Models to the application:
To use LLMBox, you must create a `models` directory and download LLM model checkpoints into that directory.

* To download LLM model checkpoints from HuggingFace, please run: `python3 downloader.py` (You'll need to change the HF repo specified in this PY file first).

## Steps to add LLMs:
 * Create `models/` directory in the main application directory.
 * Download open-source LLM checkpoint files.
    * You can use `downloader.py` to download open-source LLMs from HuggingFace.
 * Insert the model checkpoint files into the `models/` directory.
 * **Update configurator**: in the `conf/` directory, create a new `.yaml` file for your LLM model. Depending on the LLM model architecture, you may need to make additional changes to the LLMBox application code. 
 * As of September 17, 2026, LLBBox is designed to work with `Microsoft's Phi-4-mini-instruct model` and `Google's Gemma-3-270m model`. Future versions will work with `Meta's Llama models` and `Cohere's Tiny Aya Global model`.

 # Start running LLMs
 To start working with an LLM, in your terminal, run this script: `python3 startllm.py`

 # Interaction Modes
 LLMBox provides a few ways to work with or interact with large language models, and these include:

## **Mode**:
 * `chat`: this mode starts an interactive chat session with an LLM (similar to using a chatbot web interface).
 * `generate`: this mode a single-turn interaction with the LLM, e.g. one promit in and one AI response out. It does not store a chat or prompt/response history.
 * `tool_calling`: this mode facilitates passing a `tool/function` into a single-turn generation.
 * `structured_output`: this mode facilitates passing a `pydantic model` into a single-turn generation.
 * `prepare_data`: this enables a user to automatically transform a dataset into a format accepted by LLMBox's train and finetune modes.
 * `train`: this mode initiates a training job.
 * `finetune`: this mode initiates a LoRA finetune job.
 * `evaluate`: this mode initiates a human and LLM Judge evaluation session. 

## **Example Python Scripts**:
 
### **chat**: 

```
python3 startllm.py \
model=phi4_instruct \
mode=chat \
model.source=local \
model.local_path=./models/llms/microsoft/phi-4-mini-instruct
```

###  **generate**: 

```
python3 startllm.py \
model=phi4_instruct \
mode=generate \
model.source=local \
model.local_path=./models/llms/microsoft/phi-4-mini-instruct \
prompt="Explain the tides"
```
 
### **tool_calling**: 

**Gemma**

```
python3 startllm.py \
  model=gemma3_270m \
  mode=tool_calling tool_calling=weather \
  prompt="What is the weather in Tokyo?" \
  model.source=local \
  model.local_path=/Path/to/local/models/llm/checkpoint/directory/gemma-3-270m-it
```

**Phi4**

```
python3 startllm.py \
  model=phi4_instruct \
  mode=tool_calling tool_calling=weather \
  prompt="What is the weather in Tokyo?" \
  model.source=local \
  model.local_path=/Path/to/local/models/llm/checkpoint/directory/phi-4-mini-instruct
```
 
### **structured_output**: 

```
python3 startllm.py \
model=gemma3_270m \
model.source=local \
model.local_path=./models/llms/google/gemma-3-270m-it \
mode=structured_output structured_output.strict=false \
system_prompt="For each name in the user prompt use the structuredoutput format to print the name appropriately but make sure both names have a record" \
prompt="John Smith and Bob Whatever are going to the party" \
structured_output.enabled=true \
structured_output.schema_path=./datasets/stroutjson/structuredoutput.json
```

### **prepare_data**: 

```
python3 -m startllm \
mode=prepare_data \
data.type=jsonl \
data.path=/path/to/data/transformed_datasets/lsatexample.jsonl
```


### **train**: 

```
python3 startllm.py \
model=gemma3_270m \
model.source=local \
model.local_path=./models/llms/google/gemma-3-270m-it \
mode=train training.enabled=true \
data.path=<TRAIN DATA PATH>
```

### **finetune**: 

**Gemma:**

  ```
  HF_HUB_OFFLINE=1 python3 -m startllm mode=finetune \
  training.enabled=true training.method=lora \
  data.type=jsonl \
  data.path=/Path/to/train/data/file/train.jsonl \
  model.source=local \
  model.local_path=/Path/to/local/models/llm/checkpoint/directory \
  training.output_dir=./finetuned \
  training.grad_accum_steps=1 training.logging_steps=1
  ```
### **evaluate**: 

```
python3 -m startllm mode=evaluate
```
