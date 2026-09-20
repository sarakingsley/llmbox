# Python API Documentation for `src/modes.py`

**Defined Classes:**

## `Modes`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
- `__init__(self)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_make_dataloader(self, cfg)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_chat(self, cfg) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_generate(self, cfg) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_tool_calling(self, cfg) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_structured_output(self, cfg) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_json_or_jsonl(path) -> list`
    - Load a .json or .jsonl file, whichever it actually is.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_prepare_data(self, cfg) -> None`
    - Prepare a dataset for LLMBox jobs: format/standardize, optionally add special tokens/prefixes/suffixes,
split into train/test, tokenize and report readiness for downstream tasks. The config (cfg) should supply
the necessary arguments to indicate: (1) where the input dataset is; (2) where to write outputs; (3) columns; (4) etc.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_build_optimizer(self, cfg, model)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_run_training(self, cfg, allow_lora) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_train(self, cfg) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_finetune(self, cfg) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_evaluation(self, cfg) -> None`
    - Run LLM evaluation (distinct mode). Calls src.evaluator functions.
This mode is SEPARATE from train/finetune and never runs training_metrics.
Walks the user through EVALUATION tasks ONE BY ONE interactively.
    - [Question] How would you use this method in a real project? What side effects might it have?
