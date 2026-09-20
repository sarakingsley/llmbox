# Python API Documentation for `src/schema.py`

**Top-level Functions:**

### `register_configs`
Signature: `register_configs() -> None`
Docstring: Register the schema so conf/config.yaml's `- config_schema` defaults
entry can pull it in for validation.

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.


**Defined Classes:**

## `ModelConfig`
Docstring: Which model to load and how. One YAML file per model under conf/model/.

`source` picks where the weights actually come from:
  - "huggingface": `model_id` is a hub repo id (e.g. "google/gemma-3-270m-it").
    transformers will use a local cache if one exists and download
    otherwise -- normal from_pretrained behavior.
  - "local": `local_path` is a path to a directory already on disk
    (a `huggingface-cli download ... --local-dir` checkout, or any
    folder with the usual config/tokenizer/weight files in it).
    Loaded with local_files_only=True, so this never touches the
    network and fails fast with a clear error if the path is wrong,
    rather than transformers silently trying to interpret it as a
    hub repo id.

Methods:
## `GenerationConfig`
Docstring: Sampling / decoding hyperparameters. Used by every inference mode
(chat, generate, tool_calling, structured_output).

Methods:
## `ToolDef`
Docstring: One callable tool, described the way chat templates that support
function calling generally expect (JSON-schema-style parameters).

Methods:
## `ToolCallingConfig`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
## `StructuredOutputConfig`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
## `OptimizerConfig`
Docstring: One YAML file per optimizer under conf/optimizer/.

Methods:
## `TrainingConfig`
Docstring: Shared by both the 'train' (continued pretraining, full weights only)
and 'finetune' (parameter-efficient or full) modes.
New: method supports additional techniques beyond LoRA.

Methods:
## `DataConfig`
Docstring: Where training/finetuning data comes from. One YAML file per source
type under conf/data/.
For prepare_data mode, allow additional attributes for raw/original path,
text/target columns, label maps, instruction, output_dir, etc.

Methods:
## `ModeConfig`
Docstring: Which of the seven run modes to execute. One YAML file per mode under
conf/mode/.
Now supports prepare_data in addition to chat, generate, tool_calling, structured_output, train, finetune.

Methods:
## `TrainingMetrics`
Docstring: Track and record metrics and resource use for a training job.

Methods:
- `tokens_remaining(self) -> int`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `Config`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
