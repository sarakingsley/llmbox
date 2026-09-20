# Python API Documentation for `src/data_services.py`

**Top-level Functions:**

### `_utc_now_iso`
Signature: `_utc_now_iso() -> str`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.


**Defined Classes:**

## `DataTransformer`
Docstring: Prepare row-oriented datasets for causal language-model training.
Every standardized example contains:
    prompt:
        The input context. Empty for text-only language modeling.
    completion:
        The desired continuation or response.
    text:
        prompt + completion, suitable for full-sequence causal-LM training.
    messages:
        A chat representation suitable for chat-template-based fine-tuning.
This class does not tokenize data or create loss masks. Those operations
belong in the training pipeline.

Methods:
- `_json_text(value) -> str`
    - Serialize a Python value as strict JSON.
ensure_ascii=False preserves characters such as accented letters.
allow_nan=False rejects NaN and Infinity, which are not valid JSON.
Unsupported values, such as sets or tensors, must be converted by
the caller before using this utility.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_render_value(value) -> str`
    - Convert a dataset value into model-visible text.
Strings are returned unchanged. In particular, a string that already
contains JSON must not be JSON-encoded a second time.
Other values use JSON formatting:
    True -> "true"
    42 -> "42"
    ["a", "b"] -> '["a", "b"]'
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_normalize_columns(columns, argument_name) -> list[str]`
    - Normalize one column name or a sequence of names into a list.
Reject empty names and duplicates instead of silently producing
incomplete or ambiguous examples.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_read_value(row, column) -> Any`
    - Read a required, non-null field.
An absent field and a field containing None are both rejected.
Empty strings are checked separately where the field is used.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_validate_prompt_completion(example) -> tuple[str, str]`
    - Validate an already string-valued prompt/completion pair.
An empty prompt is valid for text-only language modeling.
A completion must contain at least one non-whitespace character.
Return the original strings without stripping meaningful whitespace.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `normalize_prompt_completion(example) -> dict[str, Any]`
    - Add training representations to a prompt/completion example.
Both fields are required. Unlike a schema-detection function, this
function rejects unrelated records rather than silently passing
them through.
A completion may be:
    - a string, preserved exactly;
    - a dictionary or list, serialized once as JSON.
Other fields are preserved. Existing text/messages fields are
regenerated so they agree with prompt/completion.
No separator is inserted between prompt and completion. The prompt
is responsible for its own formatting, such as a final "Output:\n".
    - [Question] How would you use this method in a real project? What side effects might it have?

- `standardize_llm_dataset(self, dataset) -> Iterator[dict[str, Any]]`
    - Convert rows into prompt/completion/text/messages examples.

Mode 1: Existing prompt/completion pairs
    Leave text_columns and target_columns unset.

    The existing prompt is preserved exactly. This avoids wrapping
    an already formatted prompt in another "Input:/Output:" block.

Mode 2: Text-only language modeling
    Set text_columns; leave target_columns unset.

    Selected text becomes the completion, and the prompt is empty.
    One text column is preserved as-is. Multiple text columns are
    combined using field names and blank-line separators.

Mode 3: Supervised generation
    Set both text_columns and target_columns.

    Input fields become a formatted prompt. One target becomes a
    text/scalar completion. Multiple targets become a JSON object.

label_maps:
    Optional mappings for selected target columns. For example:
        {"label": {0: "negative", 1: "positive"}}
    Every encountered label must appear in its mapping.

Important behavior:
    - Accepts lists, generators, and individual HF Dataset splits.
    - Rejects a single row dict or a dict of dataset splits.
    - Returns only prompt/completion/text/messages.
    - Does not modify the input rows.
    - Row errors include a zero-based row index.
    - Validation is deferred until this iterator is consumed.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_serialize_examples(examples) -> list[str]`
    - Validate and serialize all rows before opening destination files.
This prevents a malformed example from truncating an existing file.
It deliberately uses memory proportional to the serialized dataset.
Returned strings already contain their JSONL newline.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `write_jsonl(examples, path) -> int`
    - Write JSON-serializable mappings, one per line.
Nested fields such as messages are supported.
Parent directories are created if necessary.
All examples are serialized before opening the output file.
This is not an atomic write: an I/O failure can leave a partial file.
Return the number of examples written.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `split_and_save_jsonl(examples, output_directory) -> dict[str, Any]`
    - Shuffle examples and write train.jsonl and test.jsonl.
Requirements:
    - At least two examples.
    - String-valued prompt and non-empty completion fields.
    - JSON-serializable values in all retained fields.
Behavior:
    - Preserves all example fields, including text and messages.
    - Uses a local RNG, leaving global random state unchanged.
    - Keeps at least one example in each split.
    - Loads serialized examples into memory.
    - Validates all examples before opening output files.
Limitations:
    - This is a row-level random split, not a grouped or stratified split.
    - Writes are not atomic, either individually or as a pair.
      An I/O failure can leave incomplete output.
    - [Question] How would you use this method in a real project? What side effects might it have?

## `TokenizedChatDataset`
Docstring: Store conversations that have already been tokenized.

Each example contains equally sized lists:
    input_ids: tokens visible to the model
    labels: target token IDs, or IGNORE_INDEX for unsupervised positions

Padding is postponed until examples are assembled into a batch.

Methods:
- `__init__(self, examples) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `__len__(self) -> int`
    - Return the number of usable training conversations.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `__getitem__(self, index) -> dict[str, list[int]]`
    - Return one unpadded example.
    - [Question] How would you use this method in a real project? What side effects might it have?

## `CausalLMCollator`
Docstring: Right-pad examples to the longest sequence in the current batch.

Padding rules:
    input_ids      -> pad_token_id
    labels         -> IGNORE_INDEX
    attention_mask -> 0

Real tokens receive attention_mask=1, including prompt tokens whose
labels are ignored. Ignoring a label must not hide its input context.

A module-level callable class is used instead of a nested function so
the collator can be pickled by multiprocessing DataLoader workers.

Methods:
- `__init__(self, pad_token_id, ignore_index) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `__call__(self, batch) -> dict[str, torch.Tensor]`
    - Validate, pad, and convert one batch into PyTorch tensors.
    - [Question] How would you use this method in a real project? What side effects might it have?

## `TrainingDataLoader`
Docstring: Load chat records and prepare assistant-only causal-LM training data.
assistant_mask_strategy:
    "template":
        Use assistant token masks returned by the tokenizer's chat
        template. The template must mark assistant output with
        {% generation %} ... {% endgeneration %} blocks.
    "verified_prefix":
        Locate assistant responses using tokenized conversation
        prefixes. Every prefix must exactly match the corresponding
        beginning of the full token sequence.
        This works only for compatible chat templates. Incompatible
        templates raise an error instead of receiving guessed masks.
This class expects text-only system/user/assistant messages.
Tool calls and multimodal messages require additional handling.

Methods:
- `__init__(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_validate_messages(messages) -> list[dict[str, str]]`
    - Validate and copy a text-only conversation.
Require at least one non-empty assistant response because this
loader constructs assistant-only training targets.
Role alternation is not enforced here: individual chat templates
may impose additional conversation rules during tokenization.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_record_to_conversation(self, record) -> dict[str, list[dict[str, str]]]`
    - Convert a JSONL record into the common conversation format.

Prefer messages when explicitly present. Otherwise, normalize
prompt/completion using DataTransformer from this same module.

An invalid messages field is not silently replaced by other fields.
Text-only records are not implicitly treated as instruction data.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_conversations(self, cfg) -> list[dict[str, list[dict[str, str]]]]`
    - Dispatch using the program's attribute-based configuration.
Expected fields:
    cfg.data.type: "chat_log" or "jsonl"
    cfg.data.path: file or directory path
    cfg.data.min_turns: optional, defaults to 1 for chat logs
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_conversations_from_chat_log(self, chat_log_dir, min_turns) -> list[dict[str, list[dict[str, str]]]]`
    - Load one conversation per JSON chat-log file.
Expected log structure:
    {
        "session": {
            "settings": {"system_prompt": "..."}
        },
        "turns": [
            {
                "user": {"content": "..."},
                "assistant": {"content": "..."}
            }
        ]
    }
Incomplete turns are skipped. min_turns counts complete retained
user/assistant pairs, not the number of raw turn records.
Missing paths raise errors. Malformed individual files are logged
and skipped.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_conversations_from_jsonl(self, dataset_path) -> list[dict[str, list[dict[str, str]]]]`
    - Read messages or prompt/completion records from a JSONL file.
Blank lines are ignored. Malformed records are logged and skipped.
Errors opening or reading the file propagate to the caller.
Report physical, one-based line numbers for easy file inspection.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_validate_token_ids(token_ids) -> list[int]`
    - Require a flat, unbatched list of token IDs.
These methods deliberately do not request tensors. Padding and
conversion to tensors happen later in the collator.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_tokenize_with_template_mask(self, tokenizer, messages) -> tuple[list[int], list[int]]`
    - Use assistant regions explicitly marked by the chat template.
This is preferable to inferring token boundaries from string
lengths or separately tokenized message content.
Requires a tokenizer/template supporting:
    return_dict=True
    return_assistant_tokens_mask=True
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_tokenize_with_verified_prefixes(self, tokenizer, messages) -> tuple[list[int], list[int]]`
    - Infer assistant spans only when token prefixes are exact matches.
For each assistant turn:
    start = length of prior context + assistant generation header
    end   = length of conversation through that assistant turn
The assistant header is excluded from loss. Template suffixes
following the answer, such as end-of-turn tokens, are included.
This strategy is intentionally conservative. Templates that change
earlier text depending on later messages are rejected.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_build_labeled_example(self, tokenizer, messages, max_length) -> tuple[list[int], list[int]]`
    - Tokenize one conversation and keep loss only on assistant positions.

Right truncation keeps the first max_length tokens. It may remove
all assistant targets; _build_dataset detects and skips that case.

Labels remain aligned with input_ids. Hugging Face causal-LM models
normally perform the next-token shift internally.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_build_dataset(self, conversations, tokenizer, max_length) -> TokenizedChatDataset`
    - Eagerly tokenize conversations and return an unpadded Dataset.
Conversations without surviving trainable tokens are skipped.
Tokenization/template errors are fatal: silently skipping them
could hide a systematic masking bug.
Memory use is proportional to the complete tokenized dataset.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_make_collator(self, pad_token_id) -> CausalLMCollator`
    - Create a reusable right-padding collator.
    - [Question] How would you use this method in a real project? What side effects might it have?
