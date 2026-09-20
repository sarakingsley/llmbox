# Python API Documentation for `src/generation.py`

**Defined Classes:**

## `GenerationManager`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
- `__init__(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_resolve_device_and_dtype(self, cfg)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_resolve_model_path(self, cfg)`
    - Turn model.source/model_id/local_path into a (path_or_repo_id,
local_files_only) pair for from_pretrained().

source=local uses local_files_only=True so a bad path fails fast and
clearly with a filesystem-style error, instead of transformers quietly
trying to interpret it as a hub repo id. source=huggingface leaves
local_files_only off, so from_pretrained uses its normal cache-or-
download behavior.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_ensure_remote_code_compat(self)`
    - Some trust_remote_code model repos (e.g. Phi-4-mini-instruct's
modeling_phi3.py) still import `LossKwargs` from `transformers.utils`,
a name that newer transformers releases renamed to `TransformersKwargs`.
That break isn't specific to this one model -- it hits a lot of
trust_remote_code repos whose custom code hasn't been updated for the
renamed symbol. Rather than pin an older transformers version (and
lose whatever else changed since), alias the old name back in right
before loading.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_model_and_tokenizer(self, cfg)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_generation_kwargs(self, cfg, tokenizer)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_generate_once(self, model, tokenizer, device, messages, cfg)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_resolve_prompt(self, cfg) -> str`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `build_functools_system_prompt(self, base_system_prompt, tools) -> str`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `parse_tool_calls(self, text)`
    - Look for a 'functools[...]' block and parse it into a list of
{"name": ..., "arguments": {...}} calls. Returns None if no tool
call is present.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run_tool_turn(self, model, tokenizer, device, messages, cfg)`
    - Generate one assistant turn under the functools_prompt convention:
if the model requests tool call(s), execute them locally against
src.tools.TOOL_REGISTRY, feed the results back in, and generate the
final answer. `messages` is mutated in place with the raw response
and any tool result, same as run_turn did in chat_phi4mini.py.
Returns (final_text, tool_call_records).
    - [Question] How would you use this method in a real project? What side effects might it have?
