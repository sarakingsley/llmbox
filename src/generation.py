'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Shared model / generation helpers
# --------------------------------------------------------------------------

class GenerationManager:

    def __init__(self) -> None:
         self.log = logging.getLogger(__name__)

    def _resolve_device_and_dtype(self, cfg):
        import torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[cfg.model.dtype]
        if cfg.model.device != "auto":
            device = cfg.model.device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device, dtype


    def _resolve_model_path(self, cfg):
        """Turn model.source/model_id/local_path into a (path_or_repo_id,
        local_files_only) pair for from_pretrained().

        source=local uses local_files_only=True so a bad path fails fast and
        clearly with a filesystem-style error, instead of transformers quietly
        trying to interpret it as a hub repo id. source=huggingface leaves
        local_files_only off, so from_pretrained uses its normal cache-or-
        download behavior."""
        if cfg.model.source == "local":
            if not cfg.model.local_path:
                raise ValueError(
                    f"model.source=local requires model.local_path to be set for model '{cfg.model.name}'."
                )
            return cfg.model.local_path, True
        if cfg.model.source == "huggingface":
            return cfg.model.model_id, False
        raise ValueError(f"Unknown model.source '{cfg.model.source}'. Use 'huggingface' or 'local'.")


    def _ensure_remote_code_compat(self):
        """Some trust_remote_code model repos (e.g. Phi-4-mini-instruct's
        modeling_phi3.py) still import `LossKwargs` from `transformers.utils`,
        a name that newer transformers releases renamed to `TransformersKwargs`.
        That break isn't specific to this one model -- it hits a lot of
        trust_remote_code repos whose custom code hasn't been updated for the
        renamed symbol. Rather than pin an older transformers version (and
        lose whatever else changed since), alias the old name back in right
        before loading."""
        import transformers.utils as _tu
        if not hasattr(_tu, "LossKwargs") and hasattr(_tu, "TransformersKwargs"):
            _tu.LossKwargs = _tu.TransformersKwargs

    def _load_model_and_tokenizer(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device, dtype = self._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self._resolve_model_path(cfg)
        log.info(
            "Loading '%s' (source=%s, %s) onto %s as %s...",
            cfg.model.name, cfg.model.source, model_path, device, cfg.model.dtype,
        )
        if cfg.model.trust_remote_code:
            self._ensure_remote_code_compat()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=cfg.model.trust_remote_code,  #fixme
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return model, tokenizer, device

    def _generation_kwargs(self, cfg, tokenizer):
        gen = cfg.generation
        return dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=max(gen.temperature, 1e-5),
            top_p=gen.top_p,
            top_k=gen.top_k,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    def _generate_once(self, model, tokenizer, device, messages, cfg, **template_kwargs):
        import torch

        template_kwargs = {**OmegaConf.to_container(cfg.model.chat_template_kwargs, resolve=True), **template_kwargs}
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(**inputs, **self._generation_kwargs(cfg, tokenizer))

        return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    def _resolve_prompt(self, cfg) -> str:
        if cfg.prompt_file:
            text = Path(cfg.prompt_file).read_text(encoding="utf-8").strip()
        elif cfg.prompt:
            text = cfg.prompt.strip()
        else:
            raise ValueError('This mode needs a prompt. Pass prompt="..." or prompt_file=path/to/file.txt')
        if not text:
            raise ValueError("Resolved prompt is empty.")
        return text

    # --------------------------------------------------------------------------
    # "functools_prompt" tool calling -- for models (e.g. Phi-4-mini-instruct)
    # whose chat template doesn't do structured tool calling via a `tools=`
    # kwarg, and instead expects tool definitions embedded in the system
    # prompt as <|tool|>[...]<|/tool|>, with the model replying with a
    # "functools[...]"-prefixed JSON list of calls when it wants to invoke
    # one. Mirrors the convention chat_phi4mini.py established.
    # --------------------------------------------------------------------------

    def build_functools_system_prompt(self, base_system_prompt: str, tools: list) -> str:
        parts = [base_system_prompt] if base_system_prompt else []
        tools_json = json.dumps(tools)
        parts.append(f"<|tool|>{tools_json}<|/tool|>")
        parts.append(
            "In addition to plain text responses, you can choose to call one "
            "or more of the provided functions.\n"
            "Use the following rule to decide when to call a function:\n"
            "* if the response can be generated from your internal knowledge, "
            "do so\n"
            "* if you need external information that can be obtained by "
            "calling one or more of the provided functions, generate "
            "function calls\n"
            "If you decide to call functions:\n"
            "* prefix function calls with the functools marker (no closing "
            "marker required)\n"
            "* all function calls should be generated in a single JSON list "
            'formatted as functools[{"name": [function name], "arguments": '
            "[function arguments as JSON]}, ...]\n"
            "* follow the provided JSON schema. Do not hallucinate arguments "
            "or values.\n"
            "* respect the argument type formatting."
        )
        return "\n".join(parts)

    def parse_tool_calls(self, text: str):
        """Look for a 'functools[...]' block and parse it into a list of
        {"name": ..., "arguments": {...}} calls. Returns None if no tool
        call is present."""
        marker = "functools"
        idx = text.find(marker)
        if idx == -1:
            return None
        json_str = text[idx + len(marker):].strip()
        try:
            calls, _ = json.JSONDecoder().raw_decode(json_str)
            return calls
        except json.JSONDecodeError:
            return None

    def run_tool_turn(self, model, tokenizer, device, messages, cfg):
        """Generate one assistant turn under the functools_prompt convention:
        if the model requests tool call(s), execute them locally against
        src.tools.TOOL_REGISTRY, feed the results back in, and generate the
        final answer. `messages` is mutated in place with the raw response
        and any tool result, same as run_turn did in chat_phi4mini.py.
        Returns (final_text, tool_call_records)."""
        from src.tools import TOOL_REGISTRY

        raw_response = self._generate_once(model, tokenizer, device, messages, cfg)
        tool_call_records = []

        tool_calls = self.parse_tool_calls(raw_response)
        if tool_calls:
            results = []
            for call in tool_calls:
                name = call.get("name")
                arguments = call.get("arguments", {})
                func = TOOL_REGISTRY.get(name)
                if func is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = func(**arguments)
                    except Exception as exc:  # noqa: BLE001 - surface bad args to the model
                        result = {"error": str(exc)}
                results.append({"name": name, "result": result})
                tool_call_records.append({"name": name, "arguments": arguments, "result": result})

            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "tool", "content": json.dumps(results)})

            raw_response = self._generate_once(model, tokenizer, device, messages, cfg)

        messages.append({"role": "assistant", "content": raw_response})
        return raw_response, tool_call_records
