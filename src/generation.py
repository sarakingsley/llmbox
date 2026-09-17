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


    def _load_model_and_tokenizer(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device, dtype = self._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self._resolve_model_path(cfg)
        log.info(
            "Loading '%s' (source=%s, %s) onto %s as %s...",
            cfg.model.name, cfg.model.source, model_path, device, cfg.model.dtype,
        )
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
