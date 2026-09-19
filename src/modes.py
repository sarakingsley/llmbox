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

"""
modes.py

Implementations of the six run modes dispatched by configurator.py:
chat, generate, tool_calling, structured_output, train, finetune.

torch/transformers/peft are imported lazily inside each function, so that
composing or printing a config (`python configurator.py --cfg job`) never
requires them to be installed -- only actually running a mode does.
"""

import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from omegaconf import OmegaConf

# import LLMBOX application software:
from src.generation import GenerationManager
from src.datasets import DataManager
from src.sys_logger import Logger


class Modes:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.IGNORE_INDEX = -100
        self.generator = GenerationManager()
        self.datamanager = DataManager()
        self.logger = Logger()

    # --------------------------------------------------------------------------
    # chat -- interactive multi-turn session, logged the same way gemma_chat.py
    # logs sessions (one JSON file per session under <output_dir>/chat_log).
    # --------------------------------------------------------------------------
    def run_chat(self, cfg) -> None:
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        log_dir = Path(cfg.output_dir) / "chat_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        session_started_at =  self.logger._utc_now_iso()
        session_id = uuid.uuid4().hex[:8]
        log_path = log_dir / f"{session_started_at.replace(':', '-')}_{session_id}.json"
        session_record = {
            "session": {
                "session_id": session_id,
                "started_at": session_started_at,
                "model": OmegaConf.to_container(cfg.model, resolve=True),
                "generation": OmegaConf.to_container(cfg.generation, resolve=True),
                "system_prompt": cfg.system_prompt,
            },
            "turns": [],
        }
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        username = cfg.username or "user"
        print(f"[info] Logging this session to: {log_path}", file=sys.stderr)
        print("[info] Type your message and press Enter. Type /exit or Ctrl-D to quit.\n", file=sys.stderr)
        turn_number = 0
        while True:
            try:
                user_content = input(f"{username}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[info] Exiting.", file=sys.stderr)
                break
            if not user_content:
                continue
            if user_content.lower() in {"/exit", "/quit"}:
                break
            messages.append({"role": "user", "content": user_content})
            answer = self.generator._generate_once(model, tokenizer, device, messages, cfg)
            messages.append({"role": "assistant", "content": answer})
            turn_number += 1
            session_record["turns"].append({
                "turn": turn_number,
                "user": {"username": username, "content": user_content, "timestamp": self.logger._utc_now_iso()},
                "assistant": {"content": answer, "timestamp": self.logger._utc_now_iso()},
            })
            log_path.write_text(json.dumps(session_record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"assistant> {answer}\n")
        print(f"[info] Session saved to: {log_path}", file=sys.stderr)

    # --------------------------------------------------------------------------
    # generate -- single prompt in, single response out, no history kept.
    # --------------------------------------------------------------------------
    def run_generate(self, cfg) -> None:
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        prompt = self.generator._resolve_prompt(cfg)
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        print(self.generator._generate_once(model, tokenizer, device, messages, cfg))

    # --------------------------------------------------------------------------
    # tool_calling -- single-turn generation with tool/function definitions
    # passed through the chat template.
    # --------------------------------------------------------------------------
    def run_tool_calling(self, cfg) -> None:
        if not cfg.tool_calling.enabled:
            raise ValueError("mode=tool_calling requires tool_calling.enabled=true")
        if not cfg.model.supports_tool_calling:
            self.log.warning(
                "Model '%s' is not marked as supporting tool calling "
                "(model.supports_tool_calling=false); its chat template may "
                "ignore the 'tools' argument entirely.", cfg.model.name,
            )
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        prompt = self.generator._resolve_prompt(cfg)
        tools = OmegaConf.to_container(cfg.tool_calling.tools, resolve=True)

        if cfg.model.tool_calling_format == "functools_prompt":
            system_content = self.generator.build_functools_system_prompt(cfg.system_prompt, tools)
            messages = [{"role": "system", "content": system_content}]
            messages.append({"role": "user", "content": prompt})

            answer, tool_call_records = self.generator.run_tool_turn(model, tokenizer, device, messages, cfg)
            if tool_call_records:
                print("[info] Tool call(s) made:", file=sys.stderr)
                for record in tool_call_records:
                    print(f"  {record['name']}({record['arguments']}) -> {record['result']}", file=sys.stderr)
            print(answer)
            return

        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})

        answer = self.generator._generate_once(
            model, tokenizer, device, messages, cfg,
            tools=tools, tool_choice=cfg.tool_calling.tool_choice,
        )
        print(answer)

    # --------------------------------------------------------------------------
    # structured_output -- single-turn generation constrained to a JSON schema.
    # --------------------------------------------------------------------------
    def run_structured_output(self, cfg) -> None:
        if not cfg.structured_output.enabled:
            raise ValueError("mode=structured_output requires structured_output.enabled=true")
        if not cfg.model.supports_structured_output:
            self.log.warning(
                "Model '%s' is not marked as supporting structured output "
                "(model.supports_structured_output=false); results may not "
                "reliably follow the schema.", cfg.model.name,
            )

        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        prompt = self.generator._resolve_prompt(cfg)

        schema_instruction = ""
        if cfg.structured_output.schema_path:
            schema_text = Path(cfg.structured_output.schema_path).read_text(encoding="utf-8")
            schema_instruction = (
                "\n\nRespond with ONLY a single JSON object that strictly matches "
                f"this JSON Schema, with no other text:\n{schema_text}"
            )

        system_content = (cfg.system_prompt or "") + schema_instruction
        messages = []
        if system_content.strip():
            messages.append({"role": "system", "content": system_content.strip()})
        messages.append({"role": "user", "content": prompt})

        answer = self.generator._generate_once(model, tokenizer, device, messages, cfg)

        if cfg.structured_output.strict:
            try:
                print(json.dumps(json.loads(answer), indent=2))
                return
            except json.JSONDecodeError as e:
                self.log.warning("Model output was not valid JSON (%s); printing raw output instead.", e)

        print(answer)

    # --------------------------------------------------------------------------
    # train / finetune -- shared SFT trainer. 'train' only allows full weight
    # training (continued pretraining); 'finetune' also allows LoRA.
    # --------------------------------------------------------------------------
    def _build_optimizer(self, cfg, model):
        import torch
        opt = cfg.optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if opt.name == "adamw":
            return torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                    betas=tuple(opt.betas), eps=opt.eps)
        if opt.name == "sgd":
            return torch.optim.SGD(params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                    momentum=opt.momentum)
        if opt.name == "adafactor":
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                            scale_parameter=False, relative_step=False)
        raise ValueError(f"Unknown optimizer.name '{opt.name}'.")


    def _run_training(self, cfg, allow_lora: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        if cfg.training.method == "lora" and not allow_lora:
            raise ValueError("mode=train does not support training.method=lora; use mode=finetune for LoRA.")
        device, dtype = self.generator._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self.generator._resolve_model_path(cfg)
        self.log.info("Loading '%s' (source=%s, %s) for training...", cfg.model.name, cfg.model.source, model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if cfg.model.trust_remote_code:
            self.generator._ensure_remote_code_compat()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only,
        )
        if cfg.training.method == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise RuntimeError("training.method=lora requires: pip install -U peft") from e
            lora_config = LoraConfig(
                r=cfg.training.lora_r, lora_alpha=cfg.training.lora_alpha,
                lora_dropout=cfg.training.lora_dropout, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        model.to(device)
        conversations = self.datamanager._load_conversations(cfg)
        if not conversations:
            raise ValueError(f"No conversations found under data.path='{cfg.data.path}' (data.type='{cfg.data.type}').")
        self.log.info("Loaded %d conversation(s).", len(conversations))
        random.seed(cfg.seed)
        random.shuffle(conversations)
        num_eval = (
            max(1, int(len(conversations) * cfg.data.eval_split))
            if cfg.data.eval_split > 0 and len(conversations) > 1 else 0
        )
        eval_conversations = conversations[:num_eval]
        train_conversations = conversations[num_eval:]
        self.log.info("Train: %d  Eval: %d", len(train_conversations), len(eval_conversations))
        train_dataset = self.datamanager._build_dataset(train_conversations, tokenizer, cfg.training.max_length)
        eval_dataset = self.datamanager._build_dataset(eval_conversations, tokenizer, cfg.training.max_length) if eval_conversations else None
        collator = self.datamanager._make_collator(tokenizer.pad_token_id)
        optimizer = self._build_optimizer(cfg, model)

        training_args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.epochs,
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.grad_accum_steps,
            warmup_ratio=cfg.training.warmup_ratio,
            logging_steps=cfg.training.logging_steps,
            save_steps=cfg.training.save_steps,
            save_total_limit=2,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=cfg.training.eval_steps if eval_dataset else None,
            bf16=(dtype.__str__() == "torch.bfloat16" and device != "mps"),
            fp16=(dtype.__str__() == "torch.float16" and device != "mps"),
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=collator, optimizers=(optimizer, None),
        )
        self.log.info("Starting %s (method=%s, optimizer=%s)...", cfg.mode.name, cfg.training.method, cfg.optimizer.name)
        trainer.train()
        trainer.save_model(cfg.training.output_dir)
        tokenizer.save_pretrained(cfg.training.output_dir)
        self.log.info("Saved to '%s'.", cfg.training.output_dir)
        if cfg.training.method == "lora" and cfg.training.merge_adapter:
            merged_model = model.merge_and_unload()
            merged_dir = Path(cfg.training.output_dir) / "merged"
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            self.log.info("Merged model saved to '%s'.", merged_dir)

    def run_train(self, cfg) -> None:
        if not cfg.training.enabled:
            raise ValueError("mode=train requires training.enabled=true")
        self._run_training(cfg, allow_lora=False)

    def run_finetune(self, cfg) -> None:
        if not cfg.training.enabled:
            raise ValueError("mode=finetune requires training.enabled=true")
        self._run_training(cfg, allow_lora=True)
