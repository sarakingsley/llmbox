import json
import logging
import random
import sys
import uuid
import torch
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class DataManager:

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)
        self.IGNORE_INDEX = -100

    def _load_conversations(self, cfg):
        data = cfg.data
        if data.type == "chat_log":
            return self._load_conversations_from_chat_log(Path(data.path), min_turns=data.min_turns)
        elif data.type == "jsonl":
            return self._load_conversations_from_jsonl(Path(data.path))
        raise ValueError(f"Unknown data.type '{data.type}'. Use 'chat_log' or 'jsonl'.")


    def _load_conversations_from_chat_log(self, chat_log_dir: Path, min_turns: int):
        conversations = []
        if not chat_log_dir.exists():
            return conversations
        for log_file in sorted(chat_log_dir.glob("*.json")):
            try:
                record = json.loads(log_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Skipping unreadable log '%s': %s", log_file, e)
                continue
            turns = record.get("turns", [])
            if len(turns) < min_turns:
                continue
            system_prompt = (
                record.get("session", {}).get("settings", {}).get("system_prompt")
                or record.get("session", {}).get("system_prompt")
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for turn in turns:
                user_content = turn.get("user", {}).get("content")
                assistant_content = turn.get("assistant", {}).get("content")
                if not user_content or not assistant_content:
                    continue
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})
            if len(messages) >= 2:
                conversations.append({"messages": messages})
        return conversations


    def _load_conversations_from_jsonl(self, dataset_path: Path):
        conversations = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed JSONL line %d: %s", line_num, e)
                    continue
                messages = record.get("messages")
                if not messages:
                    log.warning("Skipping line %d: no 'messages' field.", line_num)
                    continue
                conversations.append({"messages": messages})
        return conversations


    def _build_labeled_example(self, tokenizer, messages, max_length):
        """Mask every token that isn't part of an assistant turn with
        IGNORE_INDEX, so the loss only trains the model to produce assistant
        responses, not to reproduce the prompt/system text."""
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        labels = [self.IGNORE_INDEX] * len(full_ids)
        prev_len = 0
        for i, message in enumerate(messages):
            prefix_ids = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)
            cur_len = len(prefix_ids)
            if message.get("role") == "assistant":
                end = min(cur_len, len(full_ids))
                labels[prev_len:end] = full_ids[prev_len:end]
            prev_len = cur_len
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]
        return full_ids, labels


    def _build_dataset(self, conversations, tokenizer, max_length):
        from torch.utils.data import Dataset

        class ChatDataset(Dataset):
            def __init__(self):
                self.examples = []
                skipped = 0
                for convo in conversations:
                    input_ids, labels = self._build_labeled_example(tokenizer, convo["messages"], max_length)
                    if all(l == self.IGNORE_INDEX for l in labels):
                        skipped += 1
                        continue
                    self.examples.append({"input_ids": input_ids, "labels": labels})
                if skipped:
                    log.info("Skipped %d conversation(s) with no trainable tokens.", skipped)

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                return self.examples[idx]

        return ChatDataset()


    def _make_collator(self, pad_token_id):
        import torch

        def collate(batch):
            max_len = max(len(ex["input_ids"]) for ex in batch)
            input_ids, labels, attention_mask = [], [], []
            for ex in batch:
                pad_len = max_len - len(ex["input_ids"])
                input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
                labels.append(ex["labels"] + [self.IGNORE_INDEX] * pad_len)
                attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        return collate
