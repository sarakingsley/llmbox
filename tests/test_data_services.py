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


"""Tests for data_services.py.

The suite covers:
- Dataset normalization and validation.
- JSONL writing and reproducible splitting.
- Conversation loading and malformed-record handling.
- Assistant-only loss masking.
- Truncation and empty-target filtering.
- Dataset indexing and batch padding.
- An end-to-end preparation/loading/batching workflow.

FakeChatTokenizer is intentionally simple. It tests our masking logic,
not compatibility with every real Hugging Face tokenizer or template.
"""

import copy
import json
import logging
import pickle
import random

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from data_services import (
    IGNORE_INDEX,
    CausalLMCollator,
    DataTransformer,
    TokenizedChatDataset,
    TrainingDataLoader,
    _utc_now_iso,
)


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read test output without depending on the module under test."""
    with path.open(encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def write_test_jsonl(path, records):
    """Create loader input independently of DataTransformer.write_jsonl."""
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_turn(user="Question", assistant="Answer"):
    """Create one complete chat-log turn."""
    return {
        "user": {"content": user},
        "assistant": {"content": assistant},
    }


@pytest.fixture
def transformer():
    return DataTransformer()


@pytest.fixture
def loader():
    return TrainingDataLoader()


@pytest.fixture
def messages():
    return [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]


@pytest.fixture
def pair_rows():
    """Unique IDs let split tests detect omissions and duplicates."""
    return [
        {
            "id": index,
            "prompt": f"Question {index}\n",
            "completion": f"Answer {index}",
        }
        for index in range(10)
    ]


class FakeChatTokenizer:
    """A deterministic, character-based chat tokenizer.

    Vocabulary:
        1  = beginning of conversation
        2  = end of turn
        10 = system header
        11 = user header
        12 = assistant header

    Content characters are represented by 100 + ord(character).

    Example:
        user "Q", assistant "A"

        input_ids:
            [1, 11, 181, 2, 12, 165, 2]

        assistant_masks:
            [0,  0,   0, 0,  0,   1, 1]

    This template includes the assistant's end-of-turn token in its mask.
    """

    ROLE_IDS = {
        "system": 10,
        "user": 11,
        "assistant": 12,
    }

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        truncation=False,
        padding=False,
    ):
        # These assertions also check how the loader calls the tokenizer.
        assert tokenize is True
        assert truncation is False
        assert padding is False

        token_ids = [1]
        assistant_mask = [0]

        for message in messages:
            role = message["role"]
            content_ids = [
                100 + ord(character)
                for character in message["content"]
            ]

            token_ids.append(self.ROLE_IDS[role])
            assistant_mask.append(0)

            token_ids.extend(content_ids)
            token_ids.append(2)

            is_assistant = int(role == "assistant")
            assistant_mask.extend(
                [is_assistant] * (len(content_ids) + 1)
            )

        if add_generation_prompt:
            token_ids.append(self.ROLE_IDS["assistant"])
            assistant_mask.append(0)

        if return_dict:
            result = {"input_ids": token_ids}

            if return_assistant_tokens_mask:
                result["assistant_masks"] = assistant_mask

            return result

        return token_ids


class FixedOutputTokenizer:
    """Return a supplied result to test malformed tokenizer outputs."""

    def __init__(self, output):
        self.output = output

    def apply_chat_template(self, messages, **kwargs):
        return copy.deepcopy(self.output)


@pytest.fixture
def tokenizer():
    return FakeChatTokenizer()


# ---------------------------------------------------------------------------
# General helpers and DataTransformer
# ---------------------------------------------------------------------------

def test_utc_timestamp_is_timezone_aware():
    timestamp = datetime.fromisoformat(_utc_now_iso())

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("already text", "already text"),
        ('{"answer": "A"}', '{"answer": "A"}'),
        (True, "true"),
        (42, "42"),
        (["a", "b"], '["a", "b"]'),
        ({"answer": "A"}, '{"answer": "A"}'),
    ],
)
def test_render_value_preserves_strings_and_serializes_other_values(
    value,
    expected,
):
    assert DataTransformer._render_value(value) == expected


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_json_serialization_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        DataTransformer._json_text({"value": value})


@pytest.mark.parametrize("value", [{1, 2}, object()])
def test_json_serialization_rejects_unsupported_values(value):
    with pytest.raises(TypeError):
        DataTransformer._json_text(value)


def test_normalization_preserves_input_and_rebuilds_derived_fields():
    completion = '{"OptionA": "Café", "CorrectAnswer": "A"}'
    original = {
        "prompt": "Input:\nQuestion\n\nOutput:\n",
        "completion": completion,
        "metadata": {"source": "example"},
        "text": "stale text",
        "messages": [{"role": "user", "content": "stale"}],
    }
    before = copy.deepcopy(original)

    result = DataTransformer.normalize_prompt_completion(original)

    assert original == before
    assert result is not original
    assert result["completion"] == completion
    assert result["text"] == original["prompt"] + completion
    assert result["metadata"] == {"source": "example"}
    assert result["messages"] == [
        {"role": "user", "content": original["prompt"]},
        {"role": "assistant", "content": completion},
    ]


@pytest.mark.parametrize(
    "completion",
    [
        {"CorrectAnswer": "A", "explanation": "Café"},
        ["A", "B"],
    ],
)
def test_normalization_serializes_structured_completions_once(completion):
    result = DataTransformer.normalize_prompt_completion({
        "prompt": "Question\n",
        "completion": completion,
    })

    assert isinstance(result["completion"], str)
    assert json.loads(result["completion"]) == completion


def test_empty_prompt_does_not_create_empty_user_message():
    result = DataTransformer.normalize_prompt_completion({
        "prompt": "",
        "completion": "A document.",
    })

    assert result["text"] == "A document."
    assert result["messages"] == [
        {"role": "assistant", "content": "A document."},
    ]


@pytest.mark.parametrize(
    ("example", "error_type"),
    [
        ([], TypeError),
        ({"completion": "A"}, ValueError),
        ({"prompt": "Q"}, ValueError),
        ({"prompt": None, "completion": "A"}, TypeError),
        ({"prompt": "Q", "completion": 123}, TypeError),
        ({"prompt": "Q", "completion": " \n"}, ValueError),
    ],
)
def test_normalization_rejects_invalid_pairs(example, error_type):
    with pytest.raises(error_type):
        DataTransformer.normalize_prompt_completion(example)


def test_existing_pairs_accept_generator_without_reformatting(transformer):
    original = {
        "prompt": "Input:\nAlready formatted\n\nOutput:\n",
        "completion": '{"CorrectAnswer": "A"}',
        "unused": "discard me",
    }

    examples = list(
        transformer.standardize_llm_dataset(
            row for row in [original]
        )
    )

    assert len(examples) == 1
    assert examples[0]["prompt"] == original["prompt"]
    assert examples[0]["completion"] == original["completion"]
    assert set(examples[0]) == {
        "prompt", "completion", "text", "messages",
    }


def test_single_column_language_modeling_preserves_whitespace(transformer):
    document = "  First paragraph.\n\nSecond paragraph.\n"

    result = list(transformer.standardize_llm_dataset(
        [{"document": document}],
        text_columns="document",
    ))[0]

    assert result["prompt"] == ""
    assert result["completion"] == document
    assert result["text"] == document


def test_multiple_language_model_columns_keep_names_and_order(transformer):
    result = list(transformer.standardize_llm_dataset(
        [{"title": "Example", "body": "Document"}],
        text_columns=["title", "body"],
    ))[0]

    assert result["completion"] == "title:\nExample\n\nbody:\nDocument"


def test_supervised_generation_applies_instruction_and_label_map(transformer):
    result = list(transformer.standardize_llm_dataset(
        [{"review": "Excellent.", "label": 1}],
        text_columns="review",
        target_columns="label",
        instruction="  Classify sentiment.  ",
        label_maps={"label": {0: "negative", 1: "positive"}},
    ))[0]

    assert result["prompt"] == (
        "Classify sentiment.\n\n"
        "Input:\nreview:\nExcellent.\n\nOutput:\n"
    )
    assert result["completion"] == "positive"
    assert result["text"] == result["prompt"] + "positive"


def test_multiple_targets_preserve_json_value_types(transformer):
    result = list(transformer.standardize_llm_dataset(
        [{"question": "Q", "answer": "A", "confidence": 0.75}],
        text_columns="question",
        target_columns=["answer", "confidence"],
    ))[0]

    assert json.loads(result["completion"]) == {
        "answer": "A",
        "confidence": 0.75,
    }


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"target_columns": "label"}, ValueError),
        ({"instruction": "Explain"}, ValueError),
        ({"instruction": 123}, TypeError),
        ({"text_columns": []}, ValueError),
        ({"text_columns": ["x", "x"]}, ValueError),
        ({"text_columns": [" "]}, ValueError),
        ({"text_columns": b"x"}, TypeError),
        ({"text_columns": 123}, TypeError),
        ({"text_columns": "x", "target_columns": "x"}, ValueError),
        ({"text_columns": "x", "instruction": "Explain"}, ValueError),
        ({"label_maps": []}, TypeError),
        ({"label_maps": {"unknown": {0: "zero"}}}, ValueError),
        (
            {
                "text_columns": "x",
                "target_columns": "y",
                "label_maps": {"y": []},
            },
            TypeError,
        ),
    ],
)
def test_standardizer_rejects_invalid_configuration(
    transformer,
    kwargs,
    error_type,
):
    # Consume the generator: validation does not run at creation time.
    with pytest.raises(error_type):
        list(transformer.standardize_llm_dataset([], **kwargs))


@pytest.mark.parametrize("dataset", [{"x": "one row"}, "file.jsonl", 123])
def test_standardizer_rejects_non_row_iterables(transformer, dataset):
    with pytest.raises(TypeError):
        list(transformer.standardize_llm_dataset(dataset))


@pytest.mark.parametrize(
    "bad_row",
    [
        [],
        {},
        {"document": None},
        {"document": " \n"},
        {"document": {1, 2}},
    ],
)
def test_row_errors_include_zero_based_index(transformer, bad_row):
    rows = [{"document": "Valid"}, bad_row]

    with pytest.raises(ValueError, match=r"Row 1:"):
        list(transformer.standardize_llm_dataset(
            rows,
            text_columns="document",
        ))


def test_missing_label_mapping_reports_row_and_problem(transformer):
    with pytest.raises(ValueError, match=r"Row 0: no label mapping"):
        list(transformer.standardize_llm_dataset(
            [{"review": "Fine", "label": 2}],
            text_columns="review",
            target_columns="label",
            label_maps={"label": {0: "negative", 1: "positive"}},
        ))


# ---------------------------------------------------------------------------
# JSONL writing and reproducible splitting
# ---------------------------------------------------------------------------

def test_write_jsonl_creates_directories_and_preserves_nested_unicode(tmp_path):
    path = tmp_path / "nested" / "examples.jsonl"
    rows = [{
        "text": "Café",
        "messages": [{"role": "assistant", "content": "こんにちは"}],
    }]

    count = DataTransformer.write_jsonl(rows, path)

    assert count == 1
    assert read_jsonl(path) == rows
    assert "Café" in path.read_text(encoding="utf-8")
    assert path.read_bytes().endswith(b"\n")


def test_write_jsonl_respects_overwrite_false(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text("original\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        DataTransformer.write_jsonl(
            [{"text": "replacement"}],
            path,
            overwrite=False,
        )

    assert path.read_text(encoding="utf-8") == "original\n"


def test_invalid_rows_do_not_truncate_existing_output(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text("original\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Example 1:"):
        DataTransformer.write_jsonl(
            [{"text": "valid"}, {"unsupported": {1, 2}}],
            path,
        )

    assert path.read_text(encoding="utf-8") == "original\n"


def test_split_is_reproducible_complete_and_preserves_global_rng(
    tmp_path,
    pair_rows,
):
    original_rows = copy.deepcopy(pair_rows)
    random_state_before = random.getstate()

    first = DataTransformer.split_and_save_jsonl(
        pair_rows,
        tmp_path / "first",
        train_fraction=0.7,
        seed=19,
    )
    second = DataTransformer.split_and_save_jsonl(
        pair_rows,
        tmp_path / "second",
        train_fraction=0.7,
        seed=19,
    )

    assert random.getstate() == random_state_before
    assert pair_rows == original_rows

    train_rows = read_jsonl(first["train_path"])
    test_rows = read_jsonl(first["test_path"])

    assert first["train_count"] == 7
    assert first["test_count"] == 3
    assert first["total_count"] == 10
    assert first["actual_train_fraction"] == pytest.approx(0.7)

    train_ids = {row["id"] for row in train_rows}
    test_ids = {row["id"] for row in test_rows}

    assert train_ids.isdisjoint(test_ids)
    assert len(train_rows) + len(test_rows) == len(pair_rows)
    assert sorted(train_rows + test_rows, key=lambda row: row["id"]) == pair_rows

    assert (
        first["train_path"].read_bytes()
        == second["train_path"].read_bytes()
    )
    assert (
        first["test_path"].read_bytes()
        == second["test_path"].read_bytes()
    )


@pytest.mark.parametrize("fraction", [0.001, 0.999])
def test_two_examples_always_produce_two_nonempty_splits(
    tmp_path,
    pair_rows,
    fraction,
):
    result = DataTransformer.split_and_save_jsonl(
        pair_rows[:2],
        tmp_path,
        train_fraction=fraction,
    )

    assert result["train_count"] == 1
    assert result["test_count"] == 1


@pytest.mark.parametrize(
    "fraction",
    [True, 0, 1, -0.1, 1.1, "0.8", float("nan"), float("inf")],
)
def test_split_rejects_invalid_fraction(tmp_path, pair_rows, fraction):
    with pytest.raises(ValueError, match="train_fraction"):
        DataTransformer.split_and_save_jsonl(
            pair_rows,
            tmp_path,
            train_fraction=fraction,
        )


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_split_rejects_invalid_seed(tmp_path, pair_rows, seed):
    with pytest.raises(TypeError, match="seed"):
        DataTransformer.split_and_save_jsonl(
            pair_rows,
            tmp_path,
            seed=seed,
        )


@pytest.mark.parametrize("count", [0, 1])
def test_split_requires_at_least_two_rows(tmp_path, pair_rows, count):
    with pytest.raises(ValueError, match="At least two"):
        DataTransformer.split_and_save_jsonl(
            pair_rows[:count],
            tmp_path,
        )


def test_split_prevalidates_before_replacing_either_file(tmp_path, pair_rows):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text("old train", encoding="utf-8")
    test_path.write_text("old test", encoding="utf-8")

    invalid_rows = pair_rows + [{"prompt": "Q", "completion": ""}]

    with pytest.raises(ValueError):
        DataTransformer.split_and_save_jsonl(invalid_rows, tmp_path)

    assert train_path.read_text(encoding="utf-8") == "old train"
    assert test_path.read_text(encoding="utf-8") == "old test"


def test_split_overwrite_false_checks_both_output_paths(tmp_path, pair_rows):
    # Only test.jsonl exists: train.jsonl must not be created first.
    (tmp_path / "test.jsonl").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        DataTransformer.split_and_save_jsonl(
            pair_rows,
            tmp_path,
            overwrite=False,
        )

    assert not (tmp_path / "train.jsonl").exists()


# ---------------------------------------------------------------------------
# Conversation validation and source loading
# ---------------------------------------------------------------------------

def test_message_validation_returns_a_copy(loader, messages):
    result = loader._validate_messages(messages)

    assert result == messages
    assert result is not messages
    assert result[0] is not messages[0]


@pytest.mark.parametrize(
    ("invalid_messages", "error_type"),
    [
        (None, ValueError),
        ([], ValueError),
        (["not a mapping"], TypeError),
        ([{"role": "tool", "content": "result"}], ValueError),
        ([{"role": "assistant", "content": []}], TypeError),
        ([{"role": "assistant", "content": " "}], ValueError),
        ([{"role": "user", "content": "No answer"}], ValueError),
        (
            [{"role": "assistant", "content": "A", "tool_calls": []}],
            ValueError,
        ),
    ],
)
def test_invalid_messages_are_rejected(loader, invalid_messages, error_type):
    with pytest.raises(error_type):
        loader._validate_messages(invalid_messages)


def test_record_loader_accepts_prompt_completion(loader):
    result = loader._record_to_conversation({
        "prompt": "Question\n",
        "completion": {"CorrectAnswer": "A"},
    })

    assert result["messages"][0] == {
        "role": "user",
        "content": "Question\n",
    }
    assert json.loads(result["messages"][1]["content"]) == {
        "CorrectAnswer": "A",
    }


def test_explicit_invalid_messages_do_not_fall_back_to_valid_pair(loader):
    with pytest.raises(ValueError, match="messages"):
        loader._record_to_conversation({
            "messages": [],
            "prompt": "Q",
            "completion": "A",
        })


def test_text_only_record_is_not_silently_converted_to_chat(loader):
    with pytest.raises(ValueError, match="Expected"):
        loader._record_to_conversation({"text": "A document"})


def test_jsonl_skips_invalid_records_and_reports_physical_line_numbers(
    tmp_path,
    loader,
    messages,
    caplog,
):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n"                                           # Line 1: blank
        + json.dumps({"messages": messages}) + "\n"     # Line 2: valid
        + "{broken json\n"                             # Line 3: invalid
        + "[]\n"                                       # Line 4: wrong shape
        + '{"text": "not chat"}\n'                      # Line 5: wrong schema
        + '{"prompt": "Q2", "completion": "A2"}\n',     # Line 6: valid
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        conversations = loader._load_conversations_from_jsonl(path)

    assert len(conversations) == 2
    assert conversations[0]["messages"] == messages
    assert conversations[1]["messages"][-1]["content"] == "A2"

    for line_number in (3, 4, 5):
        assert f"line {line_number}:" in caplog.text


def test_missing_jsonl_file_raises(tmp_path, loader):
    with pytest.raises(FileNotFoundError):
        loader._load_conversations_from_jsonl(tmp_path / "missing.jsonl")


def test_chat_logs_use_sorted_files_and_system_prompt_precedence(
    tmp_path,
    loader,
):
    records = {
        "b.json": {
            "session": {"system_prompt": "Fallback"},
            "turns": [make_turn("Q-b", "A-b")],
        },
        "a.json": {
            "session": {
                "system_prompt": "Fallback",
                "settings": {"system_prompt": "Preferred"},
            },
            "turns": [make_turn("Q-a", "A-a")],
        },
    }

    for filename, record in records.items():
        (tmp_path / filename).write_text(
            json.dumps(record),
            encoding="utf-8",
        )

    conversations = loader._load_conversations_from_chat_log(
        tmp_path,
        min_turns=1,
    )

    assert len(conversations) == 2
    assert conversations[0]["messages"][0]["content"] == "Preferred"
    assert conversations[0]["messages"][1]["content"] == "Q-a"
    assert conversations[1]["messages"][0]["content"] == "Fallback"


def test_chat_log_min_turns_counts_complete_pairs(tmp_path, loader, caplog):
    record = {
        "turns": [
            make_turn(),
            {"user": {"content": "Unanswered"}},
            make_turn("Q", " "),
        ],
    }
    (tmp_path / "session.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        too_short = loader._load_conversations_from_chat_log(
            tmp_path,
            min_turns=2,
        )
        accepted = loader._load_conversations_from_chat_log(
            tmp_path,
            min_turns=1,
        )

    assert too_short == []
    assert len(accepted) == 1
    assert len(accepted[0]["messages"]) == 2
    assert "Skipping" in caplog.text


def test_malformed_chat_log_is_logged_and_skipped(tmp_path, loader, caplog):
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "good.json").write_text(
        json.dumps({"turns": [make_turn()]}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        conversations = loader._load_conversations_from_chat_log(
            tmp_path,
            min_turns=1,
        )

    assert len(conversations) == 1
    assert "bad.json" in caplog.text


@pytest.mark.parametrize("min_turns", [True, 0, -1, 1.5])
def test_chat_log_rejects_invalid_min_turns(tmp_path, loader, min_turns):
    with pytest.raises(ValueError, match="min_turns"):
        loader._load_conversations_from_chat_log(tmp_path, min_turns)


def test_chat_log_path_must_be_existing_directory(tmp_path, loader):
    with pytest.raises(FileNotFoundError):
        loader._load_conversations_from_chat_log(tmp_path / "missing", 1)

    file_path = tmp_path / "file.json"
    file_path.write_text("{}", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        loader._load_conversations_from_chat_log(file_path, 1)


def test_config_dispatch_handles_jsonl_and_default_chat_min_turns(
    tmp_path,
    loader,
    messages,
):
    jsonl_path = tmp_path / "examples.jsonl"
    write_test_jsonl(jsonl_path, [{"messages": messages}])

    jsonl_config = SimpleNamespace(
        data=SimpleNamespace(type="jsonl", path=str(jsonl_path))
    )
    assert len(loader._load_conversations(jsonl_config)) == 1

    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    (log_directory / "session.json").write_text(
        json.dumps({"turns": [make_turn()]}),
        encoding="utf-8",
    )

    # No min_turns field: dispatch should default to 1.
    chat_config = SimpleNamespace(
        data=SimpleNamespace(type="chat_log", path=str(log_directory))
    )
    assert len(loader._load_conversations(chat_config)) == 1


def test_unknown_data_type_raises(loader):
    config = SimpleNamespace(
        data=SimpleNamespace(type="csv", path="unused.csv")
    )

    with pytest.raises(ValueError, match="Unknown data.type"):
        loader._load_conversations(config)


# ---------------------------------------------------------------------------
# Tokenization and assistant-only label masking
# ---------------------------------------------------------------------------

def test_unknown_mask_strategy_raises():
    with pytest.raises(ValueError, match="assistant_mask_strategy"):
        TrainingDataLoader(assistant_mask_strategy="guess")


@pytest.mark.parametrize("strategy", ["template", "verified_prefix"])
def test_only_assistant_content_and_end_tokens_receive_labels(
    tokenizer,
    strategy,
):
    loader = TrainingDataLoader(assistant_mask_strategy=strategy)
    conversation = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "R"},
        {"role": "assistant", "content": "B"},
    ]
    original = copy.deepcopy(conversation)

    input_ids, labels = loader._build_labeled_example(
        tokenizer,
        conversation,
        max_length=100,
    )

    # Hard-coded expectations make masking errors visible rather than
    # calculating the expected answer with the implementation under test.
    assert input_ids == [
        1,
        10, 183, 2,   # system S
        11, 181, 2,   # user Q
        12, 165, 2,   # assistant A
        11, 182, 2,   # user R
        12, 166, 2,   # assistant B
    ]
    assert labels == (
        [IGNORE_INDEX] * 8
        + [165, 2]
        + [IGNORE_INDEX] * 4
        + [166, 2]
    )
    assert conversation == original


@pytest.mark.parametrize("strategy", ["template", "verified_prefix"])
def test_right_truncation_keeps_inputs_and_labels_aligned(
    tokenizer,
    messages,
    strategy,
):
    loader = TrainingDataLoader(assistant_mask_strategy=strategy)

    input_ids, labels = loader._build_labeled_example(
        tokenizer,
        messages,
        max_length=6,
    )

    assert input_ids == [1, 11, 181, 2, 12, 165]
    assert labels == [IGNORE_INDEX] * 5 + [165]


@pytest.mark.parametrize("max_length", [True, 0, 1, 2.5])
def test_invalid_max_length_raises(loader, tokenizer, messages, max_length):
    with pytest.raises(ValueError, match="max_length"):
        loader._build_labeled_example(tokenizer, messages, max_length)


@pytest.mark.parametrize(
    ("output", "error_type", "message"),
    [
        (
            [1, 2],
            TypeError,
            "dictionary",
        ),
        (
            {"input_ids": [1, 2]},
            ValueError,
            "assistant_masks",
        ),
        (
            {"input_ids": [1, 2], "assistant_masks": [1]},
            ValueError,
            "length",
        ),
        (
            {"input_ids": [1, 2], "assistant_masks": [0, 2]},
            ValueError,
            "only 0 and 1",
        ),
        (
            {"input_ids": [1, 2], "assistant_masks": [0, 0]},
            ValueError,
            "no assistant tokens",
        ),
    ],
)
def test_template_masks_are_validated(
    loader,
    messages,
    output,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        loader._build_labeled_example(
            FixedOutputTokenizer(output),
            messages,
            max_length=20,
        )


@pytest.mark.parametrize(
    ("token_ids", "error_type"),
    [
        ((1, 2), TypeError),
        ([[1, 2]], ValueError),
        ([True], ValueError),
        ([-1], ValueError),
        ([1.5], ValueError),
    ],
)
def test_invalid_token_ids_are_rejected(loader, token_ids, error_type):
    with pytest.raises(error_type):
        loader._validate_token_ids(token_ids)


def test_first_token_is_not_counted_as_a_next_token_target(loader, messages):
    tokenizer = FixedOutputTokenizer({
        "input_ids": [7, 8],
        "assistant_masks": [1, 0],
    })

    input_ids, labels = loader._build_labeled_example(
        tokenizer,
        messages,
        max_length=20,
    )

    assert input_ids == [7, 8]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX]

    with pytest.raises(ValueError, match="No usable training examples"):
        loader._build_dataset(
            [{"messages": messages}],
            tokenizer,
            max_length=20,
        )


def test_verified_prefix_rejects_missing_generation_header(messages):
    class NoGenerationHeaderTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            kwargs["add_generation_prompt"] = False
            return super().apply_chat_template(messages, **kwargs)

    loader = TrainingDataLoader(assistant_mask_strategy="verified_prefix")

    with pytest.raises(ValueError, match="generation header"):
        loader._build_labeled_example(
            NoGenerationHeaderTokenizer(),
            messages,
            max_length=100,
        )


def test_verified_prefix_rejects_nonmatching_token_prefix(messages):
    class ChangingPrefixTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)

            # Simulate a template that changes earlier tokens when rendering
            # an unfinished assistant generation prefix.
            if kwargs.get("add_generation_prompt"):
                result[0] = 999

            return result

    loader = TrainingDataLoader(assistant_mask_strategy="verified_prefix")

    with pytest.raises(ValueError, match="prefixes do not match"):
        loader._build_labeled_example(
            ChangingPrefixTokenizer(),
            messages,
            max_length=100,
        )


def test_verified_prefix_rejects_assistant_only_documents(tokenizer):
    loader = TrainingDataLoader(assistant_mask_strategy="verified_prefix")

    with pytest.raises(ValueError, match="requires context"):
        loader._build_labeled_example(
            tokenizer,
            [{"role": "assistant", "content": "A"}],
            max_length=100,
        )


# ---------------------------------------------------------------------------
# Dataset construction and batching
# ---------------------------------------------------------------------------

def test_dataset_supports_length_and_indexing():
    examples = [
        {"input_ids": [1, 2], "labels": [IGNORE_INDEX, 2]},
        {"input_ids": [1, 3], "labels": [IGNORE_INDEX, 3]},
    ]

    dataset = TokenizedChatDataset(examples)

    assert len(dataset) == 2
    assert dataset[0] == examples[0]
    assert dataset[1] == examples[1]

    with pytest.raises(IndexError):
        _ = dataset[2]


def test_build_dataset_skips_conversations_losing_all_targets(
    loader,
    tokenizer,
    messages,
    caplog,
):
    long_prompt = [
        {"role": "user", "content": "Q" * 20},
        {"role": "assistant", "content": "A"},
    ]

    with caplog.at_level(logging.INFO):
        dataset = loader._build_dataset(
            [
                {"messages": long_prompt},
                {"messages": messages},
            ],
            tokenizer,
            max_length=7,
        )

    assert len(dataset) == 1
    assert dataset[0]["labels"] == [IGNORE_INDEX] * 5 + [165, 2]
    assert "Skipped 1 conversations" in caplog.text


def test_build_dataset_rejects_empty_input(loader, tokenizer):
    with pytest.raises(ValueError, match="No usable training examples"):
        loader._build_dataset([], tokenizer, max_length=20)


def test_build_dataset_adds_conversation_index_to_tokenizer_errors(
    loader,
    messages,
):
    bad_tokenizer = FixedOutputTokenizer({
        "input_ids": [1, 2],
        "assistant_masks": [0, 0],
    })

    with pytest.raises(ValueError, match="conversation 0") as error:
        loader._build_dataset(
            [{"messages": messages}],
            bad_tokenizer,
            max_length=20,
        )

    assert error.value.__cause__ is not None


def test_collator_right_pads_without_hiding_prompt_tokens():
    examples = [
        {
            # Token 2 is both a real EOS token and the padding token.
            "input_ids": [1, 2],
            "labels": [IGNORE_INDEX, 2],
        },
        {
            "input_ids": [1, 11, 165, 2],
            "labels": [IGNORE_INDEX, IGNORE_INDEX, 165, 2],
        },
    ]
    original = copy.deepcopy(examples)

    batch = CausalLMCollator(pad_token_id=2)(examples)

    assert batch["input_ids"].tolist() == [
        [1, 2, 2, 2],
        [1, 11, 165, 2],
    ]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX],
        [IGNORE_INDEX, IGNORE_INDEX, 165, 2],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]

    assert examples == original

    for tensor in batch.values():
        assert tensor.dtype == torch.long
        assert tensor.shape == (2, 4)


@pytest.mark.parametrize("pad_token_id", [None, True, -1, 1.5])
def test_collator_rejects_invalid_pad_token_id(pad_token_id):
    with pytest.raises(ValueError, match="pad_token_id"):
        CausalLMCollator(pad_token_id)


@pytest.mark.parametrize(
    "batch",
    [
        [],
        [{"input_ids": [], "labels": []}],
        [{"input_ids": [1, 2], "labels": [IGNORE_INDEX]}],
    ],
)
def test_collator_rejects_invalid_batches(batch):
    with pytest.raises(ValueError):
        CausalLMCollator(pad_token_id=0)(batch)


def test_dataset_and_collator_are_pickleable(loader):
    dataset = TokenizedChatDataset([
        {"input_ids": [1, 2], "labels": [IGNORE_INDEX, 2]},
    ])
    collator = loader._make_collator(pad_token_id=0)

    restored_dataset = pickle.loads(pickle.dumps(dataset))
    restored_collator = pickle.loads(pickle.dumps(collator))

    assert restored_dataset[0] == dataset[0]
    assert restored_collator([restored_dataset[0]])["input_ids"].tolist() == [
        [1, 2],
    ]


def test_end_to_end_standardize_write_load_tokenize_and_batch(
    tmp_path,
    transformer,
    loader,
    tokenizer,
):
    """Exercise the component boundaries, not just individual helpers."""
    rows = [
        {"prompt": "Q", "completion": "A"},
        {"prompt": "Longer question", "completion": "Answer"},
    ]
    path = tmp_path / "prepared.jsonl"

    standardized = transformer.standardize_llm_dataset(rows)
    assert transformer.write_jsonl(standardized, path) == 2

    conversations = loader._load_conversations_from_jsonl(path)
    dataset = loader._build_dataset(
        conversations,
        tokenizer,
        max_length=100,
    )

    batches = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=loader._make_collator(pad_token_id=0),
    )
    batch = next(iter(batches))

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape

    trainable = batch["labels"] != IGNORE_INDEX
    padding = batch["attention_mask"] == 0

    # Every example must contribute at least one target.
    assert trainable.any(dim=1).all().item()

    # All supervised targets must equal the corresponding unshifted input IDs.
    assert torch.equal(
        batch["labels"][trainable],
        batch["input_ids"][trainable],
    )

    # Padding must never contribute to the loss.
    assert (batch["labels"][padding] == IGNORE_INDEX).all().item()

    # The shorter example should actually have padding.
    assert padding[0].any().item()
