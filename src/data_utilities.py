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

import os
import sys
import json
import random

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

class DataUtility:

    def __init__(self) -> None:
        pass

    @staticmethod
    def write_jsonl(
        examples: Iterable[Mapping[str, str]],
        path: str,
    ) -> int:
        count = 0
        with open(path, "w", encoding="utf-8") as destination:
            for example in examples:
                destination.write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )
                count += 1
        return count

    @staticmethod
    def split_and_save_jsonl(
        examples: Iterable[Mapping[str, Any]],
        output_directory: str | Path,
        *,
        train_fraction: float = 0.8,
        seed: int = 42,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """
        Split formatted LLM examples and save train.jsonl and test.jsonl.

        Args:
            examples:
                Iterable of dictionaries containing string-valued
                "prompt" and "completion" fields.
            output_directory:
                Destination directory. Created if necessary.
            train_fraction:
                Fraction assigned to training, strictly between 0 and 1.
                The test split receives the remainder.
            seed:
                Random seed for reproducibility with the same input order.
            overwrite:
                Whether existing output files may be replaced.

        Returns:
            Paths, row counts, and the actual training fraction.

        Notes:
            - Requires at least two examples.
            - Keeps at least one example in each split.
            - Loads serialized examples into memory to shuffle them.
            - Preserves all fields in each example.
        """
        if (
            isinstance(train_fraction, bool)
            or not isinstance(train_fraction, (int, float))
            or not 0 < train_fraction < 1
        ):
            raise ValueError("train_fraction must be a number between 0 and 1.")
        output_directory = Path(output_directory).expanduser().resolve()
        train_path = output_directory / "train.jsonl"
        test_path = output_directory / "test.jsonl"
        if not overwrite:
            for path in (train_path, test_path):
                if path.exists():
                    raise FileExistsError(
                        f"{path} already exists. Set overwrite=True to replace it."
                    )
        # Validate and serialize before creating output files.
        lines: list[str] = []
        for index, example in enumerate(examples):
            if not isinstance(example, Mapping):
                raise TypeError(f"Example {index} must be a mapping.")

            for field in ("prompt", "completion"):
                if field not in example:
                    raise ValueError(f"Example {index} is missing {field!r}.")
                if not isinstance(example[field], str):
                    raise TypeError(
                        f"Example {index}: {field!r} must be a string."
                    )
            # An empty prompt is valid for text-only language modeling.
            if not example["completion"].strip():
                raise ValueError(f"Example {index} has an empty completion.")
            try:
                lines.append(
                    json.dumps(
                        dict(example),
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Example {index} cannot be serialized as JSON: {exc}"
                ) from exc
        total_count = len(lines)
        if total_count < 2:
            raise ValueError("At least two examples are required.")
        # Local RNG: does not modify Python's global random state.
        random.Random(seed).shuffle(lines)
        # Round down, then ensure both splits contain at least one example.
        train_count = max(
            1,
            min(total_count - 1, int(total_count * train_fraction)),
        )
        test_count = total_count - train_count
        output_directory.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        with train_path.open(mode, encoding="utf-8") as train_file:
            for index in range(train_count):
                train_file.write(lines[index])
        with test_path.open(mode, encoding="utf-8") as test_file:
            for index in range(train_count, total_count):
                test_file.write(lines[index])
        return {
            "train_path": train_path,
            "test_path": test_path,
            "train_count": train_count,
            "test_count": test_count,
            "total_count": total_count,
            "actual_train_fraction": train_count / total_count,
        }

    @staticmethod
    def standardize_llm_dataset(
        rows: Iterable[Mapping[str, Any]],
        *,
        text_columns: str | Sequence[str],
        target_columns: str | Sequence[str] | None = None,
        instruction: str = "",
        label_maps: Mapping[str, Mapping[Any, Any]] | None = None,
    ) -> Iterator[dict[str, str]]:
        """
        Convert row-oriented data into {"prompt": str, "completion": str}.

        Modes:
        target_columns=None:
            Language modeling. Selected input columns become the completion.
            The prompt is empty.

        target_columns provided:
            Supervised generation. Input columns become the prompt, and target
            columns become the completion.

        Formatting:
        - Strings are preserved.
        - Numbers, booleans, lists, and dictionaries use JSON formatting.
        - A single target produces a scalar/text completion.
        - Multiple targets produce a JSON-object completion.
        - label_maps may replace target values, e.g. 0 -> "negative".

        Validation:
        - Required columns must exist and cannot contain None.
        - Empty input strings and completions are rejected.
        - Input and target columns cannot overlap.
        - Non-JSON-compatible values must be converted by the caller.
        - Errors include the zero-based row index.

        Notes:
        - Only selected columns are retained.
        - No tokenization, splitting, truncation, or padding occurs here.
        - Because this is a generator, validation runs when it is consumed.
        """

        def normalize_columns(
            columns: str | Sequence[str],
            argument_name: str,
        ) -> list[str]:
            result = [columns] if isinstance(columns, str) else list(columns)
            if not result:
                raise ValueError(f"{argument_name} must not be empty.")
            if any(not isinstance(c, str) or not c.strip() for c in result):
                raise ValueError(
                    f"{argument_name} must contain non-empty column names."
                )
            if len(result) != len(set(result)):
                raise ValueError(f"{argument_name} contains duplicate columns.")
            return result

        def json_text(value: Any) -> str:
            # Reject NaN and Infinity instead of writing invalid JSON.
            return json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
            )

        def render(value: Any) -> str:
            return value if isinstance(value, str) else json_text(value)

        def read_value(row: Mapping[str, Any], column: str) -> Any:
            if column not in row:
                raise ValueError(f"missing column {column!r}")
            value = row[column]
            if value is None:
                raise ValueError(f"column {column!r} is null")
            return value
        inputs = normalize_columns(text_columns, "text_columns")
        targets = (
            normalize_columns(target_columns, "target_columns")
            if target_columns is not None
            else []
        )
        overlap = set(inputs) & set(targets)
        if overlap:
            raise ValueError(
                f"Input and target columns overlap: {sorted(overlap)}"
            )
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string.")
        if instruction.strip() and not targets:
            raise ValueError(
                "instruction requires target_columns. "
                "Text-only training has an empty prompt."
            )
        maps = label_maps if label_maps is not None else {}
        unknown_maps = set(maps) - set(targets)
        if unknown_maps:
            raise ValueError(
                f"label_maps refers to non-target columns: {sorted(unknown_maps)}"
            )
        # A dict of columns, one row dict, or DatasetDict is not a row iterable.
        if isinstance(rows, Mapping):
            raise TypeError(
                "rows must be an iterable of row dictionaries. "
                "For one row, use [row]. For a dataset dictionary, select a split."
            )
        for row_index, row in enumerate(rows):
            try:
                if not isinstance(row, Mapping):
                    raise TypeError("each row must be a mapping")
                input_values: dict[str, str] = {}
                for column in inputs:
                    text = render(read_value(row, column))
                    if not text.strip():
                        raise ValueError(f"input column {column!r} is empty")
                    input_values[column] = text
                if not targets:
                    # Preserve a single document as-is. For multiple input fields,
                    # retain field names to preserve their meaning.
                    completion = (
                        input_values[inputs[0]]
                        if len(inputs) == 1
                        else "\n\n".join(
                            f"{column}:\n{text}"
                            for column, text in input_values.items()
                        )
                    )
                    prompt = ""
                else:
                    output_values: dict[str, Any] = {}
                    for column in targets:
                        value = read_value(row, column)

                        if column in maps:
                            try:
                                value = maps[column][value]
                            except (KeyError, TypeError) as exc:
                                raise ValueError(
                                    f"no label mapping for {column!r} "
                                    f"value {value!r}"
                                ) from exc
                        if value is None:
                            raise ValueError(
                                f"target column {column!r} resolves to null"
                            )
                        if isinstance(value, str) and not value.strip():
                            raise ValueError(
                                f"target column {column!r} is empty"
                            )
                        output_values[column] = value
                    completion = (
                        render(output_values[targets[0]])
                        if len(targets) == 1
                        else json_text(output_values)
                    )
                    input_block = "\n\n".join(
                        f"{column}:\n{text}"
                        for column, text in input_values.items()
                    )
                    prefix = (
                        instruction.strip() + "\n\n"
                        if instruction.strip()
                        else ""
                    )
                    prompt = prefix + f"Input:\n{input_block}\n\nOutput:\n"
                yield {
                    "prompt": prompt,
                    "completion": completion,
                }
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Row {row_index}: {exc}") from exc
