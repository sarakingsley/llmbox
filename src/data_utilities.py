


import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any
import json

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
