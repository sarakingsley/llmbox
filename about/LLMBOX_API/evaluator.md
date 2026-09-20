# Python API Documentation for `src/evaluator.py`

**Defined Classes:**

## `PrepareEvaluationInput`
Docstring: Prepare and format evaluation inputs for LLM evaluation. Allows selection
of a representative sample. Supports data with query/response (pair)
or response only (single).

Methods:
- `__init__(self, data_path, sample_size, kind)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `load_data(self) -> List[Dict[str, Any]]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_detect_kind(self, record) -> str`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `select_sample(self, records, random_seed) -> List[Dict[str, Any]]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `format_for_evaluation(self) -> List[Dict[str, Any]]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `save_evaluation_dataset(self, out_path) -> str`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `HumanEvaluation`
Docstring: Human-in-the-loop rating of AI responses with Likert scale judgments per example,
logging criteria and instructions and saving results to JSON. Collects method choice for agreement
and writes both data and a criteria/notes markdown file.

Methods:
- `__init__(self, evaldataset_path)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_dataset(self) -> List[Dict[str, Any]]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run(self)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_save_criteria_md(self, outdir)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `LLMJudgeEvaluation`
Docstring: Automatically evaluate a dataset using an LLM judge. Accepts Pydantic model or dataclass (could be function/class).
Records labels and computes the chosen evaluation measure.

Methods:
- `__init__(self, evaldataset_path, judge_fn, judge_schema_source)`
    - judge_fn should accept (example, criteria_description) and return label/int/decision
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_load_dataset(self)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `run(self)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `AgreementMeasures`
Docstring: Minimal metrics for agreement between multiple labelers (inter-rater agreement).
Example: simple percent agreement, placeholder for Cohen's kappa, etc.

Methods:
- `percent_agreement(labels1, labels2) -> float`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `cohen_kappa(labels1, labels2)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `StandardEvaluationMeasures`
Docstring: Minimal standard metrics for LLM evaluation (accuracy, average rating, etc.)

Methods:
- `apply(dataset, name) -> Dict[str, Any]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `CustomEvaluationMeasures`
Docstring: Minimal stub allowing a user to define their own evaluation function.

Methods:
- `apply(dataset, name) -> Dict[str, Any]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?
