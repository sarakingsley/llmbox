# Python API Documentation for `src/utilities.py`

**Defined Classes:**

## `Utilities`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
- `__init__(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `use_sibling_directory(self, directory_name) -> Path`
    - Enable imports from a sibling directory's `src` package.

Paths are resolved relative to this Python file, not the terminal's
current working directory.

Optionally change the working directory so relative file reads also
resolve from the sibling directory.
    - [Question] How would you use this method in a real project? What side effects might it have?
