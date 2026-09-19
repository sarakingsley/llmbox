## Run the tests

From the directory containing `data_services.py`:

```bash
python3 -m pip install pytest
python3 -m pytest -q tests/test_data_services.py
```

Your module imports `torch` and `omegaconf`, so both must be installed even though the tests do not require OmegaConf directly. Alternatively, remove the unused `OmegaConf` import from the module.

Useful commands:

```bash
# Stop at the first failure and show detailed test names.
python -m pytest tests/test_data_services.py -x -vv

# Run only the assistant-masking tests.
python -m pytest tests/test_data_services.py -k "mask or prefix or truncation" -vv

# Display log messages during execution.
python -m pytest tests/test_data_services.py --log-cli-level=INFO
```

**Scope:** these tests check the module’s behavior using a controlled tokenizer. Before a real training run, also inspect token IDs and labels produced by your actual tokenizer and chat template—especially whether assistant end-of-turn tokens are supervised.
