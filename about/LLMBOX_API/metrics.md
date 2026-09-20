# Python API Documentation for `src/metrics.py`

**Top-level Functions:**

### `_safe`
Signature: `_safe(fn, default)`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_cfg_get`
Signature: `_cfg_get(cfg, section, name, default) -> Any`
Docstring: getattr chain that tolerates missing sections and OmegaConf MISSING values.

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_mean`
Signature: `_mean(xs) -> Optional[float]`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_perplexity`
Signature: `_perplexity(loss) -> Optional[float]`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_extract_losses`
Signature: `_extract_losses(trainer) -> Tuple[Optional[float], Optional[float]]`
Docstring: Latest train loss and eval loss from trainer.state.log_history.

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_count_params`
Signature: `_count_params(model) -> int`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_gpu_count`
Signature: `_gpu_count(handles) -> int`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_fmt_duration`
Signature: `_fmt_duration(seconds) -> str`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `explain_flops`
Signature: `explain_flops(flops) -> Tuple[Optional[int], Optional[str]]`
Docstring: Return (1-5 level, plain-language explanation) for a FLOP count.

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_na`
Signature: `_na(v, fmt, unit) -> str`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_bytes`
Signature: `_bytes(n) -> str`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `_report_sections`
Signature: `_report_sections(m)`
Docstring: _No docstring provided. Why might this be a problem for maintenance?_

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `format_report`
Signature: `format_report(m, fmt, loss_curve) -> str`
Docstring: Render metrics as 'text' (console) or 'markdown' (file).

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `plot_loss_curve`
Signature: `plot_loss_curve(trainer, path) -> Optional[str]`
Docstring: Save a loss curve PNG from trainer.state.log_history. Needs matplotlib.

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.

### `save_report`
Signature: `save_report(m, output_dir, trainer) -> Dict[str, Path]`
Docstring: Write training_metrics.json, training_report.md, and (if possible) loss_curve.png.

- Think: In what context would you call this function? What parameters are required and which might have defaults? Explore its implementation.


**Defined Classes:**

## `TrainingMetrics`
Docstring: Metrics and resource use for one training job.

Methods:
- `tokens_remaining(self) -> int`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `token_budget_used_percent(self) -> Optional[float]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `over_budget(self) -> bool`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `to_grouped_dict(self) -> Dict[str, Dict[str, Any]]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `ResourceMonitor`
Docstring: Samples resource use on a background thread every `interval` seconds.

Methods:
- `__init__(self, interval)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_init_nvml(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_sample(self, cpu) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_run(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `start(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `stop(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

## `MetricsTracker`
Docstring: _No class docstring. Should a class always have one? When?_

Methods:
- `__init__(self, cfg, interval, gpu_watts, cpu_watts, carbon_intensity)`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `start(self) -> None`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `add_tokens(self, n) -> None`
    - Custom loops: call once per batch with the number of tokens processed.
    - [Question] How would you use this method in a real project? What side effects might it have?

- `_tokens(self, trainer) -> Tuple[int, str]`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?

- `stop(self, trainer, model, loss, eval_loss) -> TrainingMetrics`
    - _No method docstring. Would you trust this API?_
    - [Question] How would you use this method in a real project? What side effects might it have?
