#!/usr/bin/env python3
"""
configurator.py

Single entry point for running google/gemma-3-270m-it or microsoft/phi-4
(or any local checkout of either) in one of six modes, fully driven by a
Hydra + OmegaConf config tree under `conf/`:

  chat               interactive multi-turn chat session, logged to
                      <output_dir>/chat_log/ (same shape as gemma_chat.py)
  generate           single prompt in, single response out, no history
  tool_calling       single-turn generation with tool/function definitions
                      passed through the model's chat template
  structured_output  single-turn generation constrained to a JSON schema
  train              continued pretraining / full fine-tune from scratch
                      on a data source (full weights only)
  finetune           supervised fine-tune (LoRA or full) on a data source

Which model, which mode, every generation/optimizer/training hyperparameter,
and where the data comes from are all config values, not hardcoded flags --
any of it can be swapped from the command line:

    # Chat with Gemma 3 270M-IT interactively (pulled from the HF hub, using
    # whatever's in your local HF cache or downloading it if not)
    python configurator.py model=gemma3_270m mode=chat

    # Same, but loaded strictly from a local checkout -- never touches the
    # network, fails clearly if the path is wrong
    python3 configurator.py model=gemma3_270m mode=chat model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/google/gemma-3-270m-it

    # Phi-4-mini-instruct defaults to loading from ./models/llms/microsoft/
    # Phi-4-mini-instruct locally -- override the path for your machine, or
    # switch back to the hub with model.source=huggingface
    python configurator.py model=phi4_instruct mode=chat \\
        model.local_path=/actual/path/to/Phi-4-mini-instruct

    # One-shot generation with Phi-4, higher temperature
    python configurator.py model=phi4_instruct mode=generate \\
        generation.temperature=1.2 prompt="Explain the tides."

    # Tool calling with Phi-4
    python configurator.py model=phi4_instruct mode=tool_calling \\
        tool_calling.enabled=true prompt="What's the weather in Boston?" \\
        'tool_calling.tools=[{name: get_weather, description: "Get current weather", parameters: {type: object, properties: {location: {type: string}}}}]'

    # Structured output (JSON-schema constrained)
    python configurator.py mode=structured_output structured_output.enabled=true \\
        structured_output.schema_path=./schemas/person.json prompt="Describe a fictional person."

    # LoRA fine-tune Gemma 3 270M-IT on chat_log sessions with a custom LR
    python configurator.py model=gemma3_270m mode=finetune \\
        training.enabled=true training.method=lora optimizer=adamw \\
        optimizer.learning_rate=5e-5 data=chat_log

    # Full fine-tune Phi-4 on your own JSONL dataset with SGD
    python configurator.py model=phi4_instruct mode=finetune \\
        training.enabled=true training.method=full data=jsonl data.path=my_data.jsonl \\
        optimizer=sgd optimizer.learning_rate=1e-4

Run `python configurator.py --cfg job` to print the fully composed config
for any combination of overrides without loading a model -- handy for
sanity-checking a command before it downloads/loads anything.
"""

import logging

import hydra
from omegaconf import OmegaConf

from src.modes import Modes
from src.schema import Config, register_configs

modes = Modes()

log = logging.getLogger(__name__)

register_configs()

_DISPATCH = {
    "chat": modes.run_chat,
    "generate": modes.run_generate,
    "tool_calling": modes.run_tool_calling,
    "structured_output": modes.run_structured_output,
    "train": modes.run_train,
    "finetune": modes.run_finetune,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    mode_name = cfg.mode.name
    if mode_name not in _DISPATCH:
        raise ValueError(f"Unknown mode '{mode_name}'. Choose one of: {sorted(_DISPATCH)}")

    _DISPATCH[mode_name](cfg)


if __name__ == "__main__":
    main()
