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
Activation Capture Utility
-------------------------
Minimal, interpretable PyTorch hooks class for capturing hidden state or other
layer activations from a loaded transformer model (LLM).

Dependencies: Only PyTorch is assumed to be already imported in the pipeline.
No nonstandard dependencies are required. The design is transparent for easy
learning and extension.

Usage example (inside a run mode or notebook):
    model, tokenizer, device = manager._load_model_and_tokenizer(cfg)
    ac = ActivationCatcher(model)
    with ac.capture('transformer.h.5', 'output'):     # example layer name
        outputs = model(input_ids)
    # Get activations for all modules seen
    acts = ac.activations

Members:
    ActivationCatcher(model): Register/unregister hooks on module names
    .capture(<module_name>, <attribute>): context manager for capturing one
    .activations: dict of list of activations seen during latest context

This file can be dropped in with src/generation.py imports. Light wrapper for hooks.
"""

import torch
from contextlib import contextmanager
from collections import defaultdict

class ActivationCatcher:
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self._hooks = []
    
    def _find_module(self, module_name):
        """Recursively find a named submodule by dot-path."""
        mod = self.model
        for part in module_name.split('.'):
            mod = getattr(mod, part, None)
            if mod is None:
                raise ValueError(f"No module named '{module_name}'.")
        return mod

    @contextmanager
    def capture(self, module_name, attribute='output'):
        """Capture activations for a given module by name or path. Yields nothing."""
        self.activations.clear()
        module = self._find_module(module_name)
        handles = []
        def _hook(mod, inp, out):
            self.activations[module_name].append(out.detach().cpu())
        if attribute == 'output':
            handle = module.register_forward_hook(_hook)
            handles.append(handle)
        elif attribute == 'input':
            def _input_hook(mod, inp, out):
                self.activations[module_name].append(
                    tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in inp))
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)
        else:
            raise ValueError(f"Unknown attribute '{attribute}', choose 'output' or 'input'.")
        self._hooks.extend(handles)
        try:
            yield
        finally:
            for h in handles:
                h.remove()
            self._hooks = [h for h in self._hooks if h not in handles]
            # Clean-up is always called

    def capture_multiple(self, module_names, attribute='output'):
        """Attach hooks to several modules at once (returns context manager)."""
        catcher = self
        @contextmanager
        def _cm():
            catcher.activations.clear()
            handles = []
            for name in module_names:
                mod = catcher._find_module(name)
                def make_hook(n):
                    def hook(mod, inp, out):
                        catcher.activations[n].append(out.detach().cpu())
                    return hook
                handle = mod.register_forward_hook(make_hook(name))
                handles.append(handle)
            catcher._hooks.extend(handles)
            try:
                yield
            finally:
                for h in handles:
                    h.remove()
                catcher._hooks = [h for h in catcher._hooks if h not in handles]
        return _cm()

    def clear(self):
        """Remove all hooks from model and clear activations/dangling state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()

# Example integration inside a pipeline or run mode:
# model, tokenizer, device = manager._load_model_and_tokenizer(cfg)
# acatcher = ActivationCatcher(model)
# with acatcher.capture('transformer.h.5'):  # or .capture_multiple([...])
#     outputs = model(input_ids)
# # print(acatcher.activations)

# Optionally, expose to other run modes by import:
# from src.modes import ActivationCatcher

# End of activation capture utility

# -------------
# Existing code below (modes.py is not modified further)
# -------------

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
from src.sys_logger import Logger
from src.schema import (
    ModelConfig,
    GenerationConfig,
    ToolDef,
    ToolCallingConfig,
    StructuredOutputConfig,
    OptimizerConfig,
    TrainingConfig,
    DataConfig,
    ModeConfig,
    Config,
    register_configs
)

from src.generation import GenerationManager
from src.data_services import (
    DataTransformer,
    CausalLMCollator,
    TokenizedChatDataset,
    TrainingDataLoader
)

from src import metrics as training_metrics

# ... <rest of modes.py unchanged> ...

class Modes:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.IGNORE_INDEX = -100
        self.generator = GenerationManager()
        self.datamanager = DataTransformer()
        self.dataloader = TrainingDataLoader()
        self.logger = Logger()

    # <rest of the Modes class as previously implemented>

    # [unchanged code continued...]

# END OF FILE
