import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from src.generation import GenerationManager
from src.datasets import DataManager


class Logger:

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
