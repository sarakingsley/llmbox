import os
import sys
import typing
import pydantic
import typing
import json

from typing import List, Optional
from pydantic import BaseModel

class PersonRecord(BaseModel):
    firstname: str
    lastname: str
