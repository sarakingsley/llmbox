from pathlib import Path
from pydantic import BaseModel

import src.personrecords




def save_model_json(
    model: BaseModel,
    directory: str | Path,
    filename: str = "config.json",
) -> Path:
    """Save a Pydantic model as JSON, creating the directory if needed."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    # Keep the output inside the specified directory.
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must be a file name, not a path")

    output_path = directory / filename
    output_path.write_text(
        model.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path

'''
# Example model
class ModelConfig(BaseModel):
    model: str
    mode: str = "chat"
    temperature: float = 0.7

'''
if __name__ == "__main__":
    #config = ModelConfig(model="gemma3_270m")
    #config = src.personrecords.PersonRecord()

    output = save_model_json(
        #config,
        model=src.personrecords.PersonRecord(firstname="sara", lastname="king"),
        directory="./datasets/stroutjson/",
        filename="structuredoutput.json",
    )
    print(f"Saved to {output.resolve()}")
