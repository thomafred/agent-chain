import os
from typing import Type, Any

import fsspec
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, validator


class FileReadInput(BaseModel):
    """Input for ReadFileTool."""

    file_path: str = Field(..., description="name of file")


class FileWriteInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")


class FileReadTool(BaseTool):

    fs: fsspec.AbstractFileSystem = Field(default_factory=lambda: fsspec.filesystem("file"))
    base_dir: str = "./bin"

    name: str = "read_file"
    tool_args: Type[BaseModel] = FileReadInput
    description: str = "Read file from disk"

    @validator("base_dir", always=True)
    def _abs_base_dir(cls, v: str):
        if os.path.isabs(v):
            return v

        return os.path.abspath(os.path.join(os.getcwd(), v))

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError()


class FileWriteTool(FileReadTool):

    name: str = "write_file"
    tool_args: Type[BaseModel] = FileWriteInput
    description: str = "Write file to disk"

    def _run(self, file_path: str, text: str) -> str:

        if os.path.isabs(file_path):
            if not file_path.startswith(self.base_dir):
                return "Error: file path must be within the current working directory."
        else:
            file_path = os.path.abspath(os.path.join(self.base_dir, file_path))

            if not file_path.startswith(self.base_dir):
                return "Error: file path must be within the current working directory."

        if not text:
            return "Error: Both file path and text must be provided and text cannot be empty. Please se the task docs"

        try:
            with self.fs.open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            return "File written to successfully."
        except Exception as e:
            return "Error: " + str(e)
