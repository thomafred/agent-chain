from typing import Protocol

from langchain.agents import Tool


class LlmTool(Protocol):
    """LLM Chain that can be used as a tool"""

    def as_tool(
        cls,
        name: str,
        description: str,
    ) -> Tool:
        """
        Returns a Tool instance that can be used to interact with this LLM chain as a tool
        """
