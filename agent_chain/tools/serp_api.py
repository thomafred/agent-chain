from textwrap import dedent
from typing import Type

from langchain import SerpAPIWrapper
from langchain.agents import Tool
from pydantic import BaseModel


class SerpApi(SerpAPIWrapper):

    def as_tool(self) -> Tool:
        return Tool(
            name="Search",
            func=self.run,
            description=dedent(
                """
                Useful for when you need to answer questions about current events. You should ask targeted questions
                """
            )
        )
