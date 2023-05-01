from textwrap import dedent

from langchain import WikipediaAPIWrapper
from langchain.agents import Tool


class Wikipedia(WikipediaAPIWrapper):

    def as_tool(self) -> Tool:
        return Tool(
            name="Wikipedia",
            func=self.run,
            description=dedent(
                """
                Allows you to search for topics on Wikipedia, an extensive and widely-used online encyclopedia that provides
                free access to millions of articles written collaboratively by volunteers.

                Use this function to quickly access information about a wide range of subjects, from historical events to
                scientific concepts, biographies, and more.
                """
            )
        )
