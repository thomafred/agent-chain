from textwrap import dedent

from langchain.experimental import AutoGPT
from langchain.tools import Tool


class AutoGpt(AutoGPT):
    def _run_tool(self, objective: str) -> str:
        return self.run([objective])

    def as_tool(
        self,
        name: str,
    ):
        return Tool(
            name=name,
            func=self._run_tool,
            description=dedent(
                """
                An autonomous AI agent that can be used as a tool.
                Will attempt to solve a given objective and report back once it is done.

                Input: an objective to solve.
                Output: a solution to that objective or why it failed.

                Please be very clear what the objective is! The more specific you are, the better.
                """
            ),
        )
