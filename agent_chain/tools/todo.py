from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool


class TodoChain(LLMChain):
    """Chain for todo-lists."""

    @classmethod
    def from_llm(cls, llm: LLMChain, verbose: bool = False) -> "TodoChain":
        todo_prompt = PromptTemplate.from_template(dedent(
            """
            You are a planner who is an expert at coming up with a todo list for a given objective.
            Come up with a todo list for this objective: {objective}
            """
        ))

        return cls(llm=llm, prompt=todo_prompt, verbose=verbose)

    def as_tool(self) -> Tool:
        return Tool(
            name="TODO",
            func=self.run,
            description=dedent(
                """
                Useful for when you need to come up with todo lists.
                Input: an objective to create a todo list for.
                Output: a todo list for that objective.

                Please be very clear what the objective is!
                """
            )
        )
