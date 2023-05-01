import re
from typing import Any, List, Union

from langchain.agents import AgentOutputParser, Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.vectorstores import VectorStore
from pydantic import Field, PrivateAttr

from .factories import ObjectFactoryRegistry


class ToolsPromptParser(AgentOutputParser):
    _action_rx: re.Pattern = PrivateAttr(
        re.compile(r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", re.DOTALL)
    )

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[1].strip()},
                log=llm_output,
            )

        # Parse the action and action input

        match = self._action_rx.search(llm_output)
        if not match:
            raise ValueError(f"Could not parse LLM output: {llm_output}")

        action = match.group(1).strip()
        action_input = match.group(2).strip()

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(' "'), log=llm_output)


class ToolsPromptTemplate(StringPromptTemplate):
    output_parser: ToolsPromptParser = Field(default_factory=ToolsPromptParser)

    template: str
    tools: List[Tool]

    _vectorstore: VectorStore = PrivateAttr()

    def __init__(self, **data: Any):
        data["input_variables"] = data.get("input_variables", []) + ["intermediate_steps"]

        super().__init__(**data)
        self._vectorstore = ObjectFactoryRegistry.fetch(VectorStore).call_classmethod(
            "from_documents",
            [Document(page_content=tool.description, metadata={"index": i}) for i, tool in enumerate(self.tools)],
            ObjectFactoryRegistry.fetch(OpenAIEmbeddings)(),
        )

    def _get_tools(self, input: str) -> List[Tool]:
        """Get the tools related to a specific input"""

        retriever = self._vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(input)
        return [self.tools[doc.metadata["index"]] for doc in docs]

    def format(self, **kwargs: Any) -> str:
        intermediate_steps = kwargs.get("intermediate_steps", [])
        thoughts = ""

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThoughts: "

        kwargs["agent_scratchpad"] = thoughts
        tools = self._get_tools(kwargs["task"])
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)
