from typing import List

from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import BaseLLM
from langchain.tools import Tool


class PlanAndExecuteAgent(PlanAndExecute):
    @classmethod
    def from_llm(cls, llm: BaseLLM, tools: List[Tool], verbose: bool = False) -> "PlanAndExecuteAgent":
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=verbose)

        return cls(planner=planner, executer=executor, verbose=verbose)
