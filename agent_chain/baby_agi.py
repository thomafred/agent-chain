import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field, PrivateAttr

LOG = logging.getLogger(__name__)


class TaskCreationChain(LLMChain):
    """Chain for task creation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> "TaskCreationChain":
        """Get the response parser"""
        task_creation_template = (
            "You are a task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )

        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )

        return cls(llm=llm, prompt=prompt, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain for task prioritization."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> "TaskPrioritizationChain":
        """Get the response parser"""
        task_prioritization_template = (
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )

        prompt = PromptTemplate(
            template=task_prioritization_template, input_variables=["task_names", "next_task_id", "objective"]
        )

        return cls(llm=llm, prompt=prompt, verbose=verbose)


class ExecutionChain(LLMChain):
    """Chain for task execution."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> "ExecutionChain":
        """Get the response parser"""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )

        prompt = PromptTemplate(template=execution_template, input_variables=["task", "context", "objective"])

        return cls(llm=llm, prompt=prompt, verbose=verbose)


def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task"""

    response = task_creation_chain.run(
        result=result, task_description=task_description, incomplete_tasks=", ".join(task_list), objective=objective
    )

    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    current_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize the tasks"""

    task_names = [t["task_name"] for t in task_list]

    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=current_task_id + 1, objective=objective
    )
    new_tasks = response.split("\n")

    prioritized_task_list = []
    for task_string in new_tasks:
        task_string = task_string.strip()

        if not task_string:
            continue

        task_parts = task_string.split(".", 1)
        if len(task_parts) == 2:
            task_id = int(task_parts[0].strip())
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})

    return prioritized_task_list


def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query"""

    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []

    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))

    tasks = [item.metadata.get("task") for item in sorted_results]
    ret = [str(item) for item in tasks if item]

    return ret


def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
    """Execute the task"""

    context = _get_top_tasks(vectorstore, objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)


class BabyAGI(Chain, BaseModel):
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    task_execution_chain: ExecutionChain = Field(...)
    vectorstore: VectorStore = Field(...)

    max_iterations: Optional[int] = 10

    printback: Callable[[str], None] = Field(default=print)

    _task_id_counter: int = PrivateAttr(0)

    class Config:
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        """Add a task to the task list"""

        self.task_list.append(task)

    def print_task_list(self):
        """Print the task list"""

        for task in self.task_list:
            self.printback(f"{task['task_id']}. {task['task_name']}")

    def print_next_task(self, task: Dict):
        """Print the next task"""

        self.printback(f"{task['task_id']}: {task['task_name']}")

    def print_task_result(self, result: str):
        """Print the task result"""
        self.printback(f"Task result: {result}")

    @property
    def input_keys(self) -> List[str]:
        """Get the input keys"""
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        """Get the output keys"""
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent"""

        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")

        self.add_task({"task_id": 1, "task_name": first_task})

        num_iterations = 0
        while self.task_list:
            self.print_task_list()

            # Step 1: Get the first task
            task = self.task_list.popleft()
            self.print_next_task(task)

            # Step 2: Execute the task
            result = execute_task(self.vectorstore, self.task_execution_chain, objective, task["task_name"])

            active_task_id = task["task_id"]
            self.print_task_result(result)

            # Step 3: Store the result
            result_id = f"result_{active_task_id}"
            self.vectorstore.add_texts(texts=[result], metadatas=[{"task": task["task_name"]}], ids=[result_id])

            # Step 4: Create new tasks and reprioritize task list
            new_tasks = get_next_task(
                self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
            )

            for new_task in new_tasks:
                self._task_id_counter += 1
                new_task.update({"task_id": self._task_id_counter})
                self.add_task(new_task)

            self.task_list = deque(
                prioritize_tasks(self.task_prioritization_chain, active_task_id, list(self.task_list), objective)
            )

            num_iterations += 1
            if self.max_iterations and num_iterations >= self.max_iterations:
                self.printback("Reached max iterations")
                break

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs) -> "BabyAGI":
        """Initialize the BabyAGI Controller"""

        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(llm, verbose=verbose)
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)

        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            task_execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )
