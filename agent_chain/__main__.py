import random
import string
from pathlib import Path
from typing import Iterable, Optional

import dotenv
import requests
import typer
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.llms import OpenAI
from langchain.vectorstores import VectorStore
from langchain.vectorstores.redis import Redis

from .baby_agi import BabyAGI
from .factories import ObjectFactoryRegistry
from .tools import FileWriteTool, SerpApi, TodoChain, Wikipedia, dbpedia, ocean_info_hub

app = typer.Typer(pretty_exceptions_enable=False)


def _vectorstore_knowledge() -> Iterable[str]:
    urls = [
        "https://www.w3.org/TR/rdf11-concepts/",
        "https://www.w3.org/TR/rdf11-mt/",
        "https://www.w3.org/TR/rdf11-primer/",
        "https://www.w3.org/TR/turtle/",
        "https://www.w3.org/TR/xml",
        "https://www.w3.org/TR/sparql11-query/",
        "https://www.w3.org/TR/sparql11-update/",
        "https://www.w3.org/TR/sparql11-protocol/",
        "https://www.w3.org/TR/json-ld11/",
    ]

    for url in urls:
        req = requests.get(url)
        req.raise_for_status()

        soup = BeautifulSoup(req.text, "html.parser")
        yield soup.get_text()


def _load_objective(
    objective: Optional[str] = None,
    objective_file: Optional[Path] = None,
) -> str:
    if (objective and objective_file) or (not objective and not objective_file):
        raise typer.BadParameter("Either objective or objective_file must be specified")

    if objective_file and not objective_file.exists():
        raise typer.BadParameter(f"File {objective_file} does not exist")

    if objective_file:
        objective = objective_file.read_text()

    return objective


@app.command()
def baby_agi(
    objective: str,
    max_iterations: int = 5,
    max_tokens: int = 100,
    temperature: float = 0.0,
    verbose: bool = False,
):
    ObjectFactoryRegistry.add(
        Redis,
        redis_url="redis://localhost:6379",
        index_name=lambda: "".join(random.sample(string.ascii_lowercase, 8)),
        index_cls=VectorStore,
    )

    embeddings_model = ObjectFactoryRegistry.fetch(OpenAIEmbeddings)()

    vectorstore = Redis.from_texts(
        redis_url="redis://localhost:6379",
        index_name="baby_agi",
        embedding=embeddings_model,
        texts=list(_vectorstore_knowledge()),
    )

    llm = ObjectFactoryRegistry.fetch(OpenAI)(temperature=temperature, max_tokens=max_tokens)

    agent = BabyAGI.from_llm(
        llm=llm,
        objective=objective,
        verbose=verbose,
        max_iterations=max_iterations,
        vectorstore=vectorstore,
        tools=[
            Wikipedia().as_tool(),
            SerpApi().as_tool(),
            FileWriteTool(),
        ],
    )

    agent({"objective": objective})


@app.command()
def auto_gpt(
    objective: Optional[str] = None,
    objective_file: Optional[Path] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    verbose: bool = False,
):
    objective = _load_objective(objective, objective_file)

    ObjectFactoryRegistry.add(
        Redis,
        redis_url="redis://localhost:6379",
        index_name=lambda: "".join(random.sample(string.ascii_lowercase, 8)),
        index_cls=VectorStore,
    )

    embeddings_model = ObjectFactoryRegistry.fetch(OpenAIEmbeddings)()

    vectorstore = Redis.from_texts(
        redis_url="redis://localhost:6379",
        index_name="baby_agi",
        embedding=embeddings_model,
        texts=list(_vectorstore_knowledge()),
    )

    llm = ObjectFactoryRegistry.fetch(OpenAI)(temperature=temperature, max_tokens=max_tokens)

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="assistant",
        tools=[
            SerpApi().as_tool(),
            FileWriteTool(),
            ocean_info_hub.as_tool(),
            dbpedia.as_tool(),
            TodoChain.from_llm(llm).as_tool(),
        ],
        llm=llm,
        memory=vectorstore.as_retriever(),
    )

    if verbose:
        agent.chain.verbose = True

    agent.run([objective])


@app.callback()
def main(
    openapi_key: str = typer.Option("openapi-key", envvar="OPENAI_API_KEY"),
    openapi_model_name: str = typer.Option(default="gpt-3.5-turbo", envvar="OPENAPI_MODEL_NANE"),
):
    ObjectFactoryRegistry.add(OpenAIEmbeddings, client="", openai_api_key=openapi_key, model=openapi_model_name)
    ObjectFactoryRegistry.add(OpenAI, client="", openai_api_key=openapi_key, model_name=openapi_model_name)
    ObjectFactoryRegistry.add(ChatOpenAI, client="", openai_api_key=openapi_key, model_name=openapi_model_name)


if __name__ == "__main__":
    dotenv.load_dotenv()
    app()
