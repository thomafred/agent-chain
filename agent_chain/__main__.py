import random
import string

import dotenv
import typer
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.llms import OpenAI
from langchain.vectorstores import VectorStore
from langchain.vectorstores.redis import Redis

from .baby_agi import BabyAGI
from .factories import ObjectFactoryRegistry
from .tools import FileWriteTool, SerpApi, TodoChain, Wikipedia, dbpedia, ocean_info_hub

app = typer.Typer()


@app.command()
def baby_agi(
    objective: str,
    openapi_key: str = typer.Argument("openapi-key", envvar="OPENAI_API_KEY"),
    max_iterations: int = 5,
    max_tokens: int = 100,
    temperature: float = 0.0,
    verbose: bool = False,
):
    ObjectFactoryRegistry.add(OpenAIEmbeddings, client="", openai_api_key=openapi_key)
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
        texts=["My wife's name is Xingyi"],
    )

    llm = OpenAI(
        client="",
        openai_api_key=openapi_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

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
    objective: str,
    openapi_key: str = typer.Argument("openapi-key", envvar="OPENAI_API_KEY"),
    max_tokens: int = 1000,
    temperature: float = 0.0,
    verbose: bool = False,
):
    ObjectFactoryRegistry.add(OpenAIEmbeddings, client="", openai_api_key=openapi_key)
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
        texts=["My wife's name is Xingyi"],
    )

    llm = ChatOpenAI(
        client="",
        openai_api_key=openapi_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

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


if __name__ == "__main__":
    dotenv.load_dotenv()
    app()
