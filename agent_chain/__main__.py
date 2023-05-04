import random
import string

import dotenv
import typer
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict
from langchain.vectorstores import VectorStore
from langchain.vectorstores.redis import Redis

from .baby_agi import BabyAGI
from .factories import ObjectFactoryRegistry
from .tools import FileWriteTool, SerpApi, Wikipedia

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def llm(
    prompt: str,
    max_tokens: int = 100,
    openapi_key: str = typer.Argument("openapi-key", envvar="OPENAI_API_KEY"),
):
    llm = OpenAI(client="", openai_api_key=openapi_key, max_tokens=max_tokens)

    tools = load_tools(["serpapi"])
    agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True)

    history = ChatMessageHistory()
    # history.add_user_message("Hello, in this conversation please only use metric units."
    # " Please make sure to convert all messages to metrics before displaying them to me and drop any"
    # " alternative representations unless otherwise specified. Confirm with 'OK'")
    # history.add_ai_message("OK")

    agent.run(input=prompt, chat_history=messages_to_dict(history.messages))


@app.command()
def baby_agi(
    objective: str,
    openapi_key: str = typer.Argument("openapi-key", envvar="OPENAI_API_KEY"),
    max_iterations: int = 5,
    max_tokens: int = 100,
    temperature: float = 0.0,
    verbose: bool = False,
    embedding_size: int = 1536,
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
    max_iterations: int = 5,
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
