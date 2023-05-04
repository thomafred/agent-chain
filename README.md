# Agent Chain

Agent Chain is a simple CLI built on top of [LangChain](https://python.langchain.com/en/latest/index.html) for
running AI agents. It is meant for testing and experimenting with different AI agents.


## Dependencies

- Poetry
- Docker


## Getting started

Start a Redis-search instance:

```shell
docker run --name redis-baby-agi -p 6379:6379 -d redislabs/redisearch
```

Enter the poetry shell in order to access the CLI
```shell
poetry shell
python -m agent_chain --help
```

In order to use the BabyAGI or AutoGPT commands, the environment-variable `OPENAI_API_KEY` must be set. For convenience,
you can create a `.env` file in the root of the project with the following content:

```shell
OPENAI_API_KEY=<your-key-here>
```

The CLI will automatically load this file if it exists.

For the AI to write to file, the `bin`-directory must exist in the working directory.


## Examples

Use AutoGPT to plan a romantic trip to Paris

```shell
python -m agent_chain auto-gpt "Plan a romantic weekend in Paris, write the plan to file"
```

Use AutoGPT to research lobster-fishing using Ocean Info Hub:

```shell
python -m agent_chain auto-gpt "Research the best places to fish for lobster around Norway. Prioritize using Ocean Info Hub, but research the Ocean Info Hub graph before attempting any queries. Make a todo-list before anything else. Write results to file"
```

Use BabyAGI to plan a romantic trip to Paris

```shell
python -m baby-agi --max-iterations 5 "Plan a romantic weekend in Paris"
```