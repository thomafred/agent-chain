import json
from textwrap import dedent

from langchain.tools import Tool
from pydantic import BaseModel
from SPARQLWrapper import JSON, SPARQLWrapper


class SparQlTool(BaseModel):
    name: str
    """Name/alias of sparql endpoint"""

    description: str
    """Description of sparql endpoint"""

    endpoint: str
    """SPARQL endpoint URL"""

    def query(self, query: str) -> str:
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)

        # Read query results
        results = sparql.query().convert()
        return json.dumps(results, indent=4)

    def as_tool(self) -> Tool:
        return Tool(
            name=f"SPARQL/{self.name}",
            func=self.query,
            description=dedent(
                f"""
                Execute a SPARQL query against {self.name}.

                {self.description}
                """
            ),
        )


dbpedia = SparQlTool(
    name="DBpedia",
    description="DBpedia is a crowd-sourced community effort to extract structured information from Wikipedia and make this information available on the Web. DBpedia allows you to ask sophisticated queries against Wikipedia and to link other datasets on the Web to Wikipedia data.",  # noqa: E501
    endpoint="http://dbpedia.org/sparql",
)

ocean_info_hub = SparQlTool(
    name="Ocean Info Hub",
    description="The Ocean InfoHub is a platform for sharing ocean data and information. It is a joint initiative of the IOC/IODE, the Ocean Best Practices System, and the Ocean Data Standards community.",  # noqa: E501
    endpoint="http://graph.oceaninfohub.org/blazegraph/namespace/oih/sparql",
)
