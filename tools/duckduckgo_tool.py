from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from duckduckgo_search import DDGS

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(description="The search query for DuckDuckGo")

def duckduckgo_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No results found for '{query}'."
        formatted_results = "\n".join([f"- {result['title']}: {result['body']}" for result in results])
        return formatted_results
    except Exception as e:
        return f"An error occurred while searching for '{query}': {str(e)}"

duckduckgo_tool = StructuredTool.from_function(
    func=duckduckgo_search,
    name="duckduckgo_search_tool",
    description="Perform a search using DuckDuckGo",
    args_schema=DuckDuckGoSearchInput,
    return_direct=True,
)
