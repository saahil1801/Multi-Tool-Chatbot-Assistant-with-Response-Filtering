from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
import wikipedia

class WikipediaSearchInput(BaseModel):
    query: str = Field(description="The search query for Wikipedia")

def wikipedia_search(query: str) -> str:
    try:
        page = wikipedia.page(query)
        summary = wikipedia.summary(query, sentences=2)
        return f"Wikipedia article: {page.title}\nSummary: {summary}\nURL: {page.url}"
    except:
        return f"No Wikipedia page found for the query '{query}'"

wikipedia_tool = StructuredTool.from_function(
    func=wikipedia_search,
    name="wikipedia_search_tool",
    description="Perform a search using Wikipedia",
    args_schema=WikipediaSearchInput,
    return_direct=True,
)
