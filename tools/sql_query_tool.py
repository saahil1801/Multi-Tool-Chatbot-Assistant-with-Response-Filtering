from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.tools import StructuredTool
# import sys
# import os

# # Add the root directory to the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATABASE_URL

# Define SQL Database Connection
engine = create_engine(DATABASE_URL)
db = SQLDatabase(engine)

class SQLQueryInput(BaseModel):
    query: str = Field(description="SQL query to execute against the database")

def sql_query_tool(query: str) -> str:
    return db.run(query)

sql_query_toolkit = StructuredTool.from_function(
    func=sql_query_tool,
    name="sql_query_toolkit",
    description="Run SQL queries against a database",
    args_schema=SQLQueryInput,
    return_direct=True,
)

