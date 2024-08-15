from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# import sys
# import os

# # Add the root directory to the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import OPENAI_API_KEY

from tools.duckduckgo_tool import duckduckgo_tool
from tools.wikipedia_tool import wikipedia_tool
from tools.sql_query_tool import sql_query_toolkit
from tools.weather_tool import weather_tool
from tools.translate_tool import translate_text_tool

# Set up the LLM and tools
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)
tools = [duckduckgo_tool, wikipedia_tool, sql_query_toolkit, weather_tool, translate_text_tool]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt for the primary agent
qa_system_prompt = """You are a helpful assistant. You can use the following tools: DuckDuckGo for internet searches, Wikipedia for finding information from Wikipedia, sql_query_toolkit for querying a database, 
weather_tool to get current weather data or forecasts, and translate_text_tool for translation or not use any of these tools at all.
Determine whether a tool should be used or not based on the query.
When dealing with weather reports, display and filter the information the user specifically wants.
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ]
)

# Create the primary agent
primary_agent = create_openai_functions_agent(llm, tools, prompt=qa_prompt)

# Initialize the primary agent executor
primary_agent_executor = AgentExecutor(agent=primary_agent, tools=tools, verbose=True, return_intermediate_steps=True, memory=memory)
