import gradio as gr
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from deep_translator import GoogleTranslator
from duckduckgo_search import DDGS
import wikipedia
import requests
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_core.tools import StructuredTool
from datetime import datetime

# Define the DuckDuckGo search tool
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

# Define the Wikipedia search tool
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

# Define the SQL Database Connection
DATABASE_URL = ""  # Replace with your actual database URL
engine = create_engine(DATABASE_URL)

# Initialize the SQLDatabase toolkit
db = SQLDatabase(engine)

# Create a tool for running SQL queries
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

# Define the weather tool
class WeatherInput(BaseModel):
    location: str = Field(description="The location to get the weather for")
    specific_info: str = Field(default=None, description="Specific information to filter (e.g., 'humidity', 'temperature', etc.)")

def get_weather(location: str, specific_info: str = None) -> str:
    api_key = ''
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            "description": data['weather'][0]['description'].capitalize(),
            "temperature": f"{data['main']['temp']}°C",
            "feels like": f"{data['main']['feels_like']}°C",
            "humidity": f"{data['main']['humidity']}%",
            "wind speed": f"{data['wind']['speed']} m/s",
            "sunrise": datetime.utcfromtimestamp(data['sys']['sunrise']).strftime('%Y-%m-%d %H:%M:%S UTC'),
            "sunset": datetime.utcfromtimestamp(data['sys']['sunset']).strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        if specific_info:
            specific_info = specific_info.lower()
            if specific_info in weather_info:
                return f"The {specific_info} in {location} is {weather_info[specific_info]}."
            else:
                return f"Sorry, I could not find the {specific_info} information for {location}."

        weather_report = (
            f"Weather in {location}:\n"
            f"- Description: {weather_info['description']}\n"
            f"- Temperature: {weather_info['temperature']} (feels like {weather_info['feels like']})\n"
            f"- Humidity: {weather_info['humidity']}\n"
            f"- Wind Speed: {weather_info['wind speed']}\n"
            f"- Sunrise: {weather_info['sunrise']}\n"
            f"- Sunset: {weather_info['sunset']}"
        )
        return weather_report
    else:
        return f"Could not retrieve weather data for {location}"
    
def get_weather_tool(location: str, specific_info: str = None) -> str:
    return get_weather(location, specific_info)

weather_tool = StructuredTool.from_function(
    func=get_weather_tool,
    name="weather_tool",
    description="Get the current weather for a specific location, with an option to filter specific information like humidity, temperature, wind speed, etc.",
    args_schema=WeatherInput,
    return_direct=True,
)

# Define the translation tool
class TranslationInput(BaseModel):
    text: str = Field(description="The text to translate")
    dest_lang: str = Field(description="The target language code (e.g., 'en' for English)")

def translate_text(text: str, dest_lang: str) -> str:
    translator = GoogleTranslator(target=dest_lang)
    translated_text = translator.translate(text)
    return translated_text

translate_text_tool = StructuredTool.from_function(
    func=translate_text,
    name="translate_text_tool",
    description="Translate text to a specified language",
    args_schema=TranslationInput,
    return_direct=True,
)

# Set up the LLM and tools
llm = ChatOpenAI(temperature=0, api_key='')
tools = [duckduckgo_tool, wikipedia_tool, sql_query_toolkit, weather_tool, translate_text_tool]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt for the primary agent
qa_system_prompt = """You are a helpful assistant. You can use the following tools: DuckDuckGo for internet searches, Wikipedia for finding information from Wikipedia, sql_query_toolkit for querying a database, 

weather_tool to get current weather data or forecasts, and translate_text_tool for translation or not use any of these tools at all.

Determine whether a tool should be used or not based on the query  .

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

# Define the secondary LLM for filtering
filter_llm = ChatOpenAI(temperature=0, api_key='')

def filter_response(original_response: str, filter_instructions: str) -> str:
    filter_prompt = f"Given the response: '{original_response}', refine the content based on the following instruction: '{filter_instructions}'."
    response_message = filter_llm.invoke([{"role": "system", "content": filter_prompt}])
    
    # Access the content directly from the AIMessage object
    refined_response = response_message.content  # Use .content or .text depending on the attribute available
    
    return refined_response


# Function to handle chatbot interactions with filtering
def chatbot_interface(query, filter_instructions, history):
    if history is None:
        history = []
    response = primary_agent_executor.invoke({"input": query})
    filtered_response = filter_response(response['output'], filter_instructions)
    history.append((query, filtered_response))
    return history, history


# Update the Gradio interface
chatbot_ui = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(label="Your Query"), gr.Textbox(label="Filter Instructions"), gr.State()],
    outputs=[gr.Chatbot(label="Chat History"), gr.State()],
    title="Multi-Tool Chatbot Assistant with Response Filtering",
    description="Interact with the chatbot and ask questions. The assistant will use appropriate tools to answer your queries and refine the responses based on your instructions.",
)

# Launch the app
chatbot_ui.launch()
