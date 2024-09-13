from langchain.chat_models import ChatOpenAI
import sys
# import os

# # Add the root directory to the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import OPENAI_API_KEY

filter_llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

def filter_response(original_response: str , query:str) -> str:
    filter_prompt = f"Given the response: '{original_response}', refine & filter the content and answer based on user query {query}'."
    response_message = filter_llm.invoke([{"role": "system", "content": filter_prompt}])
    
    # Access the content directly from the AIMessage object
    refined_response = response_message.content  # Use .content or .text depending on the attribute available
    
    return refined_response
