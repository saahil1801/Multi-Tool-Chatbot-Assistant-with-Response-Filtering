import gradio as gr
from llm.primary_agent import primary_agent_executor
from llm.filter_llm import filter_response

def chatbot_interface(query, filter_instructions, history):
    if history is None:
        history = []
    response = primary_agent_executor.invoke({"input": query})
    filtered_response = filter_response(response['output'], filter_instructions)
    history.append((query, filtered_response))
    return history, history

chatbot_ui = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(label="Your Query"), gr.Textbox(label="Filter Instructions"), gr.State()],
    outputs=[gr.Chatbot(label="Chat History"), gr.State()],
    title="Multi-Tool Chatbot Assistant with Response Filtering",
    description="Interact with the chatbot and ask questions. The assistant will use appropriate tools to answer your queries and refine the responses based on your instructions.",
)

# export PYTHONPATH=$PYTHONPATH:/Users/saahil/Desktop/Coding_Projects/LLMS/ToolCalling
