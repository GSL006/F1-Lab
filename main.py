from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import OpenAI
import openai
import streamlit as st

# Set your OpenAI API key
openai.api_key = " " #Enter your respective openai api key

def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                                                              system_prompt="You are an expert in analyzing student data of their particular University, Assume all input prompts to be with respect to the input data, Don't answer anything apart from educational related prompt"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

def generate_response(chat_engine, prompt):
    try:
        response = chat_engine.chat(prompt)
        return response.response
    except openai.OpenAIError as e:
        if e.status_code == 429:
            st.error("Error: You have exceeded your OpenAI API quota. Please check your plan and billing details.")
        else:
            st.error(f"OpenAI Error: {e}")
        return None

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Initialize chat history
chat_history = []

st.title("F1 GPT")

# Add options in the sidebar
st.sidebar.title("Options")
display_history = st.sidebar.checkbox("Display Chat History", value=True)
clear_history = st.sidebar.button("Clear Chat History")

# Chat input
prompt = st.text_input("It's Lights Out and Away We Go, How may I Assist you Today?")

if st.button("Generate") and prompt:
    # Generate a response
    response = generate_response(chat_engine, prompt)

    if response:
        # Add the conversation to the chat history
        chat_history.append({'User': prompt, 'Assistant': response})

        # Display the response
        st.write("Assistant: ", response)

        # Display chat history if the option is selected
        if display_history:
            st.subheader("Chat History")
            for entry in chat_history:
                st.write(f"User: {entry['User']}")
                st.write(f"Assistant: {entry['Assistant']}")
                st.write("---")

if clear_history:
    chat_history = []
    st.sidebar.success("Chat History Cleared!")
