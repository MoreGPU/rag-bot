import os
import json
import argparse
import streamlit as st

from config import *
from utils.azure_utils import CustomAzureSearch
from utils.openai_utils import RAGBot, OpenAIChat

def setup_streamlit_ui():
    """Configures streamlit user interface"""
    st.markdown("# Hello Greg and/or Maziar and/or whoever!")
    st.markdown("\n\n\n")
    st.markdown("## Welcome to RAGbot")
    st.markdown("RAGbot is a RAG based QA chatbot that uses GPT-3.5-turbo and Azure AI Search to answer user inquiries on the following document: \n\n\n")
    st.image(
        image="https://www.databricks.com/wp-content/uploads/2022/09/hh-lp-heroimage.png",
        caption="MLOps for Dummies",
        width=500
        )
    user_input = st.text_input(
        "Enter question about MLOps here:",
        "Tell me about MLOps!")
    return user_input 

def initialize_model():
    """Initializes the RAG model with necessary components"""
    custom_search = CustomAzureSearch(
        searchservice=searchservice,
        searchkey=searchkey,
        index_name=index_name,
        number_results_to_return=3,
        number_near_neighbors=3,
        embedding_field_name="embedding",
        openai_api_key=openai_api_key,
        embedding_model="text-embedding-ada-002" 
    )

    system_message = "You are an assistant here to answer questions about the ebook: 'MLOps for Dummies: Databricks Special Edition'" 
    openai_chat = OpenAIChat(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        system_message=system_message,
        n=1,
        temperature=0.2
    )

    model = RAGBot(
        fields_to_return=["id", "sourcepage", "content"],
        azure_search_object=custom_search,
        openai_chat_object=openai_chat
    )
    
    return model 

def display_model_output(model, user_input):
    """Displays the model output on the Streamlit UI"""
    response, memory = model(query=user_input)
    
    st.markdown("## Model response:")
    st.write(response)
    st.write("\n\n\n")
    
    st.markdown("## Context used:")
    st.markdown("The following documents were returned from the vector search.")
    context_dicts = model.context_dicts
    st.json(json.dumps(context_dicts, indent=4))
    
    st.markdown("## API Messages:")
    st.markdown("(Memory not yet implemented)")
    st.json(json.dumps(memory, indent=4))


if __name__ == "__main__":
    user_input = setup_streamlit_ui()
    model = initialize_model()
    
    if st.button('submt'):
        st.write("Waiting for response (streaming not yet implemented)...")
        display_model_output(model, user_input)