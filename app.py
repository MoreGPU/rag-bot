import json
import streamlit as st

# configure document 
st.markdown("# Hello Greg and/or Maziar!")
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

#############################################
# initialize RAGModel
#############################################
import sys
sys.path.append('./utils/')
from config import *
from azure_utils import CustomAzureSearch
from openai_utils import RAGBot, OpenAIChat

# initialze Azure AI Search object 
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

# initialize OpenAI Chat object 
system_message = "You are an assistant here to answer questions about the ebook: 'MLOps for Dummies: Databricks Special Edition'" 
openai_chat = OpenAIChat(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    system_message=system_message,
    n=1,
    temperature=0.2
)

# initalize RAG model
model = RAGBot(
    fields_to_return=["id", "sourcepage", "content"],
    azure_search_object=custom_search,
    openai_chat_object=openai_chat
)

if st.button("Submit"):
    st.write("Waiting for response...")
    response, memory = model(query=user_input)
    
    st.markdown("### Model response:")   
    st.write(response)
    st.write("\n\n\n")
    
    st.markdown("### Context used:")
    context_dicts = model.context_dicts
    pretty_json = json.dumps(context_dicts, indent=4)
    st.json(pretty_json)
    
    st.markdown("### API messages:")
    st.markdown("(Memory not yet implemented)")
    pretty_json = json.dumps(memory, indent=4)
    st.json(pretty_json)