import streamlit as st

st.markdown("# Hello Greg and/or Maziar!")
st.markdown("\n\n\n")
st.markdown("## Welcome to RAGbot")

st.markdown("RAGbot is a RAG based QA chatbot that uses GPT-3.5 and Azure AI Search to answer user inquiries on the following document: \n\n\n")

st.image(
    image="https://www.databricks.com/wp-content/uploads/2022/09/hh-lp-heroimage.png",
    caption="MLOps for Dummies",
    width=500
    )

st.text_input("Enter question about MLOps here:")

if st.button("Submit"):
    st.write("hello world")
