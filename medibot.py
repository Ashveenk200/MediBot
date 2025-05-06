import os
import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import pipeline

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

def set_custom_prompt(custom_prompt_template):
    try:
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )
        return prompt
    except Exception as e:
        st.error(f"Error setting custom prompt: {e}")

def load_llm(repo_id, HF_TOKEN):
    try:
        
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.9,
            max_new_tokens=512,
            huggingfacehub_api_token=HF_TOKEN
        )
        return llm
    except Exception as e:
        st.warning(f"Failed to load HuggingFaceEndpoint: {e}. Trying local pipeline.")
        try:
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline(
                "text-generation",
                model=repo_id,
                tokenizer=repo_id,
                max_new_tokens=512,
                temperature=0.7,
                device=device,
                model_kwargs={"load_in_4bit": True}
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            st.error(f"Failed to load local model: {e}")
            raise

def main():
    st.set_page_config(page_title="Document ChatBot", layout="wide")
    st.title("MediBot Using RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask me anything about the document:")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        custom_prompt_template = """
        Use the piece of information provided in the context to answer the user's question.
        If you don't know the answer, say "I don't know". Don't try to make up an answer.
        Don't provide anything outside the context.

        Context:
        {context}

        Question:
        {question}

        Start the answer directly. No small talk please.
        You are the master of uploaded documents.
        if the response is in point wise then arrange in the following format:
        1
        2
        3 etc..
        """

        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        try:
            llm = load_llm(repo_id, HF_TOKEN)
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
            )

            response = qa_chain({"query": user_prompt})
            result = response["result"]
            source_docs = response["source_documents"]

            sources = "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in source_docs])
            result_to_show = f"{result}"

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
