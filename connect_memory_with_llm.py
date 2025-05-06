import os
import torch
try:
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    print(f"Import error: {e}. Please ensure langchain and related packages are installed.")
    exit(1)
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please set it in the environment variables.")
print("HF_TOKEN:", HF_TOKEN)  # Debug: Verify token

# HF model repos
primary_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"  # Gated model
fallback_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Non-gated fallback

# Step 1: Load LLM from HuggingFace
def load_llm(repo_id):
    try:
        # Attempt to use HuggingFaceEndpoint
        print(f"Attempting to load HuggingFaceEndpoint with repo: {repo_id}")
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.9,
            max_new_tokens=512,
            huggingfacehub_api_token=HF_TOKEN
        )
        print("Loaded HuggingsFaceEndpoint successfully.")
        return llm
    except Exception as e:
        print(f"Failed to load HuggingFaceEndpoint: {e}")
        print("Falling back to HuggingFacePipeline (local model)...")
        
        # Fallback to HuggingFacePipeline
        try:
            device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
            pipe = pipeline(
                "text-generation",
                model=repo_id,
                tokenizer=repo_id,
                max_new_tokens=512,
                temperature=0.7,
                device=device,
                model_kwargs={"load_in_4bit": True}  # Enable quantization for lower memory usage
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            print("Loaded HuggingFacePipeline successfully.")
            return llm
        except Exception as e:
            print(f"Failed to load HuggingFacePipeline: {e}")
            raise

# Step 2: Define custom prompt
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
"""

def set_custom_prompt(template_str):
    try:
        prompt = PromptTemplate(
            template=template_str,
            input_variables=["context", "question"]
        )
        return prompt
    except Exception as e:
        print(f"Error setting custom prompt: {e}")
        raise

# Step 3: Load vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("Loaded FAISS vector store successfully.")
except Exception as e:
    print(f"Error loading FAISS vector store: {e}")
    raise

# Step 4: Build QA chain
try:
    # Try primary model, fallback to alternative if it fails
    try:
        llm = load_llm(primary_repo_id)
    except Exception as e:
        print(f"Primary model {primary_repo_id} failed: {e}")
        print(f"Trying fallback model {fallback_repo_id}...")
        llm = load_llm(fallback_repo_id)
    
    qa_chain = RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt(custom_prompt_template),
        },
    )
    print("Built QA chain successfully.")
except Exception as e:
    print(f"Error building QA chain: {e}")
    raise

# Step 5: Get user query
try:
    user_query = input("Write your query: ")
    response = qa_chain.invoke({"query": user_query})  # Updated to use invoke
except Exception as e:
    print(f"Error processing query: {e}")
    raise

# Step 6: Output response
print("\nResponse:\n", response["result"])
print("\nSource Documents:")
for doc in response["source_documents"]:
    print(doc.metadata, "\n", doc.page_content, "\n")