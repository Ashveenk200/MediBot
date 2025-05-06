# MediBot

I developed MediBot, an innovative Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate and contextually relevant responses based on medical documents. This project showcases my expertise in building AI-driven applications using Streamlit for the user interface, LangChain for RAG implementation, HuggingFace for advanced language models (Mistral-7B-Instruct-v0.3), and FAISS for efficient vector storage and retrieval. By integrating sentence-transformers/all-MiniLM-L6-v2 for embeddings and leveraging PyTorch for GPU-accelerated processing, I ensured optimal performance and scalability. The bot features a custom prompt template for precise answers and robust error handling, demonstrating my ability to create production-ready solutions. MediBot is hosted on my GitHub repository, reflecting my proficiency in Python, machine learning, and modern AI frameworks, as well as my commitment to advancing healthcare accessibility through technology.


Tools and Technologies Used in MediBot Project:

Python: Core programming language for development.
Streamlit: Framework for building the interactive web-based user interface.
LangChain: Library for implementing Retrieval-Augmented Generation (RAG) and managing LLM workflows.
HuggingFace Transformers: For accessing and deploying the Mistral-7B-Instruct-v0.3 language model and sentence-transformers/all-MiniLM-L6-v2 for embeddings.
FAISS: Vector store for efficient similarity search and document retrieval.
PyTorch: Backend for GPU-accelerated model inference and processing.
HuggingFace Endpoint: Cloud-based API for accessing large language models.
HuggingFace Pipeline: Local fallback for text generation when cloud endpoint fails.
dotenv: For secure management of environment variables (e.g., HF_TOKEN).
