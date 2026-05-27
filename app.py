import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document

# 1. Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("💬 Normal AI Chatbot (+ ChromaDB)")
st.write(".")

# 2. Initialize Chroma Database with some default knowledge
@st.cache_resource
def setup_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Custom vector data
    docs = [
        Document(page_content="Groq developed the LPU (Language Processing Unit) for high-speed AI inference."),
        Document(page_content="Chroma is an AI-native open-source vector database."),
        Document(page_content="Render is a unified cloud to build and run all your apps and websites.")
    ]
    
    vector_db = Chroma.from_documents(docs, embeddings)
    return vector_db

vector_db = setup_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 3. Setup Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# 4. A much more natural, conversational prompt
prompt = ChatPromptTemplate.from_template(
    """You are a helpful, friendly, and intelligent AI Assistant. 
    
    Feel free to greet the user, engage in normal chitchat, or answer general questions using your own knowledge base.
    
    However, if the user asks something specifically related to Groq, Chroma, or Render, use the provided context below to inform your answer:
    
    <context>
    {context}
    </context>

    User's Message: {input}
    Your Response:"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 5. Maintaining a continuous chat history on screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field at the bottom (like a real chatbot)
if user_input := st.chat_input("Say hello or ask something..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})