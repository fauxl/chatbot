import streamlit as st
import os
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load the Hugging Face token
token = os.getenv("API_KEY")

# Memory management functions
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def initialize_embeddings_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load embeddings model '{model_name}': {e}")
        return None

def main():
    # Clear GPU memory at the start
    clear_gpu_memory()

    st.sidebar.title("Select From The List Below: ")
    selection = st.sidebar.radio("GO TO: ", ["Document Embedding", "RAG Chatbot"])

    if selection == "Document Embedding":
        display_document_embedding_page()

    elif selection == "RAG Chatbot":
        display_chatbot_page()

def display_chatbot_page():
    st.title("Multi Source Chatbot")

    # Initialize the LLM Model
    with st.expander("Initialize the LLM Model"):
        st.markdown("""
            Please Insert the Token and Select Vector Store, Temperature, and Maximum Character Length to create the chatbot.

            **NOTE:**
            - **Token:** API Key From Hugging Face.
            - **Temperature:** Controls creativity (0 to 1).
        """)
        with st.form("setting"):
            row_1 = st.columns(3)
            with row_1[0]:
                text = st.text_input("Hugging Face Token (No need to insert)", type='password', value=f"{'*' * len(token)}")

            with row_1[1]:
                llm_model = st.text_input("LLM model", value="GroNLP/gpt2-small-italian")

            with row_1[2]:
                instruct_embeddings = st.text_input("Instruct Embeddings", value="paraphrase-multilingual-MiniLM-L12-v2")

            row_2 = st.columns(3)
            with row_2[0]:
                vector_store_list = os.listdir("vector store/")
                default_choice = vector_store_list.index('naruto_snake') if 'naruto_snake' in vector_store_list else 0
                existing_vector_store = st.selectbox("Vector Store", vector_store_list, default_choice)

            with row_2[1]:
                temperature = st.number_input("Temperature", value=1.0, step=0.1)

            with row_2[2]:
                max_length = st.number_input("Maximum character length", value=1024, step=1)

            create_chatbot = st.form_submit_button("Launch chatbot")

    # Prepare the LLM model
    if create_chatbot:
        st.session_state.conversation = None
        st.session_state.history = []
        st.session_state.source = []

        embeddings_model = initialize_embeddings_model(instruct_embeddings)
        if embeddings_model:
            st.success(f"Embeddings model '{instruct_embeddings}' initialized successfully!")

        else:
            st.error("Failed to initialize embeddings model. Check the model name.")

    # Chat history and input
    if "history" in st.session_state:
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if question := st.chat_input("Ask a question"):
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate an answer
        answer = f"Mock answer for: {question}"  # Replace this with your actual LLM call
        st.session_state.history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

def display_document_embedding_page():
    st.title("Document Embedding Page")
    st.markdown("""
        This page is used to upload documents as the custom knowledge base for the chatbot.

        **NOTE:** If uploading a new file, insert a new vector store name to store it in the vector database.
    """)

    with st.form("document_input"):
        document = st.file_uploader("Knowledge Documents", type=['pdf', 'txt'], accept_multiple_files=True)
        instruct_embeddings = st.text_input("Model Name of the Instruct Embeddings", value="paraphrase-multilingual-MiniLM-L12-v2")
        chunk_size = st.number_input("Chunk Size", value=200, step=1)
        chunk_overlap = st.number_input("Chunk Overlap", value=10, step=1)
        vector_store_list = ["<New>"] + os.listdir("vector store/")
        existing_vector_store = st.selectbox("Vector Store to Merge the Knowledge", vector_store_list)
        new_vs_name = st.text_input("New Vector Store Name", value="new_vector_store_name")
        save_button = st.form_submit_button("Save vector store")

    if save_button:
        if document:
            combined_content = ""
            for file in document:
                try:
                    if file.name.endswith(".pdf"):
                        combined_content += falcon.read_pdf(file)
                    elif file.name.endswith(".txt"):
                        combined_content += falcon.read_txt(file)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

            # Embedding and storing logic (Mocked for now)
            st.success(f"Processed {len(document)} documents successfully!")
        else:
            st.warning("Please upload at least one file.")

if __name__ == "__main__":
    main()
