import streamlit as st
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()


# Function to read PDF files
def read_pdf(file):
    document = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            document += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return document


# Function to read TXT files
def read_txt(file):
    try:
        document = file.read().decode("utf-8")
        document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        document = ""
    return document


# Function to split documents into chunks
def split_doc(document, chunk_size, chunk_overlap):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split = splitter.split_text(document)
        split = splitter.create_documents(split)
    except Exception as e:
        st.error(f"Error splitting document: {e}")
        split = []
    return split


# Function to store embeddings and merge vector stores
def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    try:
        # Initialize embeddings
        instructor_embeddings = HuggingFaceEmbeddings(
            model_name="dbmdz/camembert-base-italian",
            model_kwargs={'device': 'cpu'}
        )

        # Create new FAISS database from documents
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs is True:
            db.save_local(f"vector store/{new_vs_name}")
        else:
            # Load existing vector store and merge
            load_db = FAISS.load_local(
                f"vector store/{existing_vector_store}",
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            load_db.merge_from(db)
            load_db.save_local(f"vector store/{new_vs_name}")

        st.success("The document has been saved successfully.")
    except Exception as e:
        st.error(f"Error storing embeddings: {e}")


# Function to prepare Retrieval-Augmented Generation (RAG) model
def prepare_rag_llm(token, vector_store_list, temperature, max_length):
    try:
        # Initialize embeddings
        instructor_embeddings = HuggingFaceEmbeddings(
            model_name="dbmdz/camembert-base-italian",
            model_kwargs={'device': 'cpu'}
        )

        # Load FAISS vector store
        loaded_db = FAISS.load_local(
            f"vector store/{vector_store_list}",
            instructor_embeddings,
            allow_dangerous_deserialization=True
        )

        # Initialize the LLM from Hugging Face Hub
        llm = HuggingFaceHub(
            repo_id='GroNLP/gpt2-small-italian',
            model_kwargs={"temperature": temperature, "max_length": max_length},
            huggingfacehub_api_token=token
        )

        # Set up memory for conversation
        memory = ConversationBufferWindowMemory(
            k=2,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        # Create the conversational chain
        qa_conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            memory=memory,
        )

        return qa_conversation
    except Exception as e:
        st.error(f"Error preparing RAG model: {e}")
        return None


# Function to generate answers using the RAG model
def generate_answer(question, token):
    try:
        if not token:
            return "Insert the Hugging Face token", ["no source"]

        question = "Rispondi in italiano: " + question
        response = st.session_state.conversation({"question": question})

        # Extract answer and sources
        answer = response.get("answer", "An error has occurred").split("Helpful Answer:")[-1].strip()
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]

        return answer, doc_source
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "An error has occurred", ["no source"]
