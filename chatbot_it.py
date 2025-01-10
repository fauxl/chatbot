import streamlit as st
import os
import falcon
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# Configura il modello di base per l'italiano
def configure_model_for_italian():
    # Usa un modello specifico per l'italiano (puoi cambiare il modello se necessario)
    model_name = "Helsinki-NLP/opus-mt-en-it"  # Traduzione da inglese a italiano
    italian_model = pipeline("translation", model=model_name)
    return italian_model

def main():
    st.sidebar.title("Seleziona un'opzione")
    selection = st.sidebar.radio("Vai a:", ["Chatbot Multilingua", "Document Embedding"])

    if selection == "Chatbot Multilingua":
        display_chatbot_page()

    elif selection == "Document Embedding":
        display_document_embedding_page()

# Load the LLM model once
@st.cache_resource
def load_italian_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_italian_response(question, tokenizer, model):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def display_chatbot_page():
    st.title("Chatbot Multilingua")

    st.markdown("""
    Questo chatbot supporta l'italiano e altre lingue. Inizia facendo una domanda qui sotto!
    """)

    # Load the LLM model
    tokenizer, model = load_italian_model()

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input dell'utente
    question = st.chat_input("Fai una domanda")

    if question:
        # Salva la domanda dell'utente nella cronologia
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate a meaningful response
        response = generate_italian_response(question, tokenizer, model)

        # Mostra la risposta
        with st.chat_message("assistant"):
            st.markdown(response)

        # Salva la risposta del chatbot nella cronologia
        st.session_state.history.append({"role": "assistant", "content": response})

    # Visualizza la cronologia
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_document_embedding_page():
    st.title("Document Embedding Page")
    st.markdown("""This page is used to upload the documents as the custom knowledge base for the chatbot.
                  **NOTE:** If you are uploading a new file (for the first time) please insert a new vector store name to store it in vector database
                """)

    with st.form("document_input"):
        
        document = st.file_uploader(
            "Knowledge Documents", type=['pdf', 'txt'], help=".pdf or .txt file", accept_multiple_files= True
        )

        row_1 = st.columns([2, 1, 1])
        with row_1[0]:
            instruct_embeddings = st.text_input(
                "Model Name of the Instruct Embeddings", value="sentence-transformers/distiluse-base-multilingual-cased-v1"
            )
        
        with row_1[1]:
            chunk_size = st.number_input(
                "Chunk Size", value=200, min_value=0, step=1,
            )
        
        with row_1[2]:
            chunk_overlap = st.number_input(
                "Chunk Overlap", value=10, min_value=0, step=1,
                help="Lower than chunk size"
            )
        
        row_2 = st.columns(2)
        with row_2[0]:
            # List the existing vector stores
            vector_store_list = os.listdir("vector store/")
            vector_store_list = ["<New>"] + vector_store_list
            
            existing_vector_store = st.selectbox(
                "Vector Store to Merge the Knowledge", vector_store_list,
                help="""
                Which vector store to add the new documents.
                Choose <New> to create a new vector store.
                    """
            )

        with row_2[1]:
            # List the existing vector stores     
            new_vs_name = st.text_input(
                "New Vector Store Name", value="new_vector_store_name",
                help="""
                If choose <New> in the dropdown / multiselect box,
                name the new vector store. Otherwise, fill in the existing vector
                store to merge.
                """
            )

        save_button = st.form_submit_button("Save vector store")

    if save_button:
        if document is not None:
            # Aggregate content of all uploaded files
            combined_content = ""
            for file in document:
                if file.name.endswith(".pdf"):
                    combined_content += falcon.read_pdf(file)
                elif file.name.endswith(".txt"):
                    combined_content += falcon.read_txt(file)
                else:
                    st.error("Check if the uploaded file is .pdf or .txt")

            # Split combined content into chunks
            split = falcon.split_doc(combined_content, chunk_size, chunk_overlap)

            # Check whether to create new vector store
            create_new_vs = None
            if existing_vector_store == "<New>" and new_vs_name != "":
                create_new_vs = True
            elif existing_vector_store != "<New>" and new_vs_name != "":
                create_new_vs = False
            else:
                st.error("Check the 'Vector Store to Merge the Knowledge' and 'New Vector Store Name'")

            # Embeddings and storing
            falcon.embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name)
            print(f'"Document info":{combined_content}')    
            print(f'"Splitted info":{split}')   

        else:
            st.warning("Please upload at least one file.")
    

if __name__ == "__main__":
    main()
