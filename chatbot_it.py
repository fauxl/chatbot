import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Global variables to avoid repeated loading
model = None
tokenizer = None

# Function to load the model
def load_italian_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_name = "distilgpt2"  # Use a smaller model to reduce memory usage
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Function to clear GPU memory if available
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Generate responses
def generate_italian_response(question, tokenizer, model):
    try:
        # Add a system-level instruction
        system_prompt = "You are a helpful assistant that responds in Italian with accurate and concise answers.\n"
        
        # Limit conversation history to the last 5 exchanges
        MAX_HISTORY_LENGTH = 5
        recent_history = st.session_state.history[-MAX_HISTORY_LENGTH:]
        conversation_history = system_prompt
        for message in recent_history:
            conversation_history += f"{message['role']}: {message['content']}\n"
        conversation_history += f"user: {question}\nassistant:"

        # Tokenize and ensure input length is within model limits
        inputs = tokenizer(conversation_history, return_tensors="pt", truncation=True, max_length=1024)

        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,          # Limit the response length
            temperature=0.7,         # Adjust creativity
            top_p=0.9,               # Use nucleus sampling
            repetition_penalty=1.2,  # Penalize repeated phrases
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Clear memory after generation
        clear_memory()

        # Validate response to avoid repetition
        if response.lower() == question.lower():
            response = "Non sono sicuro di come rispondere a questa domanda."

        return response
    except Exception as e:
        print("Error during generation:", e)
        return "Mi dispiace, si è verificato un errore durante la generazione della risposta."

# Display the chatbot page
def display_chatbot_page():
    st.title("Chatbot Multilingua")
    st.markdown("Questo chatbot supporta l'italiano e altre lingue. Inizia facendo una domanda qui sotto!")

    # Load the LLM model
    tokenizer, model = load_italian_model()

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    question = st.chat_input("Fai una domanda")

    if question:
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        response = generate_italian_response(question, tokenizer, model)
        st.session_state.history.append({"role": "assistant", "content": response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

    # Display the history in a separate section
    with st.expander("Cronologia della conversazione"):
        for message in st.session_state.history:
            role = "Utente" if message["role"] == "user" else "Assistente"
            st.write(f"**{role}:** {message['content']}")

# Display the document embedding page
def display_document_embedding_page():
    st.title("Document Embedding Page")
    st.markdown("""Questa pagina permette di caricare documenti come base di conoscenza personalizzata per il chatbot.""")

    with st.form("document_input"):
        documents = st.file_uploader(
            "Carica i documenti", type=['pdf', 'txt'], accept_multiple_files=True,
            help="Supporta file in formato PDF e TXT."
        )

        row_1 = st.columns([2, 1, 1])
        with row_1[0]:
            instruct_embeddings = st.text_input(
                "Nome del modello di embedding", value="sentence-transformers/distiluse-base-multilingual-cased-v1"
            )
        with row_1[1]:
            chunk_size = st.number_input(
                "Dimensione dei blocchi (chunk size)", value=200, min_value=0, step=1,
            )
        with row_1[2]:
            chunk_overlap = st.number_input(
                "Sovrapposizione dei blocchi", value=10, min_value=0, step=1,
                help="Deve essere inferiore alla dimensione dei blocchi."
            )

        row_2 = st.columns(2)
        with row_2[0]:
            vector_store_list = os.listdir("vector store/") if os.path.exists("vector store/") else []
            vector_store_list = ["<Nuovo Database>"] + vector_store_list
            selected_store = st.selectbox(
                "Seleziona o crea un database vettoriale", vector_store_list
            )

        with row_2[1]:
            new_store_name = st.text_input(
                "Nome del nuovo database", value="", 
                help="Inserisci un nome se hai scelto di creare un nuovo database."
            )

        save_button = st.form_submit_button("Salva i documenti")

    if save_button:
        if documents:
            # Placeholder for processing documents (replace with actual logic)
            st.success("Documenti elaborati e salvati con successo!")
        else:
            st.error("Devi caricare almeno un documento.")

# Main function
def main():
    st.sidebar.title("Seleziona un'opzione")
    selection = st.sidebar.radio("Vai a:", ["Chatbot Multilingua", "Document Embedding"])

    if selection == "Chatbot Multilingua":
        display_chatbot_page()
    elif selection == "Document Embedding":
        display_document_embedding_page()

if __name__ == "__main__":
    main()
