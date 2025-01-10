import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import os
import torch

# Global variables for model and document content
model = None
tokenizer = None
document_content = ""

# Function to load the model
def load_italian_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_name = "distilgpt2"  # Use a smaller model for better performance
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Function to clear GPU memory
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return "Impossibile estrarre il testo dal documento."

# Generate responses
def generate_italian_response(question, tokenizer, model, document_content):
    try:
        # Add a system-level instruction
        system_prompt = "You are a helpful assistant that responds in Italian with accurate and concise answers.\n"
        
        # Append document content as context
        if document_content:
            document_context = f"Here is relevant information from the document:\n{document_content}\n"
        else:
            document_context = "No document content provided.\n"

        # Limit conversation history to the last 5 exchanges
        MAX_HISTORY_LENGTH = 5
        recent_history = st.session_state.history[-MAX_HISTORY_LENGTH:]
        conversation_history = system_prompt + document_context
        for message in recent_history:
            conversation_history += f"{message['role']}: {message['content']}\n"
        conversation_history += f"user: {question}\nassistant:"

        # Tokenize and ensure input length is within model limits
        inputs = tokenizer(conversation_history, return_tensors="pt", truncation=True, max_length=1024)

        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Clear memory after generation
        clear_memory()

        return response
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "Mi dispiace, si è verificato un errore durante la generazione della risposta."

# Display chatbot page
def display_chatbot_page():
    global document_content

    st.title("Chatbot Multilingua con Conoscenza da Documenti")
    st.markdown("Questo chatbot può rispondere in base ai documenti caricati. Fai una domanda qui sotto!")

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

        # Generate response using the document content
        response = generate_italian_response(question, tokenizer, model, document_content)
        st.session_state.history.append({"role": "assistant", "content": response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

    # Display the history in a separate section
    with st.expander("Cronologia della conversazione"):
        for message in st.session_state.history:
            role = "Utente" if message["role"] == "user" else "Assistente"
            st.write(f"**{role}:** {message['content']}")

# Display document embedding page
def display_document_embedding_page():
    global document_content

    st.title("Caricamento Documenti")
    st.markdown("Carica un documento per aggiungere conoscenza al chatbot.")

    with st.form("document_input"):
        document = st.file_uploader(
            "Carica un documento (PDF)", type=['pdf'], help="Carica un file PDF contenente il testo."
        )

        submit_button = st.form_submit_button("Elabora Documento")

    if submit_button:
        if document:
            document_content = extract_text_from_pdf(document)
            st.success("Documento elaborato con successo!")
        else:
            st.error("Devi caricare un documento valido.")

# Main function
def main():
    st.sidebar.title("Seleziona un'opzione")
    selection = st.sidebar.radio("Vai a:", ["Chatbot Multilingua", "Caricamento Documenti"])

    if selection == "Chatbot Multilingua":
        display_chatbot_page()
    elif selection == "Caricamento Documenti":
        display_document_embedding_page()

if __name__ == "__main__":
    main()
