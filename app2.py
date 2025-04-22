import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
# from dotenv import load_dotenv # Keep if you prefer .env for local dev outside Streamlit

# --- Configuration ---
# Use Streamlit secrets for API key management
# Create a .streamlit/secrets.toml file with:
# GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not GOOGLE_API_KEY:
        st.error("Google API Key not found in Streamlit secrets!")
        st.stop()
except KeyError:
    st.error("GOOGLE_API_KEY not set in Streamlit secrets! Please create .streamlit/secrets.toml")
    st.info("See https://docs.streamlit.io/library/advanced-features/secrets-management")
    st.stop()

# Optional: If you want to use .env for local dev, uncomment below
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY environment variable not set.")
#     st.stop()

# Configure the Gemini API (this might not be strictly necessary when using Langchain integrations)
# import google.generativeai as genai
# genai.configure(api_key=GOOGLE_API_KEY)


# --- HTML Templates --- (Simplified for demonstration)
css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 15%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""


# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    if not pdf_docs:
        return text
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF '{pdf.name}': {e}")
            # Optionally skip the file or handle differently
            # return None # Indicate failure if needed
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    if not text:
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, # Overlap helps maintain context between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks using Gemini embeddings."""
    if not text_chunks:
        st.warning("No text chunks found to create a vector store.")
        return None
    try:
        # Use the official Langchain Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        # Potentially log the error for debugging
        # print(f"Vector store creation error: {e}")
        return None

def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain."""
    if not vectorstore:
        return None
    try:
        # Use the official Langchain Google Generative AI Chat Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7, # Adjust creativity (0=deterministic, 1=max creativity)
            convert_system_message_to_human=True # Important for some chains/models
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True # Return Langchain message objects
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            # You might want to customize the prompt if needed:
            # combine_docs_chain_kwargs={"prompt": your_custom_prompt}
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None


def handle_userinput(user_question):
    """Processes user input, gets response from chain, and updates chat history."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF files first.")
        return

    # Ensure chat_history exists in session state
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    try:
        # Invoke the chain - it uses memory implicitly
        response = st.session_state.conversation({'question': user_question})

        # The chain updates the memory object automatically.
        # 'chat_history' in the response might contain the history *up to that point*
        # but the primary source is the memory object within the chain.
        # For display, we retrieve the latest history from the memory buffer directly.
        # st.session_state.chat_history = response['chat_history'] # Use memory instead

        # Retrieve the full history from memory after the call
        st.session_state.chat_history = st.session_state.conversation.memory.buffer_as_messages

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage): # Check message type
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage): # Check message type
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            # You might add handling for other message types if needed

    except Exception as e:
        st.error(f"An error occurred during conversation: {e}")
        # Optionally add the error to the chat display or log it
        # st.write(bot_template.replace("{{MSG}}", f"Sorry, an error occurred: {e}"), unsafe_allow_html=True)


# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True) # Apply custom CSS

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None # Store Langchain message objects

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:", key="user_question_input")

    # Handle user input submission
    if user_question:
        # Check if the question is not empty or just whitespace before processing
        if user_question.strip():
             handle_userinput(user_question)
             # Clear the input box after submission (optional)
             # st.session_state.user_question_input = "" # This causes issues with rerun, better not clear automatically


    # --- Sidebar for PDF Upload and Processing ---
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type="pdf" # Ensure only PDF files are accepted
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        # 1. Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text:
                             st.warning("Could not extract text from the provided PDF(s). Check file integrity or content.")

                        # 2. Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                             st.warning("No text chunks generated. Ensure PDFs contain extractable text.")

                        # 3. Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # 4. Create conversation chain
                        if vectorstore:
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.chat_history = None # Reset chat history on new processing
                            st.success("Processing complete! Ready to chat.")
                        else:
                             st.error("Failed to create vector store. Cannot proceed.")

                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        st.session_state.conversation = None # Reset on failure
            else:
                st.warning("Please upload at least one PDF file.")

    # --- Display a welcome message or instructions if no chat history ---
    if st.session_state.chat_history is None and st.session_state.conversation is not None:
         st.info("PDFs processed. Ask a question above to start chatting!")
    elif st.session_state.conversation is None:
         st.info("Upload PDF documents and click 'Process' in the sidebar to begin.")


if __name__ == '__main__':
    main()