# from langchain_community.chat_models import ChatCohere
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import json
import PyPDF2
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import  hashlib
from functools import lru_cache


st.set_page_config("ChatSDK Fund","💬",layout="wide")

load_dotenv()

# API Keys
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Using Cohere's embed-english-v3.0 embedding model
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")


# For OpenAI's gpt-4.o-turbo llm
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo" openai_api_key=OPENAI_API_KEY)

# For Cohere's command-r llm
# llm = ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")
if 'selected_model' not in st.session_state:
    st.session_state.selected_model="cohere-command-r"

def get_llm():
    if st.session_state.selected_model== "cohere-command-r":
        return ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")
    # elif st.session_state.selected_model == "ollama-llama2":
    #     return Ollama(model="llama2")
    # elif st.session_state.selected_model == "ollama-llama2-uncensored":
    #     return Ollama(model="llama2-uncensored")
    elif st.session_state.selected_model == "ollama-gemma-3-12b":
        return Ollama(model="gemma:12b")
    elif st.session_state.selected_model == "ollama-llama3.1":
        return Ollama(model="llama3.1:8b")
    else:
        return ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")


@lru_cache(maxsize=64)
def cached_query(query_hash, model_name,chunks=5):
    # Use the function to get the LLM
    llm = get_llm()
    retriever = st.session_state.book_docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k":chunks}
    )
    qa =RetrievalQA.from_llm(llm=llm, retriever=retriever,verbose=True)
    return qa({"query":query_hash})["result"]


# For reading PDFs and returning text string
def read_pdf(files):
    file_content=""
    for file in files:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # Iterate through each page and extract text
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()
    return file_content


#-----------------------------------------------------------#
#------------------------💬 CHATBOT -----------------------#
#----------------------------------------------------------#
# def chatbot():
#     st.subheader("Ask questions from the PDFs")
#     st.markdown("<br>", unsafe_allow_html=True)
#     # Check if it is empty
#     if st.session_state.book_docsearch:   
#         prompt = st.chat_input("Say something")
        
#         # Write previous converstions
#         for i in st.session_state.conversation_chatbot:
#             user_msg = st.chat_message("human", avatar="🐒")
#             user_msg.write(i[0])
#             computer_msg = st.chat_message("ai", avatar="🧠")
#             computer_msg.write(i[1])
            
#         if prompt:                    
#             user_text = f'''{prompt}'''
#             user_msg = st.chat_message("human", avatar="🐒")
#             user_msg.write(user_text)

#             with st.spinner("Getting Answer..."):
#                 # No of chunks the search should retrieve from the db
#                 chunks_to_retrieve = 5
#                 retriever = st.session_state.book_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":chunks_to_retrieve})

#                 ## RetrievalQA Chain ##
#                 qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True)
#                 answer = qa({"query": prompt})["result"]
#                 computer_text = f'''{answer}'''
#                 computer_msg = st.chat_message("ai", avatar="🧠") 
#                 computer_msg.write(computer_text)
                
#                 # Showing chunks with score
#                 doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
#                 with st.popover("See chunks..."):
#                     st.write(doc_score)
#                 # Adding current conversation_chatbot to the list.
#                 st.session_state.conversation_chatbot.append((prompt, answer))   
#     else:
#         st.warning("Please upload a file")

def chatbot():
    # Create a sidebar for configuration options
    with st.sidebar:
        st.title("Model Settings")
        model_option = st.selectbox(
            "Choose Language Model",
            # ["cohere-command-r", "ollama-llama2", "ollama-llama2-uncensored"],
            ["cohere-command-r", "ollama-llama3.1", "ollama-gemma-3-12b"],
            index=0
        )
        st.session_state.selected_model = model_option
        
        # Update the LLM if model selection changes
        llm = get_llm()

        chunks_to_retrieve = st.slider(
            "Detail Level", 
            min_value=2, 
            max_value=10, 
            value=5, 
            help="Higher values provide more detailed responses but take longer"
        )

        st.divider()
        with st.expander("About the Models"):
            st.markdown("""
            - **Cohere Command-R**: Powerful model for precise responses
            - **Ollama Llama3.1 (8B)**: Latest Llama model with improved financial understanding
            - **Ollama Gemma 3 (12B)**: Google's powerful open model with strong reasoning
            - **Ollama Llama2**: Local model for general text analysis
            - **Ollama Llama2-Uncensored**: Local model with fewer restrictions
            
            > Note: To use Ollama models, you need to run Ollama in the background.
            """)
    
    # Main chat interface
    st.subheader("Ask questions about Mutual Funds")
    
    # Check if vector store exists
    if st.session_state.book_docsearch:   
        # Chat container with custom styling
        chat_container = st.container()
        with chat_container:
            # Custom CSS for a more modern look
            st.markdown("""
            <style>
            .user-message {
                background-color: #f0f2f6;
                border-radius: 15px;
                padding: 15px;
                margin: 5px 0;
            }
            .ai-message {
                background-color: #e6f3ff;
                border-radius: 15px;
                padding: 15px;
                margin: 5px 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Show conversation history
            for i in st.session_state.conversation_chatbot:
                user_msg = st.chat_message("user")
                user_msg.write(i[0])
                
                ai_msg = st.chat_message("assistant")
                ai_msg.write(i[1])
        
        # Input area at the bottom
        prompt = st.chat_input("Ask something about mutual funds...")
        
        # if prompt:                    
        #     user_msg = st.chat_message("user")
        #     user_msg.write(prompt)
            # with st.spinner("Thinking..."):
            #     # No of chunks the search should retrieve from the db
            #     chunks_to_retrieve = 5
            #     retriever = st.session_state.book_docsearch.as_retriever(
            #         search_type="similarity", 
            #         search_kwargs={"k":chunks_to_retrieve}
            #     )

            #     # Update LLM based on current selection
            #     llm = get_llm()
                
            #     ## RetrievalQA Chain ##
            #     qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True)
            #     answer = qa({"query": prompt})["result"]
                
            #     ai_msg = st.chat_message("assistant")
            #     ai_msg.write(answer)
                
            #     # Showing chunks with score - move to expandable section
            #     doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
            #     with st.expander("View source chunks"):
            #         st.write(doc_score)
                
            #     # Adding current conversation_chatbot to the list.
            #     st.session_state.conversation_chatbot.append((prompt, answer))   
            # Replace the spinner in the chatbot function
        if prompt:                    
            user_msg = st.chat_message("user")
            user_msg.write(prompt)
            with st.status("Processing with " + st.session_state.selected_model, state="running") as status:
                st.write(f"Retrieving context from {chunks_to_retrieve} document chunks...")
                
                retriever = st.session_state.book_docsearch.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": chunks_to_retrieve}
                )

                # Use cached query for better performance
                query_hash = hashlib.md5(prompt.encode()).hexdigest()
                answer = cached_query(query_hash, st.session_state.selected_model, chunks_to_retrieve)
                
                status.update(label="Response ready!", state="complete")
                # Display response
            ai_msg = st.chat_message("assistant")
            ai_msg.write(answer)
            
            doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
            with st.expander("View source chunks"):
                st.write(doc_score)
                
                # Add conversation to history
                st.session_state.conversation_chatbot.append((prompt, answer))
                
                status.update(label="Response ready!", state="complete")
        else:
            st.info("Please upload files on the Upload Files page first.")
            st.page_link("pages/Upload_Files.py", label="Go to Upload Files", icon="⬆️")

            
# For initialization of session variables
def initial(flag=False):
    path="db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if ('selected_option' not in st.session_state) or flag:
        try:
            st.session_state.selected_option = st.session_state.existing_indices[0]
        except:
            st.session_state.selected_option = None
    
    if 'conversation_chatbot' not in st.session_state:
        st.session_state.conversation_chatbot = []
    if 'book_docsearch' not in st.session_state:
        st.session_state.book_docsearch = None
    

def main():
    initial(True)
    # Streamlit UI
    st.title("💰 Mutual Fund Chatbot")
    
    # For showing the index selector
    file_list=[]
    for index in st.session_state.existing_indices:
        with open(f"db/{index}/desc.json", "r") as openfile:
            description = json.load(openfile)
            file_list.append(",".join(description["file_names"]))

    with st.popover("Select index", help="👉 Select the datastore from which data will be retrieved"):
        st.session_state.selected_option = st.radio("Select a Document...", st.session_state.existing_indices, captions=file_list, index=0)

    st.write(f"*Selected index* : **:orange[{st.session_state.selected_option}]**")
    
    # Load the selected index from local storage
    if st.session_state.selected_option:
        st.session_state.book_docsearch = FAISS.load_local(f"db/{st.session_state.selected_option}", embeddings, allow_dangerous_deserialization=True)
        # Call the chatbot function
        chatbot()
    else:
        st.warning("⚠️ No index present. Please add a new index.")
        # st.page_link("pages/Upload_Files.py", label="Upload Files", icon="⬆️")
            
            
 

            
main()