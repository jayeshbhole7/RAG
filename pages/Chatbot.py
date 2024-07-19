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

st.set_page_config("ChatSDK Fund","💬")

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')
# GEMINI_API_KEY = os.getenv('AIzaSyDKMP-6mrJ_2MuCaMbiW849kpk4QFwBUzI')

# Using Cohere's embed-english-v3.0 embedding model
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")



# For Cohere's command-r llm
llm = ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")


# return text string by reading pdfs
def read_pdf(files):
    file_content=""
    for file in files:
        

        pdf_reader = PyPDF2.PdfReader(file)

        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):

            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()
    return file_content


#-----------------------------------------------------------#
#------------------------💬 CHATBOT -----------------------#
#----------------------------------------------------------#
def chatbot():
    st.subheader("Ask questions from the PDFs")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.book_docsearch:   
        prompt = st.chat_input("Say something")

        
        for i in st.session_state.conversation_chatbot:
            user_msg = st.chat_message("human", avatar="🐒")
            user_msg.write(i[0])
            computer_msg = st.chat_message("ai", avatar="🧠")
            computer_msg.write(i[1])

        if prompt:                    
            user_text = f'''{prompt}'''
            user_msg = st.chat_message("human", avatar="🐒")
            user_msg.write(user_text)

            with st.spinner("Getting Answer..."):

                chunks_to_retrieve = 5
                retriever = st.session_state.book_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":chunks_to_retrieve})

              
                qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True)
                answer = qa({"query": prompt})["result"]
                computer_text = f'''{answer}'''
                computer_msg = st.chat_message("ai", avatar="🧠") 
                computer_msg.write(computer_text)

         
                doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
                with st.popover("See chunks..."):
                    st.write(doc_score)
                # listing conversations
                st.session_state.conversation_chatbot.append((prompt, answer))   
    else:
        st.warning("Please upload a file")




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

    st.title("💰 Mutual Fund Chatbot")


    file_list=[]
    for index in st.session_state.existing_indices:
        with open(f"db/{index}/desc.json", "r") as openfile:
            description = json.load(openfile)
            file_list.append(",".join(description["file_names"]))

    with st.popover("Select index", help="👉 Select the datastore from which data will be retrieved"):
        st.session_state.selected_option = st.radio("Select a Document...", st.session_state.existing_indices, captions=file_list, index=0)

    st.write(f"*Selected index* : **:orange[{st.session_state.selected_option}]**")


    if st.session_state.selected_option:
        st.session_state.book_docsearch = FAISS.load_local(f"db/{st.session_state.selected_option}", embeddings, allow_dangerous_deserialization=True)
        
        chatbot()
    else:
        st.warning("⚠️ No index present. Please add a new index.")
        st.page_link("pages/Upload_Files.py", label="Upload Files", icon="⬆️")





main()