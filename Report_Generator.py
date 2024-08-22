# from langchain_community.chat_models import ChatCohere
# from langchain_cohere import ChatCohere
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_cohere import CohereEmbeddings
# from langchain_community.document_loaders.csv_loader import CSVLoader
# import json
# import PyPDF2
# import streamlit as st
# import os
# from dotenv import load_dotenv
# import base64
# import pandas as pd
# from io import StringIO
# import re
# # import wget
# import tarfile
# # os.environ['JAVA_HOME'] = './jdk'


# # if "JAVA_HOME" not in os.environ:
# #     java_url = "https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz"
# #     java_tar_gz = wget.download(java_url)
# #     # st.write("Extracting Java 11...")
# #     with tarfile.open(java_tar_gz, "r:gz") as tar:
# #         tar.extractall()
# #     java_dir = os.path.basename(java_url).split(".tar.gz")[0]
# #     java_home = os.path.abspath(java_dir)
# #     os.environ["JAVA_HOME"] = java_home
# #     os.environ["PATH"] += os.pathsep + os.path.join(java_home, "bin")





# st.set_page_config("Report Funds","🤖")

# load_dotenv()

# # API Keys
# # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# COHERE_API_KEY = os.getenv('COHERE_API_KEY')






# #Retrieve Schemes from CSV
# def schemeRetrieve(path):
#     df=pd.read_csv(path)
#     df.columns=df.iloc[0].values
#     df=df.drop(df.index[0])
#     mf_scheme_list=["All Schemes"]
#     for i in df['Scheme Name'].values:
#         if "Schemes" in str(i):
#             if str(i) not in mf_scheme_list:
#                 mf_scheme_list.append(str(i)) 
#     return mf_scheme_list

# def field_retrieve(path):
#     df=pd.read_csv(path)
#     df.columns=df.iloc[0].values
#     df=df.drop(df.index[0])
#     l=df.columns[2:].values
#     ll=[]
#     for i in l:
#         newl=i.replace("\r"," ")
#         newl=newl.replace("segregat ed","segregated")
#         if "as" in newl:
#             pos=newl.find("as")
#         if "for" in newl:
#             pos=newl.find("for")
#         newl=newl[0:pos-1]
#         ll.append(newl)
#     ll.append("Net Inflow or Outflow")
#     ll.append("Net Asset under Management per Scheme")
#     ll.append("Net Inflow or Outflow per Scheme")
    
#     return ll



# # Using Cohere's embed-english-v3.0 embedding model
# embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")


# # For OpenAI's gpt-3.5-turbo llm
# # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# # For Cohere's command-r llm
# llm = ChatCohere(temperature=1, cohere_api_key=COHERE_API_KEY, model="command-r")


# # For reading PDFs and returning text string
# def read_pdf(files):
#     file_content=''' Net Inflow or Outflow: Calculated as taking the difference of `Funds Mobilized for the month` and `Repurchase/Redemption for the month` field.

#      Net Asset under Management per Scheme: Determined by dividing the `Net Assets Under Management` by the `No. of Schemes`.

#      Net Inflow or Outflow per Scheme: This is the ratio of `Net Inflow or Outflow` to the `No. of Schemes`.
#      '''
#     for file in files:
#         # Create a PDF file reader object
#         pdf_reader = PyPDF2.PdfReader(file)
#         # Get the total number of pages in the PDF
#         num_pages = len(pdf_reader.pages)
#         # Iterate through each page and extract text
#         for page_num in range(num_pages):
#             # Get the page object
#             page = pdf_reader.pages[page_num]
#             file_content += page.extract_text()
#     return file_content


# #Download CSV
# def download_df(content, filename='data.csv'):
#     df = pd.read_csv(StringIO(content), sep="|", skipinitialspace=True)

#     # Remove leading and trailing whitespaces from column names
#     df.columns = df.columns.str.strip()
#     df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
#     # csv_bytes = content.encode()  # Convert string to bytes
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()    # Encode bytes to base64
#     href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
#     return href

# #-----------------------------------------------------------#
# #------------------------💬 CHATBOT -----------------------#
# #----------------------------------------------------------#
# def chatbot():
#     st.subheader("Generate your Desired Mutual Fund Report")
#     st.markdown("<br>", unsafe_allow_html=True)
#     btn=st.button("Generate", type="primary")
#     # Check if it is empty
#     if st.session_state.book_docsearch:   
#         # prompt = st.chat_input("Say something")
#         prompt="give the data of "+", ".join(st.session_state.selected_field)
        
#         # Write previous converstions
#         for i in st.session_state.conversation:
#             user_msg = st.chat_message("human", avatar="🐒")
#             user_msg.write(i[0])
#             computer_msg = st.chat_message("ai", avatar="🧠")
#             computer_msg.write(i[1])
            
#         if btn and prompt:
#             exprompt=prompt                       #to store the previous prompt
#             exprompt="For the "+", ".join(st.session_state.selected_scheme)+", "+exprompt
#             exprompt+=" . Give the result in tabular format."          #want to show data in tabular form
#             # promptcsv=prompt+" in csv format"   #for downloading csv version                                    
#             user_text = f'''Scheme: {", ".join(st.session_state.selected_scheme)}, Field: {", ".join(st.session_state.selected_field)}'''
#             user_msg = st.chat_message("human", avatar="🐒")
#             user_msg.write(user_text)

#             with st.spinner("Getting Answer..."):
#                 # No of chunks the search should retrieve from the db
#                 chunks_to_retrieve = 5
#                 retriever = st.session_state.book_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":chunks_to_retrieve})

#                 ## RetrievalQA Chain ##
#                 qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True)
#                 answer = qa({"query": exprompt})["result"]
#                 # answercsv = qa({"query": promptcsv})["result"]    #for generating csv file
                
#                 computer_text = f'''{answer}'''
#                 # print(answer)
#                 computer_msg = st.chat_message("ai", avatar="🧠") 
#                 computer_msg.write(computer_text)
                
#                 #Download Data
#                 st.markdown(download_df(answer), unsafe_allow_html=True)
                
#                 # Showing chunks with score
#                 doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
#                 with st.popover("See chunks..."):
#                     st.write(doc_score)
#                 # Adding current conversation to the list.
#                 st.session_state.conversation.append((user_text, answer))   
#     else:
#         st.warning("Please upload a file")


            
# # For initialization of session variables
# def initial(flag=False):
    # path="db"
    # if 'existing_indices' not in st.session_state or flag:
    #     st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    # if ('selected_option' not in st.session_state) or flag:
    #     try:
    #         st.session_state.selected_option = st.session_state.existing_indices[0]
    #     except:
    #         st.session_state.selected_option = None
    
    # if 'conversation' not in st.session_state:
    #     st.session_state.conversation = []
    # if 'book_docsearch' not in st.session_state:
    #     st.session_state.book_docsearch = None
        
        
    # if 'mf_schemes' not in st.session_state or flag:
    #     try:
    #         st.session_state.mf_schemes=schemeRetrieve(f"./{path}/table.csv")
    #     except:
    #         st.session_state.mf_schemes=None
            
    # if ('selected_scheme' not in st.session_state) or flag:
    #     try:
    #         st.session_state.selected_scheme = st.session_state.mf_schemes[0]
    #     except:
    #         st.session_state.selected_scheme = None
    
    # if 'mf_field' not in st.session_state or flag:
    #     try:
    #         st.session_state.mf_field=field_retrieve(f"./{path}/table.csv")
    #     except:
    #         st.session_state.mf_field=None
    
    # if ('selected_field' not in st.session_state) or flag:
    #     try:
    #         st.session_state.selected_field = st.session_state.mf_schemes[0]
    #     except:
    #         st.session_state.selected_field = None
            

# def main():
#     initial(True)
#     # Streamlit UI
#     st.title("💰 Mutual Fund Report Generator")
    
#     # For showing the index selector
#     file_list=[]
#     for index in st.session_state.existing_indices:
#         with open(f"db/{index}/desc.json", "r") as openfile:
#             description = json.load(openfile)
#             file_list.append(",".join(description["file_names"]))
    
    

#     with st.popover("Select Index", help="👉 Select the datastore from which data will be retrieved"):
#         st.session_state.selected_option = st.radio("Select a Document...", st.session_state.existing_indices, captions=file_list, index=0)

#     st.write(f"*Selected Index* : **:orange[{st.session_state.selected_option}]**")
    
#     if 'existing_indices' in st.session_state:
        
#         st.session_state.mf_schemes=schemeRetrieve(f"./db/table.csv")
        
#         with st.popover("Select Scheme", help="👉 Select the Mutual Fund Scheme"):
        
#             radio_scheme = st.multiselect("Select a Scheme...", st.session_state.mf_schemes)
#             # if radio_scheme == "Other (Add Manually)...":
#             #     radio_scheme = st.text_input("Write the name of the Schemes with comma")
#             st.session_state.selected_scheme = radio_scheme
#             # st.write(f"You have selected: {st.session_state.selected_scheme}")
#         temp_scheme=", ".join(st.session_state.selected_scheme)
#         st.write(f"*Selected Scheme* : **:green[{temp_scheme}]**")
        
#         with st.popover("Select Field", help="👉 Select the Field"):
        
#             radio_field = st.multiselect("Select a Scheme...", st.session_state.mf_field)
#             st.session_state.selected_field = radio_field
#         temp_field=", ".join(st.session_state.selected_field)
#         st.write(f"*Selected Scheme* : **:green[{temp_field}]**")
        
    
#     # Load the selected index from local storage
#     if st.session_state.selected_option:
#         st.session_state.book_docsearch = FAISS.load_local(f"db/{st.session_state.selected_option}", embeddings, allow_dangerous_deserialization=True)
#         # Call the chatbot function
#         chatbot()
#     else:
#         st.warning("⚠️ No index present. Please add a new index.")
#         st.page_link("pages/Upload_Files.py", label="Upload Files", icon="⬆️")
#         st.page_link("pages/Chatbot.py", label="Basic Chatbot", icon="💬")
        
            
            
 

            
# main()

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import google.generativeai as genai
import json
import PyPDF2
import streamlit as st
import os
from dotenv import load_dotenv
import base64
import pandas as pd
from io import StringIO

st.set_page_config("Report Funds", "🤖")
load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "models/chat-bison-001"

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")

# Function to generate text using Gemini API
def generate_text_with_gemini(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        response = genai.generate_text(
            model=GEMINI_MODEL,
            prompt=prompt,
            temperature=0.0  # Adjust for creativity
        )
        return response.result
    except Exception as e:
        st.error(f"Error generating text with Gemini: {e}")
        return ""

# Function to retrieve schemes from CSV
def scheme_retrieve(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.iloc[0].values
        df = df.drop(df.index[0])
        mf_scheme_list = ["All Schemes"]
        for i in df['Scheme Name'].values:
            if "Schemes" in str(i) and str(i) not in mf_scheme_list:
                mf_scheme_list.append(str(i))
        return mf_scheme_list
    except Exception as e:
        st.error(f"Error reading schemes from CSV: {e}")
        return []

# Function to retrieve fields from CSV 
def field_retrieve(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.iloc[0].values
        df = df.drop(df.index[0])
        l = df.columns[2:].values
        ll = []
        for i in l:
            newl = i.replace("\r", " ").replace("segregat ed", "segregated")
            pos = newl.find("as") if "as" in newl else newl.find("for") if "for" in newl else -1
            if pos != -1:
                newl = newl[0:pos - 1]
            ll.append(newl)
        ll.extend(["Net Inflow or Outflow", "Net Asset under Management per Scheme", "Net Inflow or Outflow per Scheme"])
        return ll
    except Exception as e:
        st.error(f"Error reading fields from CSV: {e}")
        return []

# Function to read PDFs and returning text string
def read_pdf(files):
    file_content = ''' 
    Net Inflow or Outflow: Calculated as taking the difference of `Funds Mobilized for the month` and `Repurchase/Redemption for the month` field.

    Net Asset under Management per Scheme: Determined by dividing the `Net Assets Under Management` by the `No. of Schemes`.

    Net Inflow or Outflow per Scheme: This is the ratio of `Net Inflow or Outflow` to the `No. of Schemes`.
    '''
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

# Download CSV function 
def download_df(content, filename='data.csv'):
    df = pd.read_csv(StringIO(content), sep="|", skipinitialspace=True, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# ----------------------------------------------------------- #
# ------------------------ 💬 CHATBOT ------------------------ #
# ----------------------------------------------------------- #

def chatbot():
    st.subheader("Generate your Desired Mutual Fund Report")
    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("Generate", type="primary")
    
    # Check if documents have been loaded
    if st.session_state.book_docsearch:  
        # Construct the prompt based on selected fields and schemes
        prompt = "Give the data of " + ", ".join(st.session_state.selected_field)

        # Display previous conversations (if any)
        for user_message, ai_response in st.session_state.conversation:
            st.chat_message("human", avatar="🐒").write(user_message)
            st.chat_message("ai", avatar="🧠").write(ai_response)

        if btn and prompt:
            # Prepare prompt for Gemini API
            exprompt = f"For the {', '.join(st.session_state.selected_scheme)}, {prompt}. Give the result in tabular format."
            
            # Display user input in chat interface
            user_text = f"Scheme: {', '.join(st.session_state.selected_scheme)}, Field: {', '.join(st.session_state.selected_field)}"
            st.chat_message("human", avatar="🐒").write(user_text)

            with st.spinner("Generating Report..."):
                chunks_to_retrieve = 5  
                retriever = st.session_state.book_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": chunks_to_retrieve})
                
                # Use the Gemini API directly for the QA chain
                qa = RetrievalQA.from_chain_type(llm=generate_text_with_gemini, chain_type="stuff", retriever=retriever, verbose=True)
                answer = qa.run(exprompt)
                
                # Display the answer in the chat interface
                st.chat_message("ai", avatar="🧠").write(answer)

                # Option to download results as CSV
                st.markdown(download_df(answer), unsafe_allow_html=True)

                # Show relevant document chunks (for debugging/transparency)
                doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
                with st.expander("See chunks..."):
                    st.write(doc_score)

                # Store the conversation history
                st.session_state.conversation.append((user_text, answer))
    else:
        st.warning("Please upload a file")

# Initialize session variables
def initial(flag=False):
    path = "db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if 'selected_option' not in st.session_state or flag:
        try:
            st.session_state.selected_option = st.session_state.existing_indices[0]
        except:
            st.session_state.selected_option = None
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'book_docsearch' not in st.session_state:
        st.session_state.book_docsearch = None
        
    if 'mf_schemes' not in st.session_state or flag:
        try:
            st.session_state.mf_schemes = scheme_retrieve(f"./{path}/table.csv")
        except:
            st.session_state.mf_schemes = None
            
    if 'selected_scheme' not in st.session_state or flag:
        try:
            st.session_state.selected_scheme = st.session_state.mf_schemes[0]
        except:
            st.session_state.selected_scheme = None
    
    if 'mf_field' not in st.session_state or flag:
        try:
            st.session_state.mf_field = field_retrieve(f"./{path}/table.csv")
        except:
            st.session_state.mf_field = None
    
    if 'selected_field' not in st.session_state or flag:
        try:
            st.session_state.selected_field = st.session_state.mf_schemes[0]
        except:
            st.session_state.selected_field = None

def main():
    initial(True)
    # Streamlit UI
    st.title("💰 Mutual Fund Report Generator")

    # For showing the index selector
    file_list = []
    if 'existing_indices' in st.session_state:
        for index in st.session_state.existing_indices:
            try:
                with open(f"db/{index}/desc.json", "r") as openfile:
                    description = json.load(openfile)
                    file_list.append(",".join(description["file_names"]))
            except Exception as e:
                st.error(f"Error loading description for index {index}: {e}")

    with st.popover("Select Index", help="👉 Select the datastore from which data will be retrieved"):
        if file_list:
            st.session_state.selected_option = st.radio(
                "Select a Document...",
                st.session_state.existing_indices,
                captions=file_list,
                index=0
            )
        else:
            st.warning("No indices found. Please upload files.")

    st.write(f"*Selected Index* : **:orange[{st.session_state.selected_option}]**")

    if 'existing_indices' in st.session_state:
        st.session_state.mf_schemes = scheme_retrieve(f"./db/table.csv")

        with st.popover("Select Scheme", help="👉 Select the Mutual Fund Scheme"):
            if st.session_state.mf_schemes:
                radio_scheme = st.multiselect("Select a Scheme...", st.session_state.mf_schemes)
                st.session_state.selected_scheme = radio_scheme
                temp_scheme = ", ".join(st.session_state.selected_scheme)
                st.write(f"*Selected Scheme* : **:green[{temp_scheme}]**")
            else:
                st.warning("No schemes found. Please check your CSV file.")

        with st.popover("Select Field", help="👉 Select the Field"):
            if st.session_state.mf_field:
                radio_field = st.multiselect("Select a Field...", st.session_state.mf_field)
                st.session_state.selected_field = radio_field
                temp_field = ", ".join(st.session_state.selected_field)
                st.write(f"*Selected Field* : **:green[{temp_field}]**")
            else:
                st.warning("No fields found. Please check your CSV file.")

    # Load the selected index from local storage
    if st.session_state.selected_option:
        try:
            st.session_state.book_docsearch = FAISS.load_local(
                f"db/{st.session_state.selected_option}",
                embeddings,
                allow_dangerous_deserialization=True
            )
            # Call the chatbot function
            chatbot()
        except Exception as e:
            st.error(f"Error loading index: {e}")
    else:
        st.warning("⚠️ No index present. Please add a new index.")
        st.page_link("pages/Upload_Files.py", label="Upload Files", icon="⬆️")
        st.page_link("pages/Chatbot.py", label="Basic Chatbot", icon="💬")

if __name__ == "__main__":
    main()
