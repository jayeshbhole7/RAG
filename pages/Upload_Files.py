# import os
# import time
# import json
# import shutil
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI
# import PyPDF2
# import streamlit as st
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_cohere import CohereEmbeddings
# # import tabula
# # import camelot as cm
# # os.environ['JAVA_HOME'] = './jdk'



# st.set_page_config("Upload Files", "📤")

# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")


# def read_pdf(files):
#     # df=""
#     # for file in files:
#     #     temp=tabula.read_pdf(file, pages="all", multiple_tables=True)
#     #     for t in temp:
#     #         df+=(', '.join(map(str, t.values.tolist())))
#     # return df
    
#     file_content=""
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

# def store_index(uploaded_file, index_option, file_names):
#                 # If index exists
#             if os.path.exists(f"db/{index_option}/index.faiss"):
#                 st.toast(f"Storing in **existing index** :green[**{index_option}**]...", icon = "🗂️")
#                 # Opening JSON file
#                 with open(f'db/{index_option}/desc.json', 'r') as openfile:
#                     description = json.load(openfile)
#                     description["file_names"] = file_names + description["file_names"]
                
#                 with st.spinner("Processing..."):
#                     #Read the pdf file
#                     file_content = read_pdf(uploaded_file)
#                     #Create document chunks
#                     book_documents = recursive_text_splitter.create_documents([file_content])
#                     # Limit the no of characters
#                     book_documents = [Document(page_content = text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
#                     docsearch = FAISS.from_documents(book_documents, embeddings)
#                     old_docsearch = FAISS.load_local(f"db/{index_option}", embeddings, allow_dangerous_deserialization=True)
#                     docsearch.merge_from(old_docsearch)
#                     docsearch.save_local(f"db/{index_option}")
#                     # Write the json file
#                     with open(f"db/{index_option}/desc.json", "w") as outfile:
#                         json.dump(description, outfile)

#             # If index does not exist
#             else:         
#                 st.toast(f"Storing in **new index** :green[**{index_option}**]...", icon="🗂️")

#                 with st.spinner("Processing..."):
#                     #Read the pdf file
#                     file_content = read_pdf(uploaded_file)
#                     #Create document chunks
#                     book_documents = recursive_text_splitter.create_documents([file_content])
#                     # Limit the no of characters, remove \n
#                     book_documents = [Document(page_content = text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
#                     docsearch = FAISS.from_documents(book_documents, embeddings)
                    
#                     docsearch.save_local(f"db/{index_option}")
#                     # Read json file
#                     with open(f'db/{index_option}/desc.json', 'r') as openfile:
#                         description = json.load(openfile)
#                         description["file_names"] = file_names + description["file_names"]
#                     # Write the json file
#                     with open(f"db/{index_option}/desc.json", "w") as outfile:
#                         json.dump(description, outfile)

#             st.success(f"Successfully added to **{description['name']}**!")


# recursive_text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=2000,
#     chunk_overlap=20)

# def initial(flag=False):
#     path="db"
#     if 'existing_indices' not in st.session_state or flag:
#         st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# def main():
    
#     initial()
   
#     st.title("📤 Upload new files")       
#     uploaded_file = st.file_uploader("&nbsp;Upload PDF", type="pdf", accept_multiple_files=True, help="Upload PDF files to store")
    
#     if uploaded_file:
#         # Get the uploaded file names
#         file_names = [file_name.name for file_name in uploaded_file]
#         # Select the index in which to store
#         st.subheader("Select Index")
#         st.caption("Create and select a new index or use an existing one")
#         # Create new index
#         with st.popover("➕ Create new index"):
#             form = st.form("new_index")
#             index_name = form.text_input("Enter Index Name*")
#             about = form.text_area("Enter description for Index")
#             submitted = form.form_submit_button("Submit", type="primary")
#             if submitted:
#                 os.makedirs(f"db/{index_name}")

#                 description = {
#                     "name": index_name,
#                     "about": about,
#                     "file_names": []
#                 }

#                 with open(f"db/{index_name}/desc.json", "w") as f:
#                     json.dump(description, f)
#                 st.session_state.existing_indices = [index_name] + st.session_state.existing_indices
#                 st.success(f"New Index **{index_name}** created successfully")
                
                
#         # tabula.convert_into(uploaded_file[0], f"db/{index_name}/table.csv",pages='all', output_format='csv')
#         # tables=cm.read_pdf(uploaded_file[0],pages='1',orient="records")
#         # Existing indices
#         index_option = st.selectbox('Add to existing Indices', st.session_state.existing_indices)
#         st.write(index_option)

        
#         if st.button("Store", type="primary"):
#             # Store index in local storage
#             store_index(uploaded_file, index_option, file_names)
            
    
#     st.title("\n\n\n")
    
#     # Show already stored indices
#     st.subheader("💽 Stored Indices", help="💽 See all the indices you have previously stored.")
#     with st.expander("🗂️ See existing indices"):
#         st.divider()
#         if len(st.session_state.existing_indices) == 0:
#             st.warning("No existing indices. Please upload a pdf to start.")
#         for index in st.session_state.existing_indices:
#             if os.path.exists(f"db/{index}/desc.json"):
#                 col1, col2 = st.columns([6,1], gap="large")
#                 with col1:
#                     with open(f"db/{index}/desc.json", "r") as openfile:
#                         description = json.load(openfile)
#                         file_list = ",".join(description["file_names"])
#                         st.markdown(f"<h4>{index}</h4>", unsafe_allow_html=True)
#                         st.caption(f"***Desc :*** {description['about']}")
#                         st.caption(f"***Files :*** {file_list}")
#                         openfile.close()
#                 with col2:
#                     t = st.button("🗑️", key=index, help="❌ :red[Clicking on this will delete this index]")
#                     if t:
#                         script_directory = os.path.dirname(os.path.abspath(__file__))
#                         del_path = os.path.join(script_directory, "../db", index) 
#                         shutil.rmtree(del_path)
#                         initial(flag=True)
#                         st.toast(f"**{index}** :red[deleted]", icon='🗑️')
#                         time.sleep(1)
#                         st.rerun()
                        
        
        
# main()

import os
import time
import json
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import PyPDF2
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_cohere import CohereEmbeddings
import google.generativeai as genai

st.set_page_config("Upload Files", "📤")

load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "models/chat-bison-001"

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")

# Function to generate text using Gemini API
def generate_text_with_gemini(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    response = genai.generate_text(
        model=GEMINI_MODEL,
        prompt=prompt,
        temperature=0.0  # Adjust for creativity
    )
    return response.result

# Function to read PDFs efficiently
def read_pdf(files):
    file_content = ""
    for file in files:
        with open(file, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            file_content += "".join(page.extract_text() for page in pdf_reader.pages)
    return file_content

def store_index(uploaded_file, index_option, file_names):
    if os.path.exists(f"db/{index_option}/index.faiss"):
        st.toast(f"Storing in existing index :green[**{index_option}**]...", icon="🗂️")
        with open(f'db/{index_option}/desc.json', 'r') as openfile:
            description = json.load(openfile)
            description["file_names"] = file_names + description["file_names"]

        with st.spinner("Processing..."):
            file_content = read_pdf(uploaded_file)

            # Summarize file content using Gemini
            summary_prompt = f"Please summarize the following text from a financial document:\n\n{file_content}\n\nSummary:"
            summary = generate_text_with_gemini(summary_prompt)

            # Split and clean document chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
            book_documents = text_splitter.create_documents([file_content])

            # Create FAISS index and merge with existing one if applicable
            docsearch = FAISS.from_documents(book_documents, embeddings)
            if os.path.exists(f"db/{index_option}/index.faiss"):
                old_docsearch = FAISS.load_local(f"db/{index_option}", embeddings, allow_dangerous_deserialization=True)
                docsearch.merge_from(old_docsearch)
            
            docsearch.save_local(f"db/{index_option}")

            # Add summary to the description
            description["summary"] = summary

            with open(f"db/{index_option}/desc.json", "w") as outfile:
                json.dump(description, outfile)

    else:  # Code for creating a new index (similar to existing logic)
        # ... (unchanged)
    
            st.success(f"Successfully added to {description['name']}!")
    
def initial(flag=False):
    path = "db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def main():
    initial()

    st.title("📤 Upload new files")
    uploaded_file = st.file_uploader("&nbsp;Upload PDF", type="pdf", accept_multiple_files=True, help="Upload PDF files to store")

    if uploaded_file:
        file_names = [file_name.name for file_name in uploaded_file]

        st.subheader("Select Index")
        st.caption("Create and select a new index or use an existing one")

        # New Index Creation (using form)
        with st.form("new_index"):
            index_name = st.text_input("Enter Index Name*", key="index_name")
            about = st.text_area("Enter description for Index", key="about")
            submitted = st.form_submit_button("Create New Index", type="primary")
            if submitted:
                if index_name: 
                    os.makedirs(f"db/{index_name}")
                    description = {
                        "name": index_name,
                        "about": about,
                        "file_names": []
                    }

                    with open(f"db/{index_name}/desc.json", "w") as f:
                        json.dump(description, f)
                    st.session_state.existing_indices = [index_name] + st.session_state.existing_indices
                    st.success(f"New Index **{index_name}** created successfully")
                else:
                    st.error("Please enter an index name.")
        # Select Existing Index
        index_option = st.selectbox('Add to existing Indices', st.session_state.existing_indices)

        if st.button("Store", type="primary"):
            if index_option:
                store_index(uploaded_file, index_option, file_names)
            else:
                st.error("Please select or create an index.")



    st.subheader("💽 Stored Indices", help="See all the indices you have previously stored.")
    with st.expander("🗂️ See existing indices"):
        if not st.session_state.existing_indices:
            st.warning("No existing indices. Please upload a PDF to start.")
        else:
            for index in st.session_state.existing_indices:
                if os.path.exists(f"db/{index}/desc.json"):
                    col1, col2 = st.columns([6, 1], gap="large")
                    with col1:
                        with open(f"db/{index}/desc.json", "r") as openfile:
                            description = json.load(openfile)
                            file_list = ", ".join(description["file_names"])
                            st.markdown(f"<h4>{index}</h4>", unsafe_allow_html=True)
                            st.caption(f"Desc: {description['about']}")
                            st.caption(f"Files: {file_list}")
                            if "summary" in description:
                                st.caption(f"Summary: {description['summary']}")  # Display summary if available
                    with col2:
                        if st.button("🗑️", key=index, help="Clicking on this will delete this index"):
                            shutil.rmtree(f"db/{index}")
                            initial(flag=True)
                            st.toast(f"{index} deleted", icon='🗑️')
                            time.sleep(1)
                            st.rerun()

# --- Main Execution ---
if __name__ == "__main__":
    main()
