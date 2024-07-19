import os
import json
import PyPDF2
import shutil
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
import streamlit as st

# Set the Cohere API key and model directly
COHERE_API_KEY = "Lqns8lzYYresnXB7QZ3Jc54zj8ri6X1Z6SDpgbZK"
COHERE_MODEL = "embed-english-v3.0"

# Initialize Cohere embeddings
embeddings = CohereEmbeddings(cohere_api_key="Lqns8lzYYresnXB7QZ3Jc54zj8ri6X1Z6SDpgbZK", model="embed-english-v3.0")


# Set up Streamlit
st.set_page_config(page_title="Upload Files", page_icon="📤")

# Function to read PDF
def read_pdf(files):
    file_content = ""
    for file in files:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()
    return file_content

# Store index function
def store_index(uploaded_file, index_option, file_names):
    try:
        if os.path.exists(f"db/{index_option}/index.faiss"):
            st.toast(f"Storing in existing index {index_option}...", icon="🗂️")
            with open(f'db/{index_option}/desc.json', 'r') as openfile:
                description = json.load(openfile)
                description["file_names"] = file_names + description["file_names"]
            with st.spinner("Processing..."):
                file_content = read_pdf(uploaded_file)
                book_documents = recursive_text_splitter.create_documents([file_content])
                book_documents = [Document(page_content=text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
                docsearch = FAISS.from_documents(book_documents, embeddings)
                old_docsearch = FAISS.load_local(f"db/{index_option}", embeddings, allow_dangerous_deserialization=True)
                docsearch.merge_from(old_docsearch)
                docsearch.save_local(f"db/{index_option}")
                with open(f"db/{index_option}/desc.json", "w") as outfile:
                    json.dump(description, outfile)
        else:
            st.toast(f"Storing in new index {index_option}...", icon="🗂️")
            with st.spinner("Processing..."):
                file_content = read_pdf(uploaded_file)
                book_documents = recursive_text_splitter.create_documents([file_content])
                book_documents = [Document(page_content=text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
                docsearch = FAISS.from_documents(book_documents, embeddings)
                docsearch.save_local(f"db/{index_option}")
                with open(f'db/{index_option}/desc.json', 'r') as openfile:
                    description = json.load(openfile)
                    description["file_names"] = file_names + description["file_names"]
                with open(f"db/{index_option}/desc.json", "w") as outfile:
                    json.dump(description, outfile)
        st.success(f"Successfully added to {description['name']}!")
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Recursive text splitter
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20)

# Initialize session state
def initial(flag=False):
    path = "db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# Main function
def main():
    initial()
    st.title("📤 Upload new files")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if uploaded_file:
        file_names = [file_name.name for file_name in uploaded_file]
        st.subheader("Select Index")
        st.caption("Create and select a new index or use an existing one")
        with st.popover("➕ Create new index"):
            form = st.form("new_index")
            index_name = form.text_input("Enter Index Name*")
            about = form.text_area("Enter description for Index")
            submitted = form.form_submit_button("Submit")
            if submitted:
                os.makedirs(f"db/{index_name}")
                description = {
                    "name": index_name,
                    "about": about,
                    "file_names": []
                }
                with open(f"db/{index_name}/desc.json", "w") as f:
                    json.dump(description, f)
                st.session_state.existing_indices = [index_name] + st.session_state.existing_indices
                st.success(f"New Index {index_name} created successfully")
        index_option = st.selectbox('Add to existing Indices', st.session_state.existing_indices)
        if st.button("Store"):
            store_index(uploaded_file, index_option, file_names)
    st.subheader("Stored Indices")
    with st.expander("See existing indices"):
        if len(st.session_state.existing_indices) == 0:
            st.warning("No existing indices. Please upload a pdf to start.")
        for index in st.session_state.existing_indices:
            if os.path.exists(f"db/{index}/desc.json"):
                col1, col2 = st.columns([6, 1])
                with col1:
                    with open(f"db/{index}/desc.json", "r") as openfile:
                        description = json.load(openfile)
                        file_list = ",".join(description["file_names"])
                        st.markdown(f"<h4>{index}</h4>", unsafe_allow_html=True)
                        st.caption(f"Desc: {description['about']}")
                        st.caption(f"Files: {file_list}")
                with col2:
                    t = st.button("🗑️", key=index)
                    if t:
                        shutil.rmtree(f"db/{index}")
                        initial(flag=True)
                        st.toast(f"{index} deleted", icon='🗑️')
                        time.sleep(1)
                        st.rerun()

if __name__ == "__main__":
    main()
