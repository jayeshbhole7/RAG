# Mutual Fund Chatbot

A retrieval-augmented generation (RAG) application designed to assist investors by providing insightful, personalized guidance through a conversational finance buddy. Analyze mutual fund data interactively through a user-friendly interface.

![Chatbot Interface](vscode-file://vscode-app/e:/hackathon/image/chatbot.png)

## ✨ Features

* **Friendly Finance Interface** : Built using Streamlit
* Report Generator: Create structured mutual fund reports
* Investment Buddy Chatbot: Answer investment queries naturally
* File Uploader: Seamlessly upload PDF-based mutual fund reports
* **Flexible Model Selection** : Choose between cloud and local LLMs:
* Cohere Command-R (cloud)
* Llama 3.1 8B (local)
* Gemma 3 12B (local)
* **Smart Data Processing** :
* Extract text from financial PDFs
* Organize data in a vector database (FAISS)
* Generate structured financial reports
* Download results as CSV

## 🏗️ Technical Architecture

* **Vector Database** : FAISS for efficient similarity search
* **Embedding Model** : Cohere's embed-english-v3.0
* **Large Language Models** :
* Primary: Cohere's command-r
* Alternatives: Llama 3.1 (8B) and Gemma 3 (12B) via Ollama
* **Data Processing** :
* PDF extraction via PyPDF2
* Text chunking via RecursiveCharacterTextSplitter
* CSV data manipulation using pandas

## 🚀 Installation and Setup

### 1. Clone the Repository


### 2. Create Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

Linux/Mac

python -m venv venv
source venv/bin/activate


### 3. Install Dependencies


pip install -r requirements.txt



### 4. API Keys

Create a `.env` file in the root directory:

COHERE_API_KEY=your_cohere_api_key


### 5. Setting Up Local LLMs with Ollama

To use local models like Llama 3.1 and Gemma 3, you need to set up Ollama:


1. Install Ollama from [ollama.ai](vscode-file://vscode-app/c:/Users/JAYESH%20BHOLE/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
2. Pull the models you want to use:

Latest Llama 3.1 (8B parameter version)

ollama pull llama3.1:8b

Google's Gemma model (12B parameter version)

ollama pull gemma:12b

Run Ollama in the background before starting the app:


Windows: Start in a separate terminal

start ollama serve

Linux/Mac:

ollama serve &


### 6. Launch the Application

streamlit run .\Report_Generator.py


## 📊 Usage Instructions

### Report Generator

1. Select an index containing mutual fund data
2. Choose specific mutual fund schemes
3. Select data fields of interest
4. Click "Generate" to create a report
5. Download results as CSV

### Chatbot

1. Navigate to the Chatbot page
2. Select a language model from the sidebar
3. Ask investment-related questions
4. View retrieved document chunks for transparency

### Upload Files

1. Go to the Upload Files page
2. Upload PDF mutual fund reports
3. Create a new index or add to existing one
4. View and manage stored indices

## 💡 Model Selection Guide

* **Cohere Command-R** : Best for accurate financial responses (requires API key)
* **Llama 3.1 (8B)** : Good balance of performance and speed for local usage
* **Gemma 3 (12B)** : Best comprehensive responses for complex financial questions

## 🛠️ Practical Applications

### 1. Financial Advisory Tool

* Client portfolio reviews and analysis
* Risk profiling across fund categories
* Generate custom reports for different investor types

### 2. Investment Education Platform

* Interactive Q&A about mutual fund categories
* Extract market trends and insights
* Build scenario analyses for investment strategies

### 3. Research Automation

* Compare metrics across similar funds
* Track performance changes over time
* Identify outlier funds with unusual metrics

### 4. Content Creation

* Generate first-draft research reports
* Extract key statistics for investor newsletters
* Create data-driven snippets about fund performance

## 🔮 Future Enhancements

* **Advanced Table Extraction** : Implement Tabula or Camelot for precise PDF data extraction
* **Data Visualization** : Add interactive charts showing fund performance
* **Multilingual Support** : Add support for regional languages
* **Enhanced Analytics** : Implement comparative analysis across time periods
* **User Authentication** : Add profile-based preferences and history

## 📦 Dependencies

* streamlit (User Interface)
* langchain (LLM interaction framework)
* faiss-cpu (Vector database)
* cohere (Text embeddings & LLM API)
* PyPDF2 (PDF processing)
* pandas (Financial data manipulation)
* ollama (Local LLM serving)

## 🔗 Online Demo

Available on Streamlit Cloud: [https://chatfunds.streamlit.app/](vscode-file://vscode-app/c:/Users/JAYESH%20BHOLE/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙌 Acknowledgements

* Cohere for embedding and LLM APIs
* Meta AI for Llama 3.1 model
* Google for Gemma 3 model
