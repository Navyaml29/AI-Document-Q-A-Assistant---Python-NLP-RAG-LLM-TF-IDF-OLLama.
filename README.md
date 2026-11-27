# AI-Document-Q-A-Assistant---Python-NLP-RAG-LLM-TF-IDF-OLLama.

A Python application that allows users to upload PDF and TXT documents and ask questions about their content.

## Features
- PDF and TXT document upload
- Document text extraction
- Text chunking
- TF-IDF-based document retrieval
- Cosine similarity
- LLM-based answer generation
- Display retrieved document sections

## Technologies
- Python
- Streamlit
- Scikit-learn
- PyPDF
- NLP
- Ollama
- Llama 3.2

## Installation

pip install -r requirements.txt

## Install Ollama

Download Ollama from:
https://ollama.com/

Download the model:

ollama pull llama3.2

## Run

ollama serve

In another terminal:

streamlit run app.py

## Note

This project uses TF-IDF retrieval and an LLM for question answering. Its results depend on the quality of extracted text and retrieved context.
