import streamlit as st
import re
import requests

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Document Q&A Assistant")

st.write(
    "Upload a PDF or TXT document and ask questions "
    "based on its content."
)




def extract_text(uploaded_file):

    if uploaded_file.name.lower().endswith(".pdf"):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        return text

    return uploaded_file.getvalue().decode(
        "utf-8",
        errors="replace"
    )


def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



def split_text(text, chunk_size=120, overlap=20):

    words = text.split()

    chunks = []

    if not words:
        return chunks

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks




def retrieve_chunks(question, chunks, top_k=3):

    if not chunks:
        return []

    documents = chunks + [question]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        vectors = vectorizer.fit_transform(documents)

        similarities = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        ).flatten()

    except ValueError:

        return []

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        if similarities[index] > 0:

            results.append({
                "text": chunks[index],
                "score": float(similarities[index])
            })

    return results



def generate_answer(question, context):

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the supplied document context.

If the answer is not available in the context,
clearly say that the document does not contain
enough information to answer.

Document Context:
{context}

Question:
{question}

Answer:
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json().get(
            "response",
            "No answer generated."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Ollama is not running. "
            "Please start Ollama and try again."
        )

    except requests.exceptions.Timeout:

        return "The model took too long to respond."

    except requests.exceptions.RequestException as error:

        return f"Model error: {error}"




with st.sidebar:

    st.header("About the Project")

    st.write(
        "This application uses document retrieval "
        "and an LLM to answer questions."
    )

    st.subheader("Technologies")

    st.write(
        "- Python\n"
        "- Streamlit\n"
        "- NLP\n"
        "- TF-IDF\n"
        "- Cosine Similarity\n"
        "- RAG\n"
        "- Ollama\n"
        "- Llama 3.2"
    )

    st.divider()

    st.caption(
        "Answers are generated from retrieved "
        "document content."
    )




uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"]
)




if uploaded_file:

    try:

        raw_text = extract_text(uploaded_file)

        document_text = clean_text(raw_text)

        if not document_text:

            st.error(
                "No readable text found in this document."
            )

        else:

            chunks = split_text(document_text)

            st.success(
                f"Document uploaded successfully: "
                f"{uploaded_file.name}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Total Words",
                    len(document_text.split())
                )

            with col2:

                st.metric(
                    "Text Chunks",
                    len(chunks)
                )

            # -----------------------------------
            # 10. Ask Questions
            # -----------------------------------

            st.divider()

            st.subheader("Ask Your Document")

            question = st.text_input(
                "Enter your question",
                placeholder="What is this document about?"
            )

            if st.button(
                "Generate Answer",
                type="primary"
            ):

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Searching document..."
                    ):

                        retrieved = retrieve_chunks(
                            question,
                            chunks
                        )

                    if not retrieved:

                        st.warning(
                            "No relevant content was found "
                            "in the document."
                        )

                    else:

                        context = "\n\n".join(
                            item["text"]
                            for item in retrieved
                        )

                        with st.spinner(
                            "Generating answer using Llama 3.2..."
                        ):

                            answer = generate_answer(
                                question,
                                context
                            )

                        st.subheader("Answer")

                        st.write(answer)

                        st.subheader(
                            "Retrieved Document Sections"
                        )

                        for index, item in enumerate(
                            retrieved,
                            start=1
                        ):

                            with st.expander(
                                f"Relevant Section {index}"
                            ):

                                st.write(item["text"])

                                st.caption(
                                    f"Similarity: "
                                    f"{item['score']:.2f}"
                                )

    except Exception as error:

        st.error(
            f"Unable to process document: {error}"
        )


st.divider()

st.caption(
    "Developed using Python, Streamlit, "
    "Scikit-learn, PyPDF and Ollama."
)

st.caption(
    "Note: Answers depend on the extracted "
    "document content and retrieved text."
)