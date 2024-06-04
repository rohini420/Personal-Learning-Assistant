import streamlit as st
import pdfplumber
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import hashlib
import shelve

# Load environment variables
load_dotenv()

def extract_relevant_text(text, query):
    sentences = text.split('.')
    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence for keyword in query.split())]
    return ' '.join(relevant_sentences[:10])

def process_pdf(uploaded_file):
    text = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text.append(extracted_text)
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return None
    return ' '.join(text)

def call_openai(prompt, cache_db='cache.db'):
    """Call the OpenAI model with caching and optimized token management."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "API key not found. Please check your .env file."

    hash_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    with shelve.open(cache_db) as cache:
        if hash_key in cache:
            return cache[hash_key]  # Return cached response

        llm = OpenAI(openai_api_key=api_key, temperature=0.5, max_tokens=800)
        response = llm(prompt)
        if isinstance(response, dict) and 'choices' in response and response['choices']:
            text_response = response['choices'][0]['text'].strip()
            cache[hash_key] = text_response  # Cache the response
            return text_response
        elif isinstance(response, str):
            return response.strip()
        else:
            return "Unexpected response format."

def main():
    st.title("Personal Learning Assistant")
    st.header("Prep-by-PDF")

    uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Processing PDF...'):
            full_text = process_pdf(uploaded_file)
            if not full_text:
                return

    question = st.text_input("Enter your question here:")
    if st.button("Get Answer", key="get_answer"):
        if full_text and question:
            relevant_text = extract_relevant_text(full_text, question)
            with st.spinner('Getting your answer...'):
                prompt = f"Based on the following text, answer this question: {question}\n\n---\n\n{relevant_text}"
                answer = call_openai(prompt)
                st.text_area("Answer", answer, height=300)
        else:
            st.error("Please upload a PDF and enter a question.")

if __name__ == "__main__":
    main()
