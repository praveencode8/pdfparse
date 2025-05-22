import streamlit as st
from audio_utils import analyze_audio_files
from pdf_chat_utlis import get_pdf_text, get_text_chunks, save_vector_store, handle_user_question
from transformers import pipeline
from collections import Counter
import pandas as pd

# Load sentiment analyzer once
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI
st.set_page_config(page_title="Multilingual AI Assistant", layout="wide")
st.title("🤖 AI Assistant")

st.sidebar.title("Choose Tool")
mode = st.sidebar.radio("Select an option", ["📞 Audio Call Analysis", "📄 PDF Chatbot"])

# ---------------------- AUDIO ANALYSIS MODE ------------------------
if mode == "📞 Audio Call Analysis":
    st.sidebar.subheader("Upload Call Recordings")
    audio_files = st.sidebar.file_uploader(
        "Upload one or more audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True
    )

    # Inside 📞 Audio Call Analysis block
    if "results" not in st.session_state and audio_files:
        with st.spinner("🔍 Transcribing & Summarizing..."):
            results, similarity_matrix = analyze_audio_files(audio_files)
            st.session_state.results = results

    if "results" in st.session_state:
        results = st.session_state.results

        st.subheader("📌 Call Summaries")
        for idx, res in enumerate(results):
            st.subheader(f"📞 Call {idx + 1}")

            # Summary
            st.markdown("📝 **Call Summary:**")
            st.markdown(res['summary'])

            # Sentiment
            st.info(f"🧠 **Sentiment:** {res['sentiment']}")

            # View Transcript Option
            with st.expander("🔍 View Full Transcript"):
                st.text_area("Transcript", res['transcript'], height=250, label_visibility="collapsed")



# ---------------------- PDF CHATBOT MODE ------------------------
elif mode == "📄 PDF Chatbot":
    st.sidebar.subheader("Upload PDFs")
    pdf_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("Submit & Process"):
        with st.spinner("📚 Reading PDFs..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            save_vector_store(text_chunks)
        st.success("✅ PDFs processed! You can now start chatting.")
        st.session_state.messages = []

    st.subheader("📨 Chat with your Documents")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the documents")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                chat_history = [
                    (msg["content"], st.session_state.messages[i + 1]["content"])
                    for i, msg in enumerate(st.session_state.messages[:-1])
                    if msg["role"] == "user"
                ]
                response = handle_user_question(user_input, chat_history=chat_history)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})