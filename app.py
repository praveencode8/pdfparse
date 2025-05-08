import streamlit as st
from audio_utils import analyze_audio_files
from pdf_chat_utlis import get_pdf_text, get_text_chunks, save_vector_store, handle_user_question

st.set_page_config(page_title="Multilingual AI Assistant", layout="wide")
st.title("🤖 AI Assistant")

# Sidebar Options
st.sidebar.title("Choose Tool")
mode = st.sidebar.radio("Select an option", ["📞 Audio Call Analysis", "📄 PDF Chatbot"])

if mode == "📞 Audio Call Analysis":
    st.sidebar.subheader("Upload Call Recordings")
    audio_files = st.sidebar.file_uploader(
        "Upload one or more audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True
    )

    if audio_files:
        with st.spinner("🔍 Transcribing & Summarizing..."):
            results, similarity_matrix = analyze_audio_files(audio_files)

        if len(audio_files) > 1:
            st.subheader("📊 Call Similarity Dashboard")
            st.write(similarity_matrix)

        st.subheader("📋 Call Transcripts and Summaries")
        for idx, res in enumerate(results):
            with st.expander(f"📞 Call {idx+1}: View Details", expanded=True):
                st.text_area("📝 Transcript", res['transcript'], height=200)
                if res['summary']:
                    st.markdown("📌 **Summary:**")
                    st.markdown(res['summary'])

elif mode == "📄 PDF Chatbot":
    st.sidebar.subheader("Upload PDFs")
    pdf_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        with st.spinner("📚 Reading PDFs..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            save_vector_store(text_chunks)
        st.success("✅ PDFs processed! You can now ask questions below.")

    st.subheader("Ask a Question")
    user_question = st.text_input("❓ Ask a question about the documents")

    if user_question:
        with st.spinner("🤖 Thinking..."):
            response = handle_user_question(user_question)
        st.markdown("💬 **Answer:**")
        st.write(response)