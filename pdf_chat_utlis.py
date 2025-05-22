# pdf_utils.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from config import GOOGLE_API_KEY


# Extract raw text from PDF files
def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Fallback if extract_text() returns None
    return text


# Split text into chunks for embedding
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Create and save vector store
def save_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
You are a knowledgeable and professional assistant helping users understand AMC mutual fund fact sheets and financial documents.

Your Responsibilities:
- Answer only using the information present in the provided PDF context.
- If the answer involves **data, metrics, comparisons, holdings, fund characteristics, taxation, or categories** — always prefer presenting it in a **markdown table**.
- If the user uploads **multiple PDFs** and the question indicates a **comparison**, check the context of all PDFs and generate a **side-by-side comparison table** of relevant aspects (e.g., portfolio, performance, fund manager info, etc.).
- Prioritize extracting content from tables found in the PDFs.
- After any table, follow up with a **1-2 line concise summary** highlighting key insights or differences.
- For text-only answers, use **bullet points or numbered lists** where possible.
- Never assume or infer information not explicitly present in the documents.
- Maintain a **professional, concise tone** suitable for financial advisory.
- Keep answers **within 10 lines** unless the table demands more space.

---

Context: {context}

Chat History (previous questions and answers to help you stay on-topic):
{history}

Current Question:
{question}

Answer (use markdown tables wherever possible):
"""


    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "history", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user question
def handle_user_question(user_question, chat_history=[]):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Combine user chat history as a string to retain conversational context
    history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])

    chain = get_conversational_chain()
    response = chain(
        {
            "input_documents": docs,
            "history": history_str,
            "question": user_question
        },
        return_only_outputs=True
    )
    return response["output_text"]


