from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app)  # Allow frontend to access the API


@app.route('/')
def home():
    return render_template('RAG_app.html')  # Serves index.html from /templates folder


# === PDF TEXT EXTRACTION ===
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# === TEXT CHUNKING ===
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


# === VECTOR STORE ===
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# === QA CHAIN ===
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details. If the answer is not in the
    provided context, just say, "Answer is not available in the context," and don't provide 
    a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# === API: Upload and Process PDFs ===
@app.route("/api/upload_pdf", methods=["POST"])
def upload_pdf():
    if "pdfs" not in request.files:
        return jsonify({"error": "No PDF files provided"}), 400

    pdf_files = request.files.getlist("pdfs")
    text = get_pdf_text(pdf_files)
    chunks = get_text_chunks(text)
    get_vector_store(chunks)

    return jsonify({"message": "PDFs processed and vector store saved."})


# === API: Query with Question ===
@app.route("/api/query", methods=["POST"])
def query_pdf():
    data = request.get_json()
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "Question is required."}), 400

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return jsonify({"answer": response["output_text"]})

#def open_browser():
#    webbrowser.open_new("http://localhost:8501")

if __name__ == "__main__":
    #threading.Timer(1.0, open_browser).start()  # Open browser after 1 sec
    app.run(host="0.0.0.0", port=8501, debug=False)  # Set debug=True for development
