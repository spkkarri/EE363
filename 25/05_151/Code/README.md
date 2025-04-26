
# RAG-Based Academic Chatbot 🤖📚

### 👥 Team Info
**Team Number:** 5  
**Roll Numbers:** 522151, 522153, 522112, 522250, 522213, 522148

---

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot designed for academic support. It allows users to upload PDFs, ask questions, and receive accurate answers powered by Google's Gemini API. The app includes user login, admin dashboard, chat history, and vector-based document retrieval.

---

## 🚀 Features

- Token-based authentication (JWT)
- Academic-level query answering with Gemini AI
- RAG system using uploaded PDFs and FAISS vector index
- PDF upload and parsing support
- Chat history and session tracking
- Admin panel with user/session insights
- Modern UI built with React + Vite

---

## 🗂️ Project Structure

```
.
├── assets
├── backend
│   ├── .env
│   ├── App.py
│   ├── config.py
│   ├── database.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── jwt_helper.py
│   ├── faiss_index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── __pycache__
│   │   ├── config.cpython-311.pyc
│   │   ├── config.cpython-312.pyc
│   │   ├── database.cpython-311.pyc
│   │   ├── database.cpython-312.pyc
│   │   ├── routes.cpython-311.pyc
│   │   └── routes.cpython-312.pyc
│   ├── Rag.py
│   ├── requirements.txt
│   ├── routes.py
│   └── users.db
├── frontend
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── src
│   │   ├── AdminPanel.css
│   │   ├── AdminPanel.jsx
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── Chat.css
│   │   ├── Chat.jsx
│   │   ├── config.js
│   │   ├── FormulaWidget.css
│   │   ├── FormulaWidget.jsx
│   │   ├── index.css
│   │   ├── Login.css
│   │   ├── Login.jsx
│   │   └── main.jsx
├── RAG - pdf
│   ├── faiss_index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── templates
│   │   └── RAG_app.html
│   ├── .env
│   ├── RAG_app.py
│   ├── requirements.txt
└── main.py
```

---

## 🧰 Prerequisites

- Python 3.9+
- npm or yarn
- Git (optional)

---

## 🛠️ Backend Setup (Flask + Gemini + FAISS)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Ensure `.env` contains:

```
GEMINI_API_KEY=your_google_generative_ai_key
```

Launch the full app:

```bash
python main.py
```

---


## ✅ Usage

- Log in or register to get started.
- Use the chatbot for academic learning and general queries.
- Navigate to the RAG PDF Vector Chatbot using the top navigation bar.
- Upload academic PDFs in the chat interface.
- After uploading a PDF, ask queries specifically related to the uploaded PDF topics.
- Admins can monitor user sessions and chat activities.

---

## ▶️ Demo Video

📺 📺 [Watch the Demo Video on Google Drive]
       (https://drive.google.com/file/d/1uqxKuo9SY2rbLNHu3BP1gIfMuL1uUTNT/view?usp=drive_link)

---

## 🧪 Deployment Notes

- Backend: Flask
- Frontend: React
- Use `.env` and CORS properly in production

---

## 📬 Questions?

Raise an issue or contribute!

---

> Built with ❤️ using Flask, React, and Gemini
