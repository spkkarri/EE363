from flask import request, jsonify
import requests
from functools import wraps
from flask import Flask, jsonify

from database import (
    init_db, get_user, get_conversation_history, save_message,
    create_session, get_user_sessions, get_all_users, get_user_chats
)

from config import LLM_MODEL, OLLAMA_URL
from utils.jwt_helper import generate_token, verify_token

def init_routes(app):
    init_db()
    print("Routes initializing")

    def require_auth(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid token"}), 401
            
            token = auth_header.split(" ")[1]
            decoded = verify_token(token)
            if "error" in decoded:
                return jsonify(decoded), 401

            request.user = decoded
            return f(*args, **kwargs)
        return decorated

    @app.route('/api/login', methods=['POST'])
    def login():
        data = request.json
        username = data.get('username')
        password = data.get('password')

        user = get_user(username)
        if user and user[1] == password:
            is_admin = (username == "admin")
            session_id = None
            if not is_admin:
                session_id = create_session(username)

            token = generate_token(username, is_admin)
            return jsonify({
                "token": token,
                "username": username,
                "is_admin": is_admin,
                "session_id": session_id
            })

        return jsonify({"error": "Invalid credentials"}), 401

    @app.route("/api/test")
    def test():
        return jsonify({"message": "Test route hit!"})

    @app.route('/api/sessions', methods=['GET'])
    @require_auth
    def get_sessions():
        username = request.user['username']
        return jsonify(get_user_sessions(username))

    @app.route('/api/history/<int:session_id>', methods=['GET'])
    @require_auth
    def get_history(session_id):
        username = request.user['username']
        return jsonify(get_conversation_history(username, session_id))

    @app.route('/api/users', methods=['GET'])
    @require_auth
    def get_users():
        if not request.user['is_admin']:
            return jsonify({"error": "Unauthorized"}), 403
        return jsonify(get_all_users())

    @app.route('/api/user_chats/<username>', methods=['GET'])
    @require_auth
    def get_user_chats_route(username):
        if not request.user['is_admin']:
            return jsonify({"error": "Unauthorized"}), 403
        return jsonify(get_user_chats(username))

    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        user_message = data.get('message')
        use_cot = data.get('use_cot', False)

        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        decoded = verify_token(token) if token else {}

        if 'username' in decoded:
            username = decoded['username']
            session_id = data.get('session_id')
        else:
            username = "guest"
            session_id = 0
            print("No valid token found. Treating as guest.")

        if username != "guest":
            save_message(username, session_id, "user", user_message)
            conversation_history = get_conversation_history(username, session_id)
        else:
            conversation_history = []

        history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history])


        prompt = (
            "You are an AI assistant designed for an academic audience, ranging from undergraduates to PhD scholars. "
        )
        if use_cot:
            prompt += (
                "Provide detailed, technical, and well-structured responses suitable for this audience. "
                "Use precise terminology, include relevant concepts, algorithms, and applications, and organize your response with sections or bullet points where appropriate. "
                "When including mathematical expressions, use standard LaTeX syntax with proper delimiters: use \\(...\\) for inline math and \\[...\\] for display math. "
                "Do not use custom formats like [...] for equations; always use LaTeX delimiters. "
                "Use a step-by-step Chain of Thought (CoT) approach to arrive at a logical and accurate answer, and include your reasoning in a <think> tag. "
                "The <think> content should not be part of the final response sent to the user; it will be stored separately in the database. "
                "After your main response, append 1-2 follow-up or suggestive questions in Markdown format (e.g., **Follow-up Questions:** followed by a bullet list) to encourage deeper exploration or learning.\n\n"
                f"Previous conversation:\n{history}\n\n"
                f"User question: {user_message}"
            )

        prompt += f"\n\nPrevious conversation:\n{history}\n\nUser question: {user_message}"

        try:
            payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()

            full_response = response.json().get('response', 'Error processing request')

            think_content = ""
            actual_response = full_response
            if '<think>' in full_response and '</think>' in full_response:
                start = full_response.find('<think>')
                end = full_response.find('</think>') + len('</think>')
                think_content = full_response[start:end]
                actual_response = (full_response[:start] + full_response[end:]).strip()

            if username != "guest":
                if think_content:
                    save_message(username, session_id, "think", think_content)
                save_message(username, session_id, "bot", actual_response)

            return jsonify({"response": actual_response})

        except Exception as e:
            if username != "guest":
                save_message(username, session_id, "system", str(e))
            return jsonify({"error": str(e)}), 500
    

    @app.route('/api/logout', methods=['POST'])
    def logout():
        # With JWT, logout is frontend-controlled (just discard token)
        return jsonify({"message": "Logout successful. Please delete token on client side."})
