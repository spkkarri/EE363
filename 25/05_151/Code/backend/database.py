import sqlite3
import os

DB_PATH = "backend/users.db"  # Store in chatbot/backend/

def init_db():
    print("Initializing DB...")
    if not os.path.exists(DB_PATH):
        print("Creating users.db...")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Users table
        c.execute('''CREATE TABLE users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )''')
        # Sessions table
        c.execute('''CREATE TABLE sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )''')
        # Conversations table with session_id
        c.execute('''CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            username TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )''')
        allowed_users = [
            ("user1", "password1"),
            ("admin", "admin@123"),  # Admin user
            ("user2", "password2"),
            ("user3", "password3"),
            ("user4", "password4")
        ] 
        c.executemany("INSERT INTO users (username, password) VALUES (?, ?)", allowed_users)
        conn.commit()
        conn.close()
        print("DB initialized with default users.")
    else:
        print("DB already exists. No changes made.")

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def create_session(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (username) VALUES (?)", (username,))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_user_sessions(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT session_id, start_time FROM sessions WHERE username = ? ORDER BY start_time DESC", (username,))
    sessions = c.fetchall()
    conn.close()
    return [{"session_id": row[0], "start_time": row[1]} for row in sessions]

def get_conversation_history(username, session_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if session_id:
        c.execute("SELECT role, content FROM conversations WHERE username = ? AND session_id = ? AND role != 'think' ORDER BY timestamp", (username, session_id))
    else:
        c.execute("SELECT role, content FROM conversations WHERE username = ? AND role != 'think' ORDER BY timestamp", (username,))
    history = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in history]

def save_message(username, session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (username, session_id, role, content) VALUES (?, ?, ?, ?)", (username, session_id, role, content))
    conn.commit()
    conn.close()

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = c.fetchall()
    conn.close()
    return [user[0] for user in users]

def get_user_chats(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT session_id, role, content, timestamp FROM conversations WHERE username = ? AND role != 'think' ORDER BY session_id, timestamp", (username,))
    chats = c.fetchall()
    conn.close()
    return [{"session_id": row[0], "role": row[1], "content": row[2], "timestamp": row[3]} for row in chats]



