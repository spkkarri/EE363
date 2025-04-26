import { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './AdminPanel.css';

function AdminPanel({ backendUrl, onLogout }) {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [userChats, setUserChats] = useState([]);

  const token = localStorage.getItem('token');

  useEffect(() => {
    if (backendUrl && token) {
      fetchUsers();
    }
  }, [backendUrl]);

  const fetchUsers = async () => {
    try {
      const response = await axios.get(`${backendUrl}/api/users`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchUserChats = async (username) => {
    try {
      const response = await axios.get(`${backendUrl}/api/user_chats/${username}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUserChats(response.data);
      setSelectedUser(username);
    } catch (error) {
      console.error('Error fetching user chats:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    onLogout();
  };

  return (
    <div className="container admin-panel">
      <div className="admin-header">
        <h2>Admin Panel</h2>
        <button onClick={handleLogout} className="logout-button">Logout</button>
      </div>
      <div className="admin-content">
        <div className="user-list">
          <h3>Users</h3>
          <ul>
            {users.map((user) => (
              <li
                key={user}
                onClick={() => fetchUserChats(user)}
                className={selectedUser === user ? 'active' : ''}
              >
                {user}
              </li>
            ))}
          </ul>
        </div>
        <div className="user-chats">
          <h3>{selectedUser ? `${selectedUser}'s Chats` : 'Select a user'}</h3>
          {userChats.map((chat, index) => (
            <div key={index} className="chat-session">
              {index === 0 || chat.session_id !== userChats[index - 1].session_id ? (
                <h4>Session {chat.session_id} - {new Date(chat.timestamp).toLocaleString()}</h4>
              ) : null}
              <div className={`message-wrapper ${chat.role}`}>
                <div
                  className={`message ${chat.role === 'user' ? 'user' : chat.role === 'bot' ? 'bot' : 'system'}`}
                >
                  <strong>{chat.role}:</strong>{' '}
                  {chat.role === 'bot' ? (
                    <ReactMarkdown>{chat.content}</ReactMarkdown>
                  ) : (
                    chat.content
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AdminPanel;

