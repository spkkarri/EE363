import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import Latex from 'react-latex-next';
import 'katex/dist/katex.min.css';
import { IoSend } from 'react-icons/io5';
import FormulaWidget from './FormulaWidget';
import './Chat.css';
import { backendUrl } from './config';
console.log("Backend URL:", backendUrl); // Should log the correct URL

const parseMessageContent = (content) => {
  const parts = [];
  let remainingContent = content;

  const displayMathRegex = /\$\$([^$]+)\$\$/g;
  const inlineMathRegex = /\$([^$]+)\$/g;
  const imageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;

  let match;
  while ((match = displayMathRegex.exec(remainingContent)) !== null) {
    const before = remainingContent.substring(0, match.index);
    const formula = match[1];
    const after = remainingContent.substring(match.index + match[0].length);

    if (before) parts.push({ type: 'text', content: before });
    parts.push({ type: 'formula', content: `$$${formula}$$` });
    remainingContent = after;
  }

  let tempContent = remainingContent;
  remainingContent = '';
  while ((match = inlineMathRegex.exec(tempContent)) !== null) {
    const before = tempContent.substring(0, match.index);
    const formula = match[1];
    const after = tempContent.substring(match.index + match[0].length);

    if (before) parts.push({ type: 'text', content: before });
    parts.push({ type: 'inline-formula', content: `$${formula}$` });
    remainingContent = after;
    tempContent = remainingContent;
  }

  tempContent = remainingContent || tempContent;
  remainingContent = '';
  while ((match = imageRegex.exec(tempContent)) !== null) {
    const before = tempContent.substring(0, match.index);
    const imageUrl = match[2];
    const after = tempContent.substring(match.index + match[0].length);

    if (before) parts.push({ type: 'text', content: before });
    parts.push({ type: 'image', content: imageUrl });
    remainingContent = after;
    tempContent = remainingContent;
  }

  if (remainingContent || tempContent) {
    parts.push({ type: 'text', content: remainingContent || tempContent });
  }

  return parts;
};

// ✅ JWT Authorization header helper
const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  };
};

function Chat({ backendUrl, username, onLogout }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isCotEnabled, setIsCotEnabled] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [selectedSessionMessages, setSelectedSessionMessages] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const chatEndRef = useRef(null);
  const isGuest = username === 'guest';

  useEffect(() => {
    console.log(`Chat for ${username} using backend URL: ${backendUrl}`);
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [backendUrl, username, messages]);

  const fetchSessions = async () => {
    if (isGuest) return;
    try {
      const response = await axios.get(`${backendUrl}/api/sessions`, getAuthHeaders());
      setSessions(response.data);
    } catch (error) {
      handleAuthError(error, 'Error fetching sessions');
    }
  };

  const fetchHistory = async (sessionId) => {
    if (isGuest) return;
    try {
      const response = await axios.get(`${backendUrl}/api/history/${sessionId}`, getAuthHeaders());
      setSelectedSessionMessages(response.data);
    } catch (error) {
      handleAuthError(error, 'Error fetching history');
    }
  };

  const toggleCot = () => {
    setIsCotEnabled((prev) => !prev);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input, cot: isCotEnabled };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    try {
      const response = await axios.post(
        `${backendUrl}/api/chat`,
        { message: input, use_cot: isCotEnabled },
        getAuthHeaders()
      );
      const botMessage = { role: 'bot', content: response.data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      handleAuthError(error, 'Error sending message');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    onLogout();
  };

  const handleAuthError = (error, message) => {
    console.error(`${message}:`, error);
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      onLogout();
    } else {
      const errorMessage = { role: 'system', content: `Error: ${error.message}` };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const openHistory = () => {
    fetchSessions();
    setShowHistory(true);
  };

  const closeHistory = () => {
    setShowHistory(false);
    setSelectedSessionMessages([]);
  };

  return (
    <div className="container chat-container">
      <div className="chat-header">
        <div className="chat-title">ChatBot</div>
        {!isGuest && (
          <div className="button-group">
            <button onClick={openHistory} className="history-button">History</button>
            <button onClick={handleLogout} className="logout-button">Logout</button>
            <Link to="http://localhost:8501">
              <button className="new-button">RAG PDF Chatbot</button>
            </Link>
          </div>
        )}
      </div>

      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message-wrapper ${msg.role}`}>
            <div className={`message ${msg.role}`}>
              <strong>{msg.role}:</strong>{' '}
              {msg.role === 'bot' ? (
                <div>
                  {parseMessageContent(msg.content).map((part, partIndex) => {
                    if (part.type === 'text') {
                      return <ReactMarkdown key={partIndex}>{part.content}</ReactMarkdown>;
                    } else if (part.type === 'formula') {
                      return <FormulaWidget key={partIndex} formula={part.content} />;
                    } else if (part.type === 'inline-formula') {
                      return <Latex key={partIndex}>{part.content}</Latex>;
                    } else if (part.type === 'image') {
                      return <FormulaWidget key={partIndex} imageUrl={part.content} />;
                    }
                    return null;
                  })}
                </div>
              ) : (
                msg.content
              )}
            </div>
            {msg.role === 'user' && (
              <div className={`cot-balloon ${msg.cot ? 'highlighted' : ''}`} onClick={toggleCot}>
                CoT
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
        />
        <button onClick={sendMessage}>
          <IoSend />
        </button>
        <div
          className={`cot-balloon ${isCotEnabled ? 'highlighted' : ''}`}
          onClick={toggleCot}
        >
          CoT
        </div>
      </div>

      {showHistory && !isGuest && (
        <div className="history-modal">
          <div className="history-content">
            <h3>Chat History</h3>
            <button onClick={closeHistory} className="close-button">Close</button>
            <div className="session-list">
              <ul>
                {sessions.map((session) => (
                  <li
                    key={session.session_id}
                    onClick={() => fetchHistory(session.session_id)}
                  >
                    Session {session.session_id} - {new Date(session.start_time).toLocaleString()}
                  </li>
                ))}
              </ul>
            </div>
            <div className="session-messages">
              {selectedSessionMessages.map((msg, index) => (
                <div key={index} className={`message-wrapper ${msg.role}`}>
                  <div className={`message ${msg.role}`}>
                    <strong>{msg.role}:</strong>{' '}
                    {msg.role === 'bot' ? (
                      <div>
                        {parseMessageContent(msg.content).map((part, partIndex) => {
                          if (part.type === 'text') {
                            return <ReactMarkdown key={partIndex}>{part.content}</ReactMarkdown>;
                          } else if (part.type === 'formula') {
                            return <FormulaWidget key={partIndex} formula={part.content} />;
                          } else if (part.type === 'inline-formula') {
                            return <Latex key={partIndex}>{part.content}</Latex>;
                          } else if (part.type === 'image') {
                            return <FormulaWidget key={partIndex} imageUrl={part.content} />;
                          }
                          return null;
                        })}
                      </div>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Chat;
