import { useState, useEffect } from 'react';
import Login from './Login';
import Chat from './Chat';
import AdminPanel from './AdminPanel';
import './App.css';
import { BrowserRouter } from 'react-router-dom';
import { backendUrl } from './config';

const LOCAL_IP = "127.0.0.1";
const FRONTEND_LOCAL_URL = `http://${LOCAL_IP}:5173`;
const BACKEND_LOCAL_URL = `http://${LOCAL_IP}:5000`;
const BACKEND_NGROK_URL = "https://699e-61-0-228-101.ngrok-free.app";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [backendUrl, setBackendUrl] = useState(null);

  useEffect(() => {
    const hostname = window.location.hostname;
    if (hostname === LOCAL_IP || hostname === 'localhost' || hostname === '127.0.0.1') {
      setBackendUrl(BACKEND_LOCAL_URL);
    } else {
      setBackendUrl(BACKEND_NGROK_URL);
    }

    // Check for existing token on load
    const storedToken = localStorage.getItem('token') || sessionStorage.getItem('token');
    if (storedToken) {
      // Optionally: Validate token on backend before assuming valid
      const userData = JSON.parse(atob(storedToken.split('.')[1]));
      setUsername(userData.username);
      setIsAdmin(userData.is_admin);
      setIsLoggedIn(true);
    }
  }, []);

  const handleLogin = (user, isAdminFlag, token, session_id) => {
    setUsername(user);
    setIsAdmin(isAdminFlag);
    setSessionId(session_id);
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername('');
    setIsAdmin(false);
    setSessionId(null);
    localStorage.removeItem('token');
    sessionStorage.removeItem('token');
  };

  if (!backendUrl) {
    return <div>Loading backend configuration...</div>;
  }

  return (
    <BrowserRouter>
    <div className="app">
      {isLoggedIn ? (
        isAdmin ? (
          <AdminPanel backendUrl={backendUrl} onLogout={handleLogout} />
        ) : (
          <Chat backendUrl={backendUrl} username={username} sessionId={sessionId} onLogout={handleLogout} />
        )
      ) : (
        <Login backendUrl={backendUrl} onLogin={handleLogin} />
      )}
    </div>
    </BrowserRouter>
  );
}

export default App;
