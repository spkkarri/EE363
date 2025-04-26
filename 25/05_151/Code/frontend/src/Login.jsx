import { useState } from 'react';
import './Login.css';
import botImage from './assets/5.png'; // Use your actual image path
import logo from './assets/1.webp'; // Add your logo image here
import { backendUrl } from './config';

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch('http://127.0.0.1:5000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await res.json();
      if (res.ok) {
        localStorage.setItem('token', data.token);
        onLogin(data.username, data.is_admin);
      } else {
        setError(data.error || 'Login failed');
      }
    } catch (err) {
      setError('Network error');
    }
  };

  return (
    <div className="login-page">
      <div className="form-side">
        <div className="login-container">
          <div className="login-header">
            <img src={logo} alt="Logo" className="login-logo" />
            <h2>Login</h2>
          </div>

          <form onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />

            <div className="extras">
              <label>
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={() => setRememberMe(!rememberMe)}
                />
                Remember Me
              </label>
              <button type="button" className="guest-btn" onClick={() => onLogin('guest', false)}>
                Continue as Guest
              </button>
            </div>

            <button type="submit">Login</button>
          </form>

          <div className="register">
            Don’t have an account? <a href="#">Register</a>
          </div>

          {error && <div className="error">{error}</div>}
        </div>
      </div>
    </div>
  );
};

export default Login;
