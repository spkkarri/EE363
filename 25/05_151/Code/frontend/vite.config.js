import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '172.50.8.158',
    port: 5173,
    proxy: {
      '/api': 'http://172.50.24.123:5000',
    }
  }
})
