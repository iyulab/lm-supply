import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
  // Production build: output to host/wwwroot
  build: {
    outDir: '../host/wwwroot',
    emptyOutDir: true
  },
  server: {
    port: 3000,
    proxy: {
      // API endpoints (system, cache, registry)
      // Note: /api/download SSE endpoints connect directly to backend to avoid proxy issues
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      // OpenAI-compatible endpoints (chat, embed, audio, images, etc.)
      '/v1': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      // Swagger UI
      '/swagger': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      // Health check
      '/health': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  }
})
