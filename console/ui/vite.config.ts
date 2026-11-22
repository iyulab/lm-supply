import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The host listens on :5000 by default; macOS keeps that port for AirPlay Receiver, so the target can be moved (API_BASE_URL is the same
// variable the e2e helpers read).
const apiTarget = process.env.API_BASE_URL ?? 'http://localhost:5000'

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
        target: apiTarget,
        changeOrigin: true
      },
      // OpenAI-compatible endpoints (chat, embed, audio, images, etc.)
      '/v1': {
        target: apiTarget,
        changeOrigin: true
      },
      // Swagger UI
      '/swagger': {
        target: apiTarget,
        changeOrigin: true
      },
      // Health check
      '/health': {
        target: apiTarget,
        changeOrigin: true
      }
    }
  }
})
