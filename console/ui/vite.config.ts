import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
  // 프로덕션 빌드: host/wwwroot로 출력
  build: {
    outDir: '../host/wwwroot',
    emptyOutDir: true
  },
  server: {
    port: 3000,
    proxy: {
      // API 엔드포인트
      '/api': {
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
