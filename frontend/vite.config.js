import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// dev server 把 /api 代理到本地 FastAPI（8000），浏览器视角同源，免 CORS 配置
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
