import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8765',
      // NOTE: Vite 自身的 /static/ 路径（HMR、源码映射等）优先级高于此代理规则。
      // 只有未被 Vite 内置处理器匹配的 /static/ 请求才会被转发到后端。
      // 这意味着后端 /static/ 下的静态资源若与 Vite 内部路径同名，将被 Vite 遮蔽。
      '/static': 'http://localhost:8765',
    }
  },
  base: '/static/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
  }
})
