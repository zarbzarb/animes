import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import './styles.css'
import { auth } from './stores/auth'
import { api } from './api/client'

const app = createApp(App)
app.use(router)
app.use(ElementPlus, { locale: zhCn })
app.mount('#app')

// 恢复会话：有 token 就拉一次 /users/me（失败即清 token 回登录页）
if (auth.token) {
  api.get('/api/v1/users/me').then((d) => auth.setUser(d)).catch(() => auth.logout())
}
