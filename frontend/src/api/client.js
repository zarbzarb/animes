import axios from 'axios'
import { ElMessage } from 'element-plus'
import { auth } from '../stores/auth'
import router from '../router'

/**
 * 统一信封客户端：后端约定 HTTP 恒 200、靠 body.code 区分成败
 * （0=成功；40101 未登录；40301 无权限；60401 部分降级 —— 降级不算失败，交给调用方看 meta.degraded）
 */
export const http = axios.create({ timeout: 30000 })

http.interceptors.request.use((cfg) => {
  if (auth.token) cfg.headers.Authorization = `Bearer ${auth.token}`
  return cfg
})

http.interceptors.response.use(
  (resp) => {
    const body = resp.data
    if (body && typeof body === 'object' && 'code' in body) {
      if (body.code === 40101) {
        auth.logout()
        router.push('/login')
        return Promise.reject(new Error('未登录'))
      }
      if (body.code !== 0 && body.code !== 60401) {
        ElMessage.error(body.message || `请求失败（code=${body.code}）`)
        return Promise.reject(new Error(body.message))
      }
      return body
    }
    return body
  },
  (err) => {
    // 404/422/500 等也被统一异常处理器包成信封
    const body = err.response?.data
    const msg = body?.message || err.message || '网络错误'
    if (body?.code === 40101) {
      auth.logout()
      router.push('/login')
    } else {
      ElMessage.error(msg)
    }
    return Promise.reject(new Error(msg))
  },
)

/** GET 便捷封装：直接返回 data 部分 */
export const api = {
  async get (url, params) {
    const body = await http.get(url, { params })
    return body.data
  },
  async post (url, data, config) {
    const body = await http.post(url, data, config)
    return body.data
  },
  async put (url, data) {
    const body = await http.put(url, data)
    return body.data
  },
  async del (url, params) {
    const body = await http.delete(url, { params })
    return body.data
  },
}
