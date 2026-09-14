import { reactive } from 'vue'

function load () {
  try { return JSON.parse(localStorage.getItem('anirec.auth') || 'null') } catch { return null }
}

const saved = load()

/** 模块级单例 store（免 pinia 的极简写法） */
export const auth = reactive({
  token: saved?.token || '',
  user: saved?.user || null,

  get isLogin () { return !!this.token },
  get isAdmin () { return this.user?.role === 1 },

  save () {
    localStorage.setItem('anirec.auth', JSON.stringify({ token: this.token, user: this.user }))
  },

  setToken (t) { this.token = t; this.save() },
  setUser (u) { this.user = u; this.save() },
  logout () { this.token = ''; this.user = null; localStorage.removeItem('anirec.auth') },
})
