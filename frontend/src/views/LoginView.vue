<template>
  <div class="login-wrap">
    <!-- 背景装饰：两团极淡的品牌色光斑，克制 -->
    <div class="blob a" /><div class="blob b" />

    <div class="panel">
      <!-- 顶部品牌：B 站风小电视 Logo -->
      <div class="brand">
        <svg class="tv" viewBox="0 0 48 48" aria-hidden="true">
          <path class="ant" d="M14 4l7 7M34 4l-7 7" stroke-linecap="round" />
          <rect x="4" y="11" width="40" height="30" rx="7" />
          <rect class="eye" x="16" y="23" width="3.6" height="10" rx="1.8" />
          <rect class="eye" x="28.4" y="23" width="3.6" height="10" rx="1.8" />
        </svg>
        <div class="name">AniRec</div>
        <div class="slogan">懂你的动漫追番推荐</div>
      </div>

      <!-- 登录 / 注册：文字 Tab + 粉色下划线 -->
      <div class="tabs" role="tablist">
        <button type="button" class="tab" :class="{ on: mode === 'login' }"
          @click="switchMode('login')">登录</button>
        <button type="button" class="tab" :class="{ on: mode === 'register' }"
          @click="switchMode('register')">注册</button>
      </div>

      <!-- 登录 -->
      <el-form v-show="mode === 'login'" ref="loginFormRef" :model="loginForm" :rules="loginRules"
        label-position="top" @keyup.enter="doLogin">
        <el-form-item prop="username" label="用户名">
          <el-input v-model="loginForm.username" placeholder="用户名" size="large" :prefix-icon="User" />
        </el-form-item>
        <el-form-item prop="password" label="密码">
          <el-input v-model="loginForm.password" type="password" placeholder="密码" size="large"
            show-password :prefix-icon="Lock" />
        </el-form-item>
        <el-form-item prop="captcha" label="验证码">
          <div class="cap-row">
            <el-input v-model="loginForm.captcha" placeholder="右侧字符" size="large" maxlength="4" />
            <el-tooltip content="看不清？点击刷新" placement="top">
              <div class="cap-img" v-html="captchaSvg" @click="loadCaptcha" />
            </el-tooltip>
          </div>
        </el-form-item>
        <el-button type="primary" size="large" class="submit" :loading="busy" @click="doLogin">登录</el-button>
        <p class="hint">演示账号：anifan / P@ssw0rd</p>
      </el-form>

      <!-- 注册 -->
      <el-form v-show="mode === 'register'" ref="regFormRef" :model="regForm" :rules="regRules"
        label-position="top" @keyup.enter="doRegister">
        <el-form-item prop="username" label="用户名">
          <el-input v-model="regForm.username" placeholder="4-20 位字母/数字/下划线" size="large" :prefix-icon="User" />
        </el-form-item>
        <el-form-item prop="nickname" label="昵称">
          <el-input v-model="regForm.nickname" placeholder="怎么称呼你（可选）" size="large" :prefix-icon="UserFilled" />
        </el-form-item>
        <el-form-item prop="password" label="密码">
          <el-input v-model="regForm.password" type="password" placeholder="≥8 位，含字母和数字" size="large"
            show-password :prefix-icon="Lock" />
          <div v-if="strength.label" class="strength">
            <div class="bar"><i :style="{ width: strength.pct, background: strength.color }" /></div>
            <span :style="{ color: strength.color }">{{ strength.label }}</span>
          </div>
        </el-form-item>
        <el-form-item prop="confirm" label="确认密码">
          <el-input v-model="regForm.confirm" type="password" placeholder="再输一遍" size="large"
            show-password :prefix-icon="Lock" />
        </el-form-item>
        <el-button type="primary" size="large" class="submit" :loading="busy" @click="doRegister">注册</el-button>
      </el-form>
    </div>

    <!-- 卡片下方一句话卖点，替代原来的左栏特性列表 -->
    <div class="feats">
      <span>🎯 双路融合推荐</span><i />
      <span>💬 对话式找番</span><i />
      <span>📈 兴趣漂移分析</span>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, UserFilled, Lock } from '@element-plus/icons-vue'
import { api } from '../api/client'
import { auth } from '../stores/auth'

const route = useRoute()
const router = useRouter()
const mode = ref('login')
const busy = ref(false)
const captchaSvg = ref('')

const loginFormRef = ref(null)
const regFormRef = ref(null)
const loginForm = reactive({ username: '', password: '', captcha: '' })
const regForm = reactive({ username: '', nickname: '', password: '', confirm: '' })

/* 验证码状态（captcha_id 不放表单，提交时一并带上） */
const captchaId = ref('')

async function loadCaptcha () {
  loginForm.captcha = ''
  try {
    const d = await api.get('/api/v1/auth/captcha')
    captchaId.value = d.captcha_id
    captchaSvg.value = d.svg
  } catch { /* 拦截器已提示 */ }
}

function switchMode (m) {
  mode.value = m
}

/* 校验规则：错误就地显示在表单项下，不用弹窗 */
const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
  captcha: [
    { required: true, message: '请输入验证码', trigger: 'blur' },
    { min: 4, max: 4, message: '4 个字符', trigger: 'blur' },
  ],
}
const regRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { pattern: /^[A-Za-z0-9_]{4,20}$/, message: '4-20 位字母/数字/下划线', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 8, max: 64, message: '至少 8 位', trigger: 'blur' },
    {
      validator: (_, v, cb) => {
        if (!v || (/[A-Za-z]/.test(v) && /\d/.test(v))) cb()
        else cb(new Error('需同时包含字母与数字'))
      },
      trigger: 'blur',
    },
  ],
  confirm: [
    { required: true, message: '请再输入一遍密码', trigger: 'blur' },
    {
      validator: (_, v, cb) => {
        if (v === regForm.password) cb()
        else cb(new Error('两次输入的密码不一致'))
      },
      trigger: 'blur',
    },
  ],
}

/* 密码强度（纯前端提示，三条档） */
const strength = computed(() => {
  const v = regForm.password || ''
  if (!v) return { label: '', pct: '0', color: '#c9ccd0' }
  let s = 0
  if (v.length >= 8) s++
  if (/[A-Za-z]/.test(v) && /\d/.test(v)) s++
  if (/[^A-Za-z0-9]/.test(v) || v.length >= 12) s++
  return [
    { label: '弱', pct: '33%', color: '#f49d9d' },
    { label: '中', pct: '66%', color: '#ffab5e' },
    { label: '强', pct: '100%', color: '#2ac864' },
  ][s - 1] || { label: '弱', pct: '33%', color: '#f49d9d' }
})

async function doLogin () {
  const ok = await loginFormRef.value.validate().then(() => true).catch(() => false)
  if (!ok) return
  busy.value = true
  try {
    const d = await api.post('/api/v1/auth/login', {
      username: loginForm.username,
      password: loginForm.password,
      captcha_id: captchaId.value,
      captcha_code: loginForm.captcha,
    })
    ElMessage.success('欢迎回来！')
    auth.setToken(d.access_token)
    auth.setUser(d.user)
    router.push(route.query.redirect || '/home')
  } catch {
    // 提示已由拦截器弹出；验证码是一次性的，无论失败原因都换一张
    loadCaptcha()
  } finally { busy.value = false }
}

async function doRegister () {
  const ok = await regFormRef.value.validate().then(() => true).catch(() => false)
  if (!ok) return
  busy.value = true
  try {
    // 注册接口本身就返回 token bundle，直接登录态，不再二次调用 login ——
    // 之前自动登录带上了登录页残留的 captcha_id（一次一密必然校验失败），
    // 用户就会看到「注册成功」+「验证码过期」同时出现
    const d = await api.post('/api/v1/auth/register', {
      username: regForm.username,
      nickname: regForm.nickname || regForm.username,
      password: regForm.password,
    })
    ElMessage.success('注册成功，已自动登录')
    auth.setToken(d.access_token)
    auth.setUser(d.user)
    router.push('/home')
  } catch { /* 拦截器已提示（用户名重复等） */ } finally { busy.value = false }
}

onMounted(loadCaptcha)
</script>

<style scoped>
.login-wrap {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--bg-page);
  position: relative;
  padding: 24px;
  overflow: hidden;
}
/* 极淡的品牌色光斑，只做氛围不做主角 */
.blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: .18;
  pointer-events: none;
}
.blob.a { width: 420px; height: 420px; background: var(--brand); top: -120px; left: -80px; }
.blob.b { width: 380px; height: 380px; background: var(--info-blue); bottom: -140px; right: -60px; opacity: .10; }

.panel {
  position: relative;
  z-index: 1;
  width: 420px;
  max-width: 100%;
  background: #fff;
  border-radius: 12px;
  padding: 36px 40px 28px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, .06);
}

.brand { text-align: center; margin-bottom: 20px; }
.tv { width: 52px; height: 52px; }
.tv rect { fill: var(--brand); }
.tv .ant { stroke: var(--brand); stroke-width: 3.5; }
.tv .eye { fill: #fff; }
.name { font-size: 24px; font-weight: 700; color: var(--text-1); letter-spacing: .5px; margin-top: 8px; }
.slogan { font-size: 13px; color: var(--text-3); margin-top: 4px; }

.tabs {
  display: flex;
  gap: 32px;
  justify-content: center;
  margin-bottom: 20px;
}
.tab {
  position: relative;
  border: 0;
  background: none;
  padding: 6px 2px 10px;
  font-size: 16px;
  color: var(--text-3);
  cursor: pointer;
  transition: color .2s;
}
.tab:hover { color: var(--text-1); }
.tab.on { color: var(--text-1); font-weight: 600; }
.tab.on::after {
  content: '';
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  bottom: 0;
  width: 28px;
  height: 3px;
  border-radius: 2px;
  background: var(--brand);
}

.cap-row { display: flex; gap: 10px; width: 100%; }
.cap-img {
  width: 118px; height: 40px;
  border-radius: 6px;
  border: 1px solid var(--line);
  overflow: hidden;
  cursor: pointer;
  flex: none;
  line-height: 0;
}
.cap-img :deep(svg) { width: 100%; height: 100%; }

.submit { width: 100%; margin-top: 4px; }
.hint { text-align: center; color: var(--text-3); font-size: 12px; margin: 14px 0 0; }
.strength { display: flex; align-items: center; gap: 8px; width: 100%; margin-top: 4px; }
.strength .bar { flex: 1; height: 4px; background: #f1f2f3; border-radius: 2px; overflow: hidden; }
.strength .bar i { display: block; height: 100%; border-radius: 2px; transition: width .2s; }
.strength span { font-size: 12px; flex: none; }

.feats {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 22px;
  font-size: 13px;
  color: var(--text-3);
}
.feats i { width: 1px; height: 12px; background: var(--line); }
</style>
