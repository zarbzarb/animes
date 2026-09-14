<template>
  <div class="login-wrap">
    <div class="panel">
      <!-- 左侧品牌区 -->
      <div class="brand-side">
        <div class="logo">🌸 AniRec</div>
        <h2 class="slogan">懂你的动漫追番推荐</h2>
        <ul class="feats">
          <li><span class="ico">🎯</span>多兴趣模型 + 内容语义，双路融合推荐</li>
          <li><span class="ico">💬</span>追番助手对话式找番，理由可解释</li>
          <li><span class="ico">📈</span>兴趣漂移分析，口味变化看得见</li>
        </ul>
        <div class="illus">✦ ⋆ ✦ ⋆ ✦</div>
      </div>

      <!-- 右侧表单区 -->
      <div class="form-side">
        <div class="seg" role="tablist">
          <span class="slider" :class="{ right: mode === 'register' }" />
          <button type="button" :class="{ on: mode === 'login' }" @click="switchMode('login')">登 录</button>
          <button type="button" :class="{ on: mode === 'register' }" @click="switchMode('register')">注 册</button>
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
          <el-button type="primary" size="large" class="submit" :loading="busy" @click="doLogin">登 录</el-button>
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
          <el-button type="success" size="large" class="submit" :loading="busy" @click="doRegister">注 册</el-button>
        </el-form>
      </div>
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
  if (!v) return { label: '', pct: '0', color: '#c0c4cc' }
  let s = 0
  if (v.length >= 8) s++
  if (/[A-Za-z]/.test(v) && /\d/.test(v)) s++
  if (/[^A-Za-z0-9]/.test(v) || v.length >= 12) s++
  return [
    { label: '弱', pct: '33%', color: '#f56c6c' },
    { label: '中', pct: '66%', color: '#e6a23c' },
    { label: '强', pct: '100%', color: '#67c23a' },
  ][s - 1] || { label: '弱', pct: '33%', color: '#f56c6c' }
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
    await api.post('/api/v1/auth/register', {
      username: regForm.username,
      nickname: regForm.nickname || regForm.username,
      password: regForm.password,
    })
    ElMessage.success('注册成功，已自动登录')
    const d = await api.post('/api/v1/auth/login', {
      username: regForm.username,
      password: regForm.password,
      captcha_id: captchaId.value,
      captcha_code: loginForm.captcha,
    })
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
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #fdf0f5 0%, #eef3fb 60%, #f3eefb 100%);
  padding: 24px;
}
.panel {
  width: 840px;
  max-width: 100%;
  min-height: 480px;
  display: flex;
  background: #fff;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 18px 50px rgba(150, 100, 130, .16);
}
.brand-side {
  flex: 1;
  background: linear-gradient(160deg, var(--brand) 0%, #b95f90 100%);
  color: #fff;
  padding: 44px 36px;
  display: flex;
  flex-direction: column;
}
.logo { font-size: 26px; font-weight: 800; letter-spacing: 1px; }
.slogan { margin: 26px 0 18px; font-size: 22px; font-weight: 600; }
.feats { list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; font-size: 14px; opacity: .94; }
.feats .ico { margin-right: 8px; }
.illus { margin-top: auto; font-size: 15px; letter-spacing: 6px; opacity: .5; }

.form-side { width: 380px; padding: 40px 36px 28px; display: flex; flex-direction: column; }
.seg {
  position: relative;
  display: grid;
  grid-template-columns: 1fr 1fr;
  background: #f2f3f7;
  border-radius: 10px;
  padding: 4px;
  margin-bottom: 26px;
}
.seg button {
  position: relative;
  z-index: 1;
  border: 0;
  background: transparent;
  height: 36px;
  font-size: 15px;
  color: #909399;
  cursor: pointer;
  transition: color .2s;
}
.seg button.on { color: #fff; font-weight: 600; }
.slider {
  position: absolute;
  top: 4px; left: 4px;
  width: calc(50% - 4px);
  height: 36px;
  border-radius: 8px;
  background: var(--brand);
  transition: transform .25s ease;
}
.slider.right { transform: translateX(100%); }

.cap-row { display: flex; gap: 10px; width: 100%; }
.cap-img {
  width: 118px; height: 40px;
  border-radius: 6px;
  border: 1px solid #ebeef5;
  overflow: hidden;
  cursor: pointer;
  flex: none;
  line-height: 0;
}
.cap-img :deep(svg) { width: 100%; height: 100%; }

.submit { width: 100%; margin-top: 4px; }
.hint { text-align: center; color: #c0c4cc; font-size: 12px; margin: 14px 0 0; }
.strength { display: flex; align-items: center; gap: 8px; width: 100%; margin-top: 4px; }
.strength .bar { flex: 1; height: 4px; background: #ebeef5; border-radius: 2px; overflow: hidden; }
.strength .bar i { display: block; height: 100%; border-radius: 2px; transition: width .2s; }
.strength span { font-size: 12px; flex: none; }
</style>
