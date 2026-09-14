<template>
  <div class="login-wrap">
    <el-card class="login-card">
      <h2 class="brand">🌸 AniRec</h2>
      <p class="sub">多智能体动漫追番推荐系统</p>
      <el-tabs v-model="tab">
        <el-tab-pane label="登录" name="login">
          <el-form @keyup.enter="doLogin">
            <el-form-item>
              <el-input v-model="form.username" placeholder="用户名" size="large" />
            </el-form-item>
            <el-form-item>
              <el-input v-model="form.password" type="password" placeholder="密码" size="large" show-password />
            </el-form-item>
            <el-button type="primary" size="large" class="w100" :loading="busy" @click="doLogin">登 录</el-button>
          </el-form>
        </el-tab-pane>
        <el-tab-pane label="注册" name="register">
          <el-form @keyup.enter="doRegister">
            <el-form-item><el-input v-model="reg.username" placeholder="用户名（3-20 字符）" size="large" /></el-form-item>
            <el-form-item><el-input v-model="reg.nickname" placeholder="昵称（可选）" size="large" /></el-form-item>
            <el-form-item><el-input v-model="reg.password" type="password" placeholder="密码（≥8 位）" size="large" show-password /></el-form-item>
            <el-button type="success" size="large" class="w100" :loading="busy" @click="doRegister">注 册</el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { http, api } from '../api/client'
import { auth } from '../stores/auth'

const route = useRoute()
const router = useRouter()
const tab = ref('login')
const busy = ref(false)
const form = reactive({ username: '', password: '' })
const reg = reactive({ username: '', nickname: '', password: '' })

async function doLogin () {
  if (!form.username || !form.password) return ElMessage.warning('请输入用户名和密码')
  busy.value = true
  try {
    const d = await api.post('/api/v1/auth/login', { ...form })
    auth.setToken(d.access_token)
    auth.setUser(d.user)
    router.push(route.query.redirect || '/home')
  } finally { busy.value = false }
}

async function doRegister () {
  busy.value = true
  try {
    await api.post('/api/v1/auth/register', { ...reg })
    ElMessage.success('注册成功，已自动登录')
    const d = await api.post('/api/v1/auth/login', { username: reg.username, password: reg.password })
    auth.setToken(d.access_token)
    auth.setUser(d.user)
    router.push('/home')
  } finally { busy.value = false }
}
</script>

<style scoped>
.login-wrap { height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #fdf0f5, #eef3fb); }
.login-card { width: 380px; padding: 8px 12px 4px; }
.brand { text-align: center; color: var(--brand); margin: 8px 0 0; }
.sub { text-align: center; color: #909399; font-size: 13px; margin: 4px 0 16px; }
.w100 { width: 100%; }
</style>
