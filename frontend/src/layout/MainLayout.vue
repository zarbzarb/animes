<template>
  <el-container class="layout">
    <el-aside width="200px" class="aside">
      <div class="logo">
        <svg class="tv" viewBox="0 0 48 48" aria-hidden="true">
          <path class="ant" d="M14 4l7 7M34 4l-7 7" stroke-linecap="round" />
          <rect x="4" y="11" width="40" height="30" rx="7" />
          <rect class="eye" x="16" y="23" width="3.6" height="10" rx="1.8" />
          <rect class="eye" x="28.4" y="23" width="3.6" height="10" rx="1.8" />
        </svg>
        <span>AniRec</span>
      </div>
      <el-menu :default-active="$route.path" router class="menu">
        <el-menu-item index="/home"><el-icon><HomeFilled /></el-icon>首页推荐</el-menu-item>
        <el-menu-item index="/library"><el-icon><Search /></el-icon>番剧库</el-menu-item>
        <el-menu-item index="/records"><el-icon><Collection /></el-icon>我的追番</el-menu-item>
        <el-menu-item index="/new"><el-icon><Sunny /></el-icon>新番专区</el-menu-item>
        <el-menu-item index="/interest"><el-icon><DataAnalysis /></el-icon>兴趣分析</el-menu-item>
        <el-menu-item index="/chat"><el-icon><ChatDotRound /></el-icon>追番助手</el-menu-item>
        <template v-if="auth.isAdmin">
          <div class="menu-group">管理端</div>
          <el-menu-item index="/admin/animes"><el-icon><Film /></el-icon>番库管理</el-menu-item>
          <el-menu-item index="/admin/users"><el-icon><User /></el-icon>用户管理</el-menu-item>
          <el-menu-item index="/admin/monitor"><el-icon><Monitor /></el-icon>效果监控</el-menu-item>
        </template>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="header">
        <span class="page-title">{{ $route.meta.title || '' }}</span>
        <el-dropdown v-if="auth.isLogin" @command="onCmd">
          <span class="who">
            <el-avatar :size="28" :src="avatarUrl" class="ava">
              {{ (auth.user?.nickname || auth.user?.username || '？').slice(0, 1) }}
            </el-avatar>
            {{ auth.user?.nickname || auth.user?.username }}
            <el-tag v-if="auth.isAdmin" size="small" type="danger" effect="plain">管理员</el-tag></span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="avatar">更换头像</el-dropdown-item>
              <el-dropdown-item divided command="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <!-- 头像上传：原生 file input 走 multipart，接口只收 jpg/png/webp/gif ≤2MB -->
        <input ref="fileEl" type="file" accept="image/jpeg,image/png,image/webp,image/gif"
          style="display: none" @change="uploadAvatar" />
      </el-header>
      <el-main class="main"><router-view :key="$route.fullPath" /></el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, computed } from 'vue'
import { HomeFilled, Collection, Sunny, DataAnalysis, ChatDotRound, Film, User, Monitor, Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { auth } from '../stores/auth'
import { http } from '../api/client'
import { useRouter } from 'vue-router'

const router = useRouter()
const fileEl = ref(null)
const avatarUrl = computed(() => auth.user?.avatar_url || '')

async function onCmd (cmd) {
  if (cmd === 'avatar') {
    fileEl.value?.click()
    return
  }
  if (cmd !== 'logout') return
  try { await http.post('/api/v1/auth/logout') } catch { /* 已退出也无所谓 */ }
  auth.logout()
  router.push('/login')
}

async function uploadAvatar (ev) {
  const file = ev.target.files?.[0]
  ev.target.value = ''          // 允许连选同一张图
  if (!file) return
  if (file.size > 2 * 1024 * 1024) {
    ElMessage.warning('头像不能超过 2MB'); return
  }
  const fd = new FormData()
  fd.append('file', file)
  const d = await http.post('/api/v1/users/me/avatar', fd,
    { headers: { 'Content-Type': 'multipart/form-data' } })
  auth.setUser({ ...auth.user, avatar_url: d.data.avatar_url })
  ElMessage.success('头像已更新')
}
</script>

<style scoped>
.layout { height: 100vh; }
.aside { background: #fff; border-right: 1px solid #ebeef5; }
.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: 700;
  color: var(--text-1);
  padding: 18px 20px;
}
.logo .tv { width: 26px; height: 26px; }
.logo .tv rect { fill: var(--brand); }
.logo .tv .ant { stroke: var(--brand); stroke-width: 4; }
.logo .tv .eye { fill: #fff; }
.menu { border-right: none; }
.menu-group { font-size: 12px; color: #c0c4cc; padding: 14px 20px 4px; }
.header { background: #fff; border-bottom: 1px solid #ebeef5; display: flex; align-items: center; justify-content: space-between; }
.who { cursor: pointer; color: #606266; display: flex; align-items: center; gap: 6px; }
.ava { background: var(--brand); flex-shrink: 0; }
.main { background: #f5f6fa; overflow-y: auto; }
</style>
