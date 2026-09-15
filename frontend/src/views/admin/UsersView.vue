<template>
  <div>
    <div class="toolbar">
      <el-input v-model="keyword" placeholder="用户名/昵称搜索" class="kw" clearable @keyup.enter="load" />
      <el-button type="primary" @click="load">搜索</el-button>
    </div>
    <el-table :data="rows" v-loading="busy" stripe>
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="username" label="用户名" min-width="140" />
      <el-table-column prop="nickname" label="昵称" min-width="140" />
      <el-table-column label="角色" width="100">
        <template #default="{ row }">
          <el-tag size="small" :type="row.role === 1 ? 'danger' : 'info'">
            {{ row.role === 1 ? '管理员' : '用户' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="last_login_at" label="最近登录" width="170">
        <template #default="{ row }">{{ row.last_login_at || '从未登录' }}</template>
      </el-table-column>
    </el-table>
    <el-pagination class="pager" layout="prev, pager, next, total" :total="total"
      :page-size="size" v-model:current-page="page" @current-change="load" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { api } from '../../api/client'

const rows = ref([]); const total = ref(0); const page = ref(1); const size = 20
const keyword = ref(''); const busy = ref(false)

async function load () {
  busy.value = true
  try {
    const d = await api.get('/api/v1/admin/users', { keyword: keyword.value, page: page.value, size })
    rows.value = d.list || []; total.value = d.total || 0
  } finally { busy.value = false }
}
onMounted(load)
</script>

<style scoped>
.toolbar { display: flex; gap: 12px; margin-bottom: 12px; }
.kw { width: 260px; }
.pager { margin-top: 14px; justify-content: flex-end; }
</style>
