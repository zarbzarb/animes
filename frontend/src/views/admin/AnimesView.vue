<template>
  <div>
    <div class="toolbar">
      <el-input v-model="keyword" placeholder="标题搜索" class="kw" clearable @keyup.enter="load" />
      <el-button type="primary" @click="load">搜索</el-button>
    </div>
    <el-table :data="rows" v-loading="busy" stripe>
      <el-table-column prop="src_anime_id" label="ID" width="90" />
      <el-table-column prop="title" label="标题" min-width="240" show-overflow-tooltip />
      <el-table-column prop="type" label="类型" width="80" />
      <el-table-column prop="year" label="年份" width="80" />
      <el-table-column prop="score" label="评分" width="80" />
      <el-table-column label="状态" width="100">
        <template #default="{ row }">
          <el-tag size="small" :type="onlineTag(row)">{{ onlineLabel(row) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="130">
        <template #default="{ row }">
          <el-popconfirm v-if="isOnline(row)" title="下架后用户侧将不可见，确定？"
            @confirm="offline(row)">
            <template #reference><el-button size="small" type="danger" text>下架</el-button></template>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>
    <el-pagination class="pager" layout="prev, pager, next, total" :total="total"
      :page-size="size" v-model:current-page="page" @current-change="load" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../../api/client'

const rows = ref([]); const total = ref(0); const page = ref(1); const size = 20
const keyword = ref(''); const busy = ref(false)

const isOnline = (row) => Number(row.is_online ?? 1) === 1
const onlineLabel = (row) => (Number(row.is_forbidden ?? 0) === 1 ? '已封禁' : isOnline(row) ? '在线' : '已下架')
const onlineTag = (row) => (onlineLabel(row) === '在线' ? 'success' : 'info')

async function load () {
  busy.value = true
  try {
    const d = await api.get('/api/v1/admin/animes', { keyword: keyword.value, page: page.value, size })
    rows.value = d.list || []; total.value = d.total || 0
  } finally { busy.value = false }
}

async function offline (row) {
  await api.put(`/api/v1/admin/animes/${row.src_anime_id}/offline`)
  ElMessage.success(`已下架：${row.title}（用户侧详情/相似/加追番同步不可见）`)
  load()
}

onMounted(load)
</script>

<style scoped>
.toolbar { display: flex; gap: 12px; margin-bottom: 12px; }
.kw { width: 260px; }
.pager { margin-top: 14px; justify-content: flex-end; }
</style>
