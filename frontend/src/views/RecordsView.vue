<template>
  <div>
    <el-row :gutter="12" class="stat-row">
      <el-col :span="6"><el-statistic title="追番总数" :value="stats.total || 0" /></el-col>
      <el-col :span="6"><el-statistic title="已评分" :value="stats.n_rated || 0" /></el-col>
      <el-col :span="6"><el-statistic title="平均评分" :value="stats.avg_rating ?? 0" :precision="2" /></el-col>
      <el-col :span="6"><el-statistic title="在看" :value="(stats.by_status || {}).在看 || 0" /></el-col>
    </el-row>

    <div class="toolbar">
      <el-select v-model="status" clearable placeholder="全部状态" class="sel" @change="load">
        <el-option label="想看" :value="0" /><el-option label="在看" :value="1" />
        <el-option label="看过" :value="2" /><el-option label="弃番" :value="3" />
      </el-select>
      <el-button type="primary" :icon="Plus" @click="adding = true">添加追番</el-button>
    </div>

    <el-table :data="rows" v-loading="busy" stripe>
      <el-table-column label="封面" width="76">
        <template #default="{ row }">
          <CoverImage :src="row.anime?.image_url" :title="row.anime?.title || row.title" w="48px" h="64px" />
        </template>
      </el-table-column>
      <el-table-column label="番剧" min-width="200" show-overflow-tooltip>
        <template #default="{ row }">
          {{ row.anime?.title_cn || row.anime?.title || row.title }}
          <el-button text type="primary" size="small" class="brief-btn"
            @click="showDetail(row)">简介</el-button>
        </template>
      </el-table-column>
      <el-table-column label="状态" width="104">
        <template #default="{ row }">
          <el-select :model-value="row.status" size="small" class="status-sel"
            @change="(s) => setStatus(row, s)">
            <el-option v-for="(label, v) in STATUS" :key="v" :label="label" :value="Number(v)" />
          </el-select>
        </template>
      </el-table-column>
      <el-table-column label="评分 / 评价" min-width="250">
        <template #default="{ row }">
          <div class="rate-row">
            <el-rate :model-value="row.rating" :max="10" size="small"
              @change="(v) => setRating(row, v)" />
            <el-button text type="primary" size="small" class="brief-btn"
              @click="openReview(row)">{{ row.review ? '改评价' : '写评价' }}</el-button>
          </div>
          <div class="review-text" v-if="row.review" @click="openReview(row)">{{ row.review }}</div>
        </template>
      </el-table-column>
      <el-table-column prop="updated_at" label="更新时间" width="170" />
      <el-table-column label="操作" width="90">
        <template #default="{ row }">
          <el-popconfirm title="确定删除这条追番记录？" @confirm="del(row)">
            <template #reference><el-button size="small" type="danger" text>删除</el-button></template>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination class="pager" layout="prev, pager, next, total" :total="total"
      :page-size="size" v-model:current-page="page" @current-change="load" />

    <el-dialog v-model="adding" title="添加追番" width="420px">
      <el-select v-model="picked" filterable remote :remote-method="searchAnime" :loading="searching"
        placeholder="输入番剧标题搜索（支持中文译名）" class="w100">
        <el-option v-for="a in candidates" :key="a.src_anime_id"
          :label="`${a.title_cn || a.title} (${a.year || '-'})`" :value="a.src_anime_id" />
      </el-select>
      <template #footer>
        <el-button @click="adding = false">取消</el-button>
        <el-button type="primary" :disabled="!picked" @click="add">加入追番</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="reviewOpen" title="评分与评价" width="480px">
      <div class="review-form">
        <div class="form-row">
          <span class="lbl">评分</span>
          <el-rate v-model="reviewForm.rating" :max="10" show-score score-template="{value} 分" />
        </div>
        <div class="form-row">
          <span class="lbl">评价</span>
          <el-input v-model="reviewForm.review" type="textarea" :rows="4" maxlength="500"
            show-word-limit placeholder="写点观后感吧（留空保存 = 清除评价）" />
        </div>
      </div>
      <template #footer>
        <el-button @click="reviewOpen = false">取消</el-button>
        <el-button type="primary" @click="saveReview">保存</el-button>
      </template>
    </el-dialog>

    <AnimeDetailDialog v-model="detailOpen" :anime-id="detailId" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { Plus } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { api } from '../api/client'
import CoverImage from '../components/CoverImage.vue'
import AnimeDetailDialog from '../components/AnimeDetailDialog.vue'

const rows = ref([]); const total = ref(0); const page = ref(1); const size = 20
const status = ref(null); const busy = ref(false); const stats = ref({})
const adding = ref(false); const picked = ref(null); const candidates = ref([]); const searching = ref(false)
const detailOpen = ref(false); const detailId = ref(null)

/* 记录行上的 anime_id 即 src_anime_id（与 /records POST 同口径） */
const showDetail = (row) => {
  detailId.value = row.anime?.src_anime_id ?? row.anime?.id ?? row.anime_id
  detailOpen.value = true
}

async function load () {
  busy.value = true
  try {
    const params = { page: page.value, size }
    if (status.value != null && status.value !== '') params.status = status.value
    const d = await api.get('/api/v1/records', params)
    rows.value = d.list || []; total.value = d.total || 0
    stats.value = await api.get('/api/v1/records/stats')
  } finally { busy.value = false }
}

async function searchAnime (q) {
  if (!q) return
  searching.value = true
  try {
    const d = await api.get('/api/v1/animes', { q, size: 20 })
    candidates.value = d.list || []
  } finally { searching.value = false }
}

async function add () {
  await api.post('/api/v1/records', { anime_id: picked.value, status: 1 })
  ElMessage.success('已加入追番')
  adding.value = false; picked.value = null; load()
}

async function setRating (row, v) {
  if (!v) return // el-rate 已禁用 clearable，防御性兜底（后端 rating ge=1，0 会 422）
  await api.put(`/api/v1/records/${row.id}`, { rating: v })
  ElMessage.success('已评分'); load()
}

const STATUS = { 0: '想看', 1: '在看', 2: '已看', 3: '弃番' }
async function setStatus (row, s) {
  await api.put(`/api/v1/records/${row.id}`, { status: s })
  ElMessage.success(`状态已改为「${STATUS[s]}」`); load()
}

async function del (row) {
  await api.del(`/api/v1/records/${row.id}`)
  ElMessage.success('已删除'); load()
}

onMounted(load)
</script>

<style scoped>
.stat-row { margin-bottom: 16px; }
.toolbar { display: flex; gap: 12px; margin-bottom: 12px; }
.sel { width: 140px; }
.pager { margin-top: 14px; justify-content: flex-end; }
.w100 { width: 100%; }
.brief-btn { padding: 0; margin-left: 8px; }
.status-sel { width: 88px; }
.rate-row { display: flex; align-items: center; }
.review-text { margin-top: 2px; font-size: 12px; color: var(--text-3, #909399);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
.review-form .form-row { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
.review-form .lbl { width: 36px; flex-shrink: 0; color: var(--text-2, #606266);
  font-size: 13px; line-height: 32px; }
</style>
