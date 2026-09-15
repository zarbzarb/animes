<template>
  <div>
    <div class="toolbar">
      <el-input v-model="keyword" placeholder="搜索番剧：支持中文译名 / 英文原名（回车搜索）"
        class="kw" size="large" clearable @keyup.enter="search(1)" @clear="search(1)">
        <template #prefix><el-icon><Search /></el-icon></template>
      </el-input>
      <el-select v-model="genreId" placeholder="全部题材" clearable class="sel" @change="search(1)">
        <el-option v-for="g in genres" :key="g.genre_id" :label="g.name_cn || g.name_en" :value="g.genre_id" />
      </el-select>
      <el-select v-model="orderBy" class="sel" @change="search(1)">
        <el-option label="按热度" value="n_interactions" />
        <el-option label="按评分" value="score" />
        <el-option label="按年份" value="year" />
      </el-select>
      <el-checkbox v-model="coldOnly" label="只看新番" @change="search(1)" />
    </div>

    <div v-loading="busy" class="grid-wrap">
      <el-empty v-if="!busy && !rows.length" description="没有找到匹配的番剧" />
      <div v-else class="grid">
        <div v-for="a in rows" :key="a.id" class="anime" @click="openDetail(a)">
          <CoverImage :src="a.image_url" :title="a.title_cn || a.title" w="100%" h="150px" :show-title="true" class="cover" />
          <div class="name">{{ a.title_cn || a.title }}</div>
          <div v-if="a.title_cn" class="en">{{ a.title }}</div>
          <div class="sub">
            <span v-if="a.score != null" class="star">★ {{ a.score }}</span>
            <span>{{ a.year || '' }}</span>
            <el-tag v-if="a.is_cold_start" size="small" type="warning" effect="plain">新番</el-tag>
          </div>
        </div>
      </div>
    </div>

    <el-pagination class="pager" layout="prev, pager, next, total" :total="total"
      :page-size="size" v-model:current-page="page" @current-change="search" />

    <AnimeDetailDialog v-model="detailOpen" :anime-id="detailId" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { Search } from '@element-plus/icons-vue'
import { api } from '../api/client'
import CoverImage from '../components/CoverImage.vue'
import AnimeDetailDialog from '../components/AnimeDetailDialog.vue'

const keyword = ref(''); const genreId = ref(null); const orderBy = ref('n_interactions')
const coldOnly = ref(false)
const rows = ref([]); const total = ref(0); const page = ref(1); const size = 30
const genres = ref([]); const busy = ref(false)
const detailOpen = ref(false); const detailId = ref(null)
const openDetail = (a) => { detailId.value = a.src_anime_id ?? a.id; detailOpen.value = true }

async function search (p = page.value) {
  if (typeof p === 'number') page.value = p
  busy.value = true
  try {
    const params = { page: page.value, size, order_by: orderBy.value }
    if (keyword.value.trim()) params.keyword = keyword.value.trim()
    if (genreId.value) params.genre_id = genreId.value
    if (coldOnly.value) params.cold_only = true
    const d = await api.get('/api/v1/animes', params)
    rows.value = d.list || []
    total.value = d.pagination?.total || 0
  } finally { busy.value = false }
}

onMounted(async () => {
  genres.value = (await api.get('/api/v1/genres'))?.list || []
  search(1)
})
</script>

<style scoped>
.toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.kw { width: 380px; }
.sel { width: 140px; }
.grid-wrap { min-height: 240px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 14px; }
.anime { cursor: pointer; border-radius: 8px; transition: transform .15s; }
.anime:hover { transform: translateY(-2px); }
.cover { border-radius: 8px; }
.name { font-size: 13px; font-weight: 600; color: var(--text-1); margin-top: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.en { font-size: 11px; color: var(--text-3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sub { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-3); margin-top: 2px; }
.star { color: var(--brand); font-weight: 600; }
.pager { margin-top: 16px; justify-content: center; }
</style>
