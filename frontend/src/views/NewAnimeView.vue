<template>
  <div>
    <div class="toolbar">
      <el-input v-model="season" placeholder="季度（如 2026Q3，留空=全部新番）" class="season" clearable @keyup.enter="load" />
      <el-button type="primary" @click="load">查询</el-button>
    </div>
    <el-empty v-if="!busy && !items.length" description="该季度暂无新番推荐" />
    <template v-for="it in items" :key="`${it.anime_id}-${it.rank_no}`">
      <AnimeCard :it="it" @track="onTrack" @detail="showDetail" />
    </template>
    <AnimeDetailDialog v-model="detailOpen" :anime-id="detailId" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api/client'
import AnimeCard from '../components/AnimeCard.vue'
import AnimeDetailDialog from '../components/AnimeDetailDialog.vue'

const items = ref([]); const busy = ref(false); const season = ref('')
const detailOpen = ref(false); const detailId = ref(null)
const showDetail = (id) => { detailId.value = id; detailOpen.value = true }

async function load () {
  busy.value = true
  try {
    const d = await api.get('/api/v1/recommend/new-anime',
      { size: 20, with_explain: true, season: season.value.trim() })
    items.value = d.items || []
  } finally { busy.value = false }
}
async function onTrack (it) {
  await api.post('/api/v1/records', { anime_id: it.anime_id, status: 1 })
  ElMessage.success(`已加入追番：${it.title}`)
}
onMounted(load)
</script>

<style scoped>
.toolbar { display: flex; gap: 12px; margin-bottom: 14px; }
.season { width: 280px; }
</style>
