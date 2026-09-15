<template>
  <div>
    <el-alert v-if="coldStart" type="info" :closable="false" class="cold-tip"
      title="欢迎来到 AniRec！你还没有追番记录"
      description="当前展示的是全站热门。去「我的追番」添加几部看过的番，或直接评分，推荐会立刻变得个性化。" show-icon />

    <el-alert v-if="degraded.length" type="warning" :closable="false" class="degraded"
      :title="`部分能力已降级：${degraded.join('、')}`" show-icon />

    <div class="toolbar">
      <el-radio-group v-model="mode" @change="load">
        <el-radio-button value="feed">综合推荐</el-radio-button>
        <el-radio-button value="genre">分类推荐</el-radio-button>
      </el-radio-group>
      <el-select v-if="mode === 'genre'" v-model="genreId" placeholder="选题材" class="genre-sel" @change="load">
        <el-option v-for="g in genres" :key="g.genre_id" :label="g.name" :value="g.genre_id" />
      </el-select>
      <el-button :icon="Refresh" :loading="busy" @click="load(true)">强制刷新</el-button>
      <span v-if="metaInfo" class="meta-info">{{ metaInfo }}</span>
    </div>

    <el-empty v-if="!busy && !items.length" description="暂无推荐，先去加几条追番记录吧" />
    <template v-for="it in items" :key="`${it.anime_id}-${it.rank_no}`">
      <AnimeCard :it="it" @track="onTrack" @detail="showDetail" @similar="goSimilar" />
    </template>
    <AnimeDetailDialog v-model="detailOpen" :anime-id="detailId" />
  </div>
</template>

<script setup>
import { onMounted, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { api } from '../api/client'
import AnimeCard from '../components/AnimeCard.vue'
import AnimeDetailDialog from '../components/AnimeDetailDialog.vue'

const router = useRouter()
const busy = ref(false)
const mode = ref('feed')
const genreId = ref(null)
const genres = ref([])
const items = ref([])
const meta = ref({})
const coldStart = ref(false)
const degraded = computed(() => meta.value?.degraded || [])

const metaInfo = computed(() => {
  const m = meta.value || {}
  const bits = []
  if (m.cache_hit != null) bits.push(m.cache_hit ? '缓存命中 ⚡' : '实时计算')
  if (m.elapsed_ms != null) bits.push(`${m.elapsed_ms}ms`)
  if (m.agent_chain?.length) bits.push(`链路 ${m.agent_chain.join('→')}`)
  return bits.join(' · ')
})

async function load (force = false) {
  busy.value = true
  try {
    const params = { size: 20, with_explain: true, refresh: force === true }
    const d = mode.value === 'feed'
      ? await api.get('/api/v1/recommend/feed', params)
      : await api.get('/api/v1/recommend/by-genre', { ...params, genre_id: genreId.value })
    items.value = d.items || []
    meta.value = d.meta || {}
    coldStart.value = Boolean(d.is_cold_start_user || d.meta?.used_fallback)
  } finally { busy.value = false }
}

async function onTrack (it) {
  await api.post('/api/v1/records', { anime_id: it.anime_id, status: 1 })
  ElMessage.success(`已加入追番：${it.title}`)
}

const detailOpen = ref(false)
const detailId = ref(null)
const showDetail = (id) => { detailId.value = id; detailOpen.value = true }
const goSimilar = (id) => router.push({ name: 'new', query: { similar_to: id } })

onMounted(async () => {
  genres.value = (await api.get('/api/v1/genres'))?.list || []
  load()
})
</script>

<style scoped>
.toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.genre-sel { width: 160px; }
.meta-info { font-size: 12px; color: #909399; }
.degraded { margin-bottom: 12px; }
.cold-tip { margin-bottom: 12px; }
</style>
