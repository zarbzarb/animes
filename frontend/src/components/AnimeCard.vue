<template>
  <el-card shadow="never" class="card">
    <div class="body">
      <CoverImage :src="cover" :title="title" w="76px" h="104px" class="cover" />
      <div class="main">
    <div class="head">
      <div class="title-row">
        <span class="rank">#{{ rank }}</span>
        <span class="title">{{ title || '(无标题)' }}</span>
        <el-tag size="small" effect="plain">{{ type || 'TV' }}</el-tag>
        <span class="meta">{{ year || '' }} <b v-if="score != null">★ {{ score }}</b></span>
        <el-tag v-if="it.is_cold_start" size="small" type="warning" effect="plain">冷启动</el-tag>
      </div>
      <div class="genres">
        <el-tag v-for="g in genreList" :key="g" size="small" type="info" effect="plain">{{ g }}</el-tag>
      </div>
    </div>

    <div class="bars">
      <div class="score-bar final"><span class="lb">final</span><div class="track"><div class="fill" :style="{ width: pct(it.final_score) }" /></div><span class="v">{{ fmt(it.final_score) }}</span></div>
      <div class="score-bar behavior"><span class="lb">行为</span><div class="track"><div class="fill" :style="{ width: pct(it.behavior_score) }" /></div><span class="v">{{ fmt(it.behavior_score) }}</span></div>
      <div class="score-bar content"><span class="lb">内容</span><div class="track"><div class="fill" :style="{ width: pct(it.content_score) }" /></div><span class="v">{{ fmt(it.content_score) }}</span></div>
    </div>

    <div v-if="reason" class="why">💡 {{ reason }}<span v-if="it.interest_label" class="dim">（兴趣 {{ it.interest_label }}）</span></div>

    <div class="ops">
      <el-button size="small" type="primary" text @click="$emit('detail', animeId)">简介</el-button>
      <el-button size="small" @click="$emit('similar', animeId)">相似推荐</el-button>
      <el-button size="small" type="primary" plain @click="$emit('track', { anime_id: animeId, title })">加入追番</el-button>
    </div>
      </div>
    </div>
  </el-card>
</template>

<script setup>
import { computed } from 'vue'
import CoverImage from './CoverImage.vue'

/**
 * 兼容两种 item 形状：
 * - 线上 feed：{rank, anime:{src_anime_id,title,type,year,score,genres}, final_score, ..., explain:{reason}}
 * - 宽松备选：{rank_no, anime_id, title, ..., reason}
 */
const props = defineProps({ it: { type: Object, required: true } })
defineEmits(['track', 'similar', 'detail'])

const a = computed(() => props.it.anime || {})
const rank = computed(() => props.it.rank ?? props.it.rank_no ?? '—')
const title = computed(() => a.value.title || props.it.title || '')
const type = computed(() => a.value.type || props.it.type || '')
const year = computed(() => a.value.year || props.it.year || '')
const score = computed(() => a.value.score ?? props.it.score ?? null)
const animeId = computed(() => a.value.src_anime_id ?? a.value.id ?? props.it.anime_id ?? null)
const cover = computed(() => a.value.image_url || props.it.image_url || '')
const reason = computed(() => props.it.explain?.reason || props.it.reason || '')

const genreList = computed(() => {
  const raw = a.value.genres ?? props.it.genre_names ?? props.it.genres ?? []
  if (!Array.isArray(raw)) return String(raw).split(/[,、]/).filter(Boolean)
  return raw.map((g) => (typeof g === 'object' ? (g.name || g.genre || g.title) : g)).filter(Boolean)
})
const fmt = (v) => (typeof v === 'number' ? v.toFixed(3) : (v ?? '—'))
const pct = (v) => `${Math.max(0, Math.min(1, Number(v) || 0)) * 100}%`
</script>

<style scoped>
.card { margin-bottom: 12px; }
.body { display: flex; gap: 12px; }
.main { flex: 1; min-width: 0; }
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
.title-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.rank { color: var(--brand); font-weight: 700; }
.title { font-size: 15px; font-weight: 600; color: #303133; }
.meta { color: #909399; font-size: 13px; }
.genres { display: flex; gap: 4px; flex-wrap: wrap; }
.bars { margin-top: 10px; display: grid; gap: 4px; max-width: 560px; }
.score-bar .lb { width: 38px; text-align: right; }
.score-bar .v { width: 48px; font-variant-numeric: tabular-nums; }
.why { margin-top: 10px; font-size: 13px; color: #606266; background: #faf3f6; border-left: 3px solid var(--brand); padding: 6px 10px; border-radius: 0 6px 6px 0; }
.dim { color: #b0909e; font-size: 12px; margin-left: 4px; }
.ops { margin-top: 10px; display: flex; gap: 4px; }
</style>
