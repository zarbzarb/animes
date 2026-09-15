<template>
  <el-dialog v-model="visible" width="680px" class="detail-dialog" :show-close="true"
    @closed="detail = null; failed = false">
    <template #header>
      <span class="dlg-title">番剧详情</span>
    </template>

    <div v-loading="loading" class="body">
      <template v-if="detail">
        <div class="head">
          <CoverImage :src="detail.image_url" :title="detail.title" w="150px" h="210px" class="cover"
            @click="failed = true" />
          <div class="info">
            <h3 class="title">{{ titleCn || detail.title }}</h3>
            <div v-if="titleCn" class="alt">{{ detail.title }}</div>
            <div v-else-if="detail.alt_title && detail.alt_title !== detail.title" class="alt">{{ detail.alt_title }}</div>

            <div class="meta-grid">
              <div v-if="detail.score != null" class="score">MAL ★ {{ Number(detail.score).toFixed(1) }}</div>
              <div v-if="detail.community_rating" class="score site-score">
                本站 {{ detail.community_rating.avg }}<span class="cnt">（{{ detail.community_rating.count }} 人评价）</span>
              </div>
              <span class="m">{{ detail.type || 'TV' }}</span>
              <span v-if="detail.year" class="m">{{ detail.year }}</span>
              <span v-if="detail.episodes" class="m">全 {{ detail.episodes }} 话</span>
              <span v-if="detail.n_interactions != null" class="m">{{ detail.n_interactions }} 人互动</span>
              <el-tag v-if="detail.is_cold_start" size="small" type="warning" effect="plain">冷启动</el-tag>
            </div>

            <div v-if="genreNames.length" class="genres">
              <el-tag v-for="g in genreNames" :key="g" size="small" effect="plain">{{ g }}</el-tag>
            </div>

            <div v-if="myLabel" class="mine">
              <el-tag size="small" effect="light">{{ myLabel }}</el-tag>
            </div>

            <div class="ops">
              <el-button type="primary" :disabled="!!myLabel" @click="track">加入追番</el-button>
              <el-button v-if="detail.mal_url" text tag="a" :href="detail.mal_url" target="_blank">MAL 页面 ↗</el-button>
            </div>
          </div>
        </div>

        <div class="summary-block">
          <div class="sum-head">简介<span v-if="!hasCnSummary" class="sum-lang">（暂无中文，显示英文原文）</span></div>
          <p v-if="summaryText" class="sum-text" :class="{ clamp: !expanded }">{{ summaryText }}</p>
          <p v-else class="sum-text empty">暂无简介</p>
          <el-button v-if="summaryText && summaryText.length > 120" text type="primary" size="small"
            class="expand" @click="expanded = !expanded">{{ expanded ? '收起' : '展开全部' }}</el-button>
        </div>
      </template>
    </div>
  </el-dialog>
</template>

<script setup>
/**
 * 番剧详情弹窗 —— 看推荐/追番前先读简介。
 *
 * 数据来自 GET /api/v1/animes/{id}（summary 字段），打开时按需拉取。
 * 已登录时会附带 my_record，据此展示追番状态并禁用"加入追番"。
 */
import { computed, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api/client'
import CoverImage from './CoverImage.vue'

const props = defineProps({ animeId: { type: Number, default: null } })
const visible = defineModel({ type: Boolean, default: false })

const loading = ref(false)
const detail = ref(null)
const expanded = ref(false)

watch([visible, () => props.animeId], ([v, id]) => {
  if (!v || !id) return
  detail.value = null
  expanded.value = false
  loading.value = true
  api.get(`/api/v1/animes/${id}`)
    .then((d) => { detail.value = d })
    .catch(() => { visible.value = false })
    .finally(() => { loading.value = false })
})

const genreNames = computed(() =>
  (detail.value?.genres || []).map((g) => g.genre).filter(Boolean))
/* 简介中文优先（translate_cn.py 只翻头部热门），长尾回退英文 */
const summaryText = computed(() =>
  (detail.value?.summary_cn || detail.value?.summary || '').trim())
const hasCnSummary = computed(() => Boolean((detail.value?.summary_cn || '').trim()))
const titleCn = computed(() => detail.value?.title_cn || '')

const statusNames = ['想看', '在看', '看过', '弃番']
const myLabel = computed(() => {
  const r = detail.value?.my_record
  return r ? `我的追番：${statusNames[r.status] ?? '已收藏'}` : ''
})

async function track () {
  if (!detail.value) return
  await api.post('/api/v1/records', { anime_id: detail.value.src_anime_id, status: 1 })
  ElMessage.success(`已加入追番：${detail.value.title}`)
  // 刷新个性化状态
  api.get(`/api/v1/animes/${props.animeId}`).then((d) => { detail.value = d }).catch(() => {})
}
</script>

<style scoped>
.dlg-title { font-size: 16px; font-weight: 600; color: var(--text-1); }
.body { min-height: 220px; }
.head { display: flex; gap: 18px; }
.cover { cursor: pointer; }
.info { flex: 1; min-width: 0; }
.title { margin: 0 0 4px; font-size: 18px; line-height: 1.4; color: var(--text-1); }
.alt { font-size: 12px; color: var(--text-3); margin-bottom: 6px; }
.meta-grid { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin: 10px 0; color: var(--text-2); font-size: 13px; }
.score { color: var(--brand); font-weight: 700; font-size: 15px; }
.site-score { color: #2ac864; }
.site-score .cnt { font-size: 12px; font-weight: 400; color: var(--text-3); }
.genres { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.mine { margin: 8px 0; }
.ops { margin-top: 12px; display: flex; align-items: center; gap: 4px; }

.summary-block { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }
.sum-head { font-size: 14px; font-weight: 600; color: var(--text-1); margin-bottom: 8px; }
.sum-lang { font-size: 12px; font-weight: 400; color: var(--text-3); margin-left: 6px; }
.sum-text { margin: 0; font-size: 13px; line-height: 1.8; color: var(--text-2); white-space: pre-line; }
.sum-text.empty { color: var(--text-3); }
.sum-text.clamp {
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.expand { margin-top: 4px; padding-left: 0; }
</style>
