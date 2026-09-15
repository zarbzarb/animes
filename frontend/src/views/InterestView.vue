<template>
  <div v-loading="busy">
    <!-- 概览：画像标签 + 关键数字 -->
    <el-card shadow="never" class="mb16">
      <template #header>我的画像概览
        <span class="dim">(分析页数据与推荐引擎同源)</span>
      </template>
      <el-empty v-if="!hasData && !busy">
        <template #description>
          <p class="empty-tip">你的兴趣画像还是一张白纸 —— 追几部番、打几个分，<br>
            这里就会长出你的专属分析：兴趣雷达、题材漂移、观看节奏……</p>
        </template>
        <div class="cta-row">
          <el-button type="primary" @click="$router.push('/library')">去番剧库挑几部</el-button>
          <el-button @click="$router.push('/')">看看首页推荐</el-button>
        </div>
      </el-empty>
      <template v-else>
        <div class="metric-row">
          <div class="metric"><b>{{ profile.total_records || nRecords }}</b><span>追番总数</span></div>
          <div class="metric"><b>{{ profileReady ? activityCn : '…' }}</b><span>活跃度</span></div>
          <div class="metric"><b>{{ profileReady ? fmt(profile.watch_intensity) : '…' }}</b><span>观看强度</span></div>
          <div class="metric"><b>{{ profileReady ? fmt(profile.avg_rating_tendency) : '…' }}</b><span>评分倾向</span></div>
          <div class="metric"><b>{{ profileReady ? pct(profile.dropped_rate) : '…' }}</b><span>弃番率</span></div>
        </div>
        <el-alert v-if="!profileReady" type="info" :closable="false" class="profile-pending"
          title="画像引擎正在计算你的兴趣画像（约需半分钟），稍后刷新可见完整画像"
          description="下方分布图基于你的实际追番记录，已实时更新。" show-icon />
        <div class="tag-row" v-if="topGenres.length">
          <span class="lbl">最爱题材</span>
          <el-tag v-for="g in topGenres" :key="g" effect="plain" size="large" class="gt">{{ g }}</el-tag>
        </div>
        <div class="tag-row" v-if="profile.user_tag || profile.summary_text">
          <span class="lbl">画像标签</span>
          <el-tag v-if="profile.user_tag" type="warning" effect="light">{{ profile.user_tag }}</el-tag>
          <span class="summary">{{ profile.summary_text }}</span>
        </div>
      </template>
    </el-card>

    <!-- 有数据才渲染分析图：新用户看到的是引导，不是一排空白坐标系 -->
    <template v-if="hasData">
    <el-row :gutter="16">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>兴趣雷达（12 类题材强度）</template>
          <div ref="radarEl" class="chart" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>兴趣胶囊（多兴趣模型在线产出）</template>
          <el-empty v-if="!capsules.length" description="暂无胶囊数据（需要离线任务写入 user_interest_capsule）" />
          <div v-else class="capsules">
            <el-card v-for="(c, i) in capsules" :key="i" shadow="hover" class="capsule">
              <div class="cap-name">兴趣 {{ i + 1 }}</div>
              <div class="cap-genres">
                <el-tag v-for="t in c.top_genres || c.genres || []" :key="t" size="small" effect="plain">{{ t }}</el-tag>
              </div>
            </el-card>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" class="mt16">
      <el-col :span="8">
        <el-card shadow="never"><template #header>追番状态分布</template>
          <div ref="pieEl" class="chart" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="never"><template #header>我的评分分布（1-10）</template>
          <div ref="ratingEl" class="chart" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="never"><template #header>月度观看节奏（近 12 月）</template>
          <div ref="monthEl" class="chart" />
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="mt16">
      <template #header>
        <div class="flex-between">
          <span>题材漂移趋势</span>
          <el-radio-group v-model="granularity" size="small" @change="loadTrend">
            <el-radio-button value="month">月</el-radio-button>
            <el-radio-button value="quarter">季</el-radio-button>
            <el-radio-button value="year">年</el-radio-button>
          </el-radio-group>
        </div>
      </template>
      <div ref="trendEl" class="chart wide" />
    </el-card>

    <el-card shadow="never" class="mt16" v-if="points.length">
      <template #header>检测到的兴趣漂移点（JS 散度突变）</template>
      <el-table :data="points" stripe size="small">
        <el-table-column prop="period" label="周期" width="110" />
        <el-table-column prop="js_divergence" label="JS 散度" width="110" />
        <el-table-column label="漂移方向" min-width="200">
          <template #default="{ row }">{{ row.from || '—' }} → {{ row.to || '—' }}</template>
        </el-table-column>
      </el-table>
    </el-card>
    </template>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, computed, nextTick } from 'vue'
import * as echarts from 'echarts'
import { api } from '../api/client'

const busy = ref(false)
const radarEl = ref(null); const trendEl = ref(null)
const pieEl = ref(null); const ratingEl = ref(null); const monthEl = ref(null)
const capsules = ref([]); const points = ref([])
const profile = ref({})
const nRecords = ref(0)
const statusDist = ref({}); const ratingDist = ref([]); const monthly = ref([])
const granularity = ref('quarter')
let charts = []

/* 有没有数据看**实际追番记录**（后端直接数 watch_record），
   不看 A1 画像 —— 画像是异步重排落库的，新用户加完番有几十秒空窗 */
const hasData = computed(() => nRecords.value > 0 || (profile.value.total_records || 0) > 0)
const profileReady = computed(() => (profile.value.total_records || 0) > 0)

const topGenres = computed(() =>
  (profile.value.top_genres || []).map((g) => g.genre || g.name || g).filter(Boolean).slice(0, 8))
const activityCn = computed(() =>
  ({ low: '低', medium: '中', high: '高' })[profile.value.activity_label] || '—')

const fmt = (v) => (v == null ? '—' : Number(v).toFixed(2))
const pct = (v) => (v == null ? '—' : `${(Number(v) * 100).toFixed(0)}%`)

function init (el, option) {
  if (!el.value) return
  const c = echarts.init(el.value)
  c.setOption(option)
  charts.push(c)
}

function drawRadar (radar) {
  init(radarEl, {
    tooltip: {},
    radar: {
      indicator: radar.map((r) => ({ name: r.genre, max: 1 })),
      radius: '62%',
    },
    series: [{
      type: 'radar',
      areaStyle: { opacity: 0.25 },
      data: [{ name: '兴趣强度', value: radar.map((r) => r.value),
               count: radar.map((r) => r.count) }],
      itemStyle: { color: '#e8749c' },
    }],
  })
}

function drawPie () {
  const palette = ['#909399', '#fb7299', '#67c23a', '#f56c6c']
  const data = Object.entries(statusDist.value).map(([k, v], i) =>
    ({ name: k, value: v, itemStyle: { color: palette[i % 4] } })).filter((d) => d.value)
  init(pieEl, {
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 0, itemWidth: 12, itemHeight: 12 },
    series: [{ type: 'pie', radius: ['38%', '62%'], center: ['50%', '44%'],
               label: { show: false }, data, top: -10 }],
  })
}

function drawRating () {
  init(ratingEl, {
    tooltip: { trigger: 'axis' },
    grid: { left: 36, right: 12, top: 24, bottom: 26 },
    xAxis: { type: 'category', data: ratingDist.value.map((d) => d.rating) },
    yAxis: { type: 'value', minInterval: 1 },
    series: [{ type: 'bar', barWidth: '55%',
               data: ratingDist.value.map((d) => d.count),
               itemStyle: { color: '#fb7299', borderRadius: [3, 3, 0, 0] } }],
  })
}

function drawMonthly () {
  init(monthEl, {
    tooltip: { trigger: 'axis' },
    grid: { left: 36, right: 12, top: 24, bottom: 26 },
    xAxis: { type: 'category', data: monthly.value.map((d) => d.period) },
    yAxis: { type: 'value', minInterval: 1 },
    series: [{ type: 'bar', barWidth: '55%',
               data: monthly.value.map((d) => d.count),
               itemStyle: { color: '#00a1d6', borderRadius: [3, 3, 0, 0] } }],
  })
}

function drawTrend (trend) {
  const periods = trend.map((t) => t.period)
  const genreSet = new Set()
  trend.forEach((t) => Object.keys(t.genres || {}).forEach((g) => genreSet.add(g)))
  const series = [...genreSet].slice(0, 8).map((g) => ({
    name: g, type: 'line', smooth: true, symbolSize: 4,
    data: trend.map((t) => (t.genres || {})[g] || 0),
  }))
  init(trendEl, {
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0, textStyle: { fontSize: 12, color: '#606266' } },
    grid: { left: 56, right: 24, top: 36, bottom: 48 },
    xAxis: {
      type: 'category', data: periods,
      axisLabel: { fontSize: 12, color: '#606266', rotate: 30 },
    },
    yAxis: {
      type: 'value', name: '消费次数',
      nameTextStyle: { fontSize: 13, color: '#303133', align: 'left', padding: [0, 0, 6, -30] },
      axisLabel: { fontSize: 12, color: '#606266' },
      splitLine: { lineStyle: { color: '#ebeef5' } },
    },
    series,
  })
}

async function loadTrend () {
  const d = await api.get('/api/v1/analysis/drift-trend', { granularity: granularity.value })
  drawTrend(d.trend || [])
  const p = await api.get('/api/v1/analysis/drift-points', { granularity: granularity.value })
  points.value = p.points || []
}

onMounted(async () => {
  busy.value = true
  try {
    // 画像汇总失败不阻塞主图（新用户画像可能未建）
    try {
      const s = await api.get('/api/v1/analysis/profile-summary')
      profile.value = s.profile || {}
      nRecords.value = s.n_records || 0
      statusDist.value = s.status_dist || {}
      ratingDist.value = s.rating_dist || []
      monthly.value = s.monthly_counts || []
    } catch { /* 概览区留空 */ }
    if (!hasData.value) return   // 纯新用户：空态引导，不加载雷达/漂移
    // hasData 由 false→true 会触发 v-if 重新挂载图表容器，DOM 更新是异步的：
    // 不等 nextTick 就 init，三个 ref 还是 null，饼图/柱图会被静默跳过
    await nextTick()
    drawPie(); drawRating(); drawMonthly()
    const d = await api.get('/api/v1/analysis/interest-radar')
    capsules.value = d.capsules || []
    drawRadar(d.radar || [])
    await loadTrend()
  } finally { busy.value = false }
})

onBeforeUnmount(() => { charts.forEach((c) => c.dispose()); charts = [] })
</script>

<style scoped>
.chart { height: 300px; }
.chart.wide { height: 320px; }
.mt16 { margin-top: 16px; }
.mb16 { margin-bottom: 16px; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.capsules { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.capsule .cap-name { font-weight: 600; margin-bottom: 6px; color: var(--brand); }
.cap-genres { display: flex; gap: 4px; flex-wrap: wrap; }
.metric-row { display: flex; gap: 40px; flex-wrap: wrap; margin-bottom: 14px; }
.metric { display: flex; flex-direction: column; align-items: flex-start; }
.metric b { font-size: 24px; color: var(--brand); }
.metric span { font-size: 12px; color: var(--text-3); }
.tag-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.tag-row .lbl { font-size: 12px; color: var(--text-3); flex-shrink: 0; }
.gt { margin-right: 4px; }
.summary { font-size: 13px; color: var(--text-2); }
.dim { font-size: 12px; color: var(--text-3); font-weight: 400; margin-left: 6px; }
.empty-tip { color: var(--text-2, #606266); line-height: 1.8; }
.cta-row { display: flex; gap: 12px; justify-content: center; }
.profile-pending { margin-bottom: 12px; }
</style>
