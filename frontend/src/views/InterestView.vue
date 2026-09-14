<template>
  <div v-loading="busy">
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
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as echarts from 'echarts'
import { api } from '../api/client'

const busy = ref(false)
const radarEl = ref(null); const trendEl = ref(null)
const capsules = ref([]); const points = ref([])
const granularity = ref('quarter')
let radarChart = null; let trendChart = null

function drawRadar (radar) {
  radarChart = echarts.init(radarEl.value)
  radarChart.setOption({
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

function drawTrend (trend) {
  trendChart = echarts.init(trendEl.value)
  const periods = trend.map((t) => t.period)
  const genreSet = new Set()
  trend.forEach((t) => Object.keys(t.genres || {}).forEach((g) => genreSet.add(g)))
  const series = [...genreSet].slice(0, 8).map((g) => ({
    name: g, type: 'line', smooth: true, symbolSize: 4,
    data: trend.map((t) => (t.genres || {})[g] || 0),
  }))
  trendChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0 },
    grid: { left: 40, right: 20, top: 20, bottom: 44 },
    xAxis: { type: 'category', data: periods },
    yAxis: { type: 'value', name: '消费次数' },
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
    const d = await api.get('/api/v1/analysis/interest-radar')
    capsules.value = d.capsules || []
    drawRadar(d.radar || [])
    await loadTrend()
  } finally { busy.value = false }
})

onBeforeUnmount(() => { radarChart?.dispose(); trendChart?.dispose() })
</script>

<style scoped>
.chart { height: 340px; }
.chart.wide { height: 320px; }
.mt16 { margin-top: 16px; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.capsules { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.capsule .cap-name { font-weight: 600; margin-bottom: 6px; color: var(--brand); }
.cap-genres { display: flex; gap: 4px; flex-wrap: wrap; }
</style>
