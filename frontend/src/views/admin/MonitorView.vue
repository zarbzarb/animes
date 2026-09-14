<template>
  <div v-loading="busy">
    <el-row :gutter="16">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>Agent 健康状态（调用量 / p95 耗时 / 熔断）</template>
          <el-table :data="agents" stripe size="small">
            <el-table-column prop="agent" label="Agent" width="110" />
            <el-table-column prop="state" label="状态" width="90">
              <template #default="{ row }">
                <el-tag size="small" :type="row.state === 'open' ? 'danger' : row.state === 'half_open' ? 'warning' : 'success'">
                  {{ row.state }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="n_calls" label="调用次数" width="100" />
            <el-table-column label="p95 (ms)" width="100">
              <template #default="{ row }">{{ row.p95_ms ?? row.p95 ?? '—' }}</template>
            </el-table-column>
            <el-table-column prop="n_errors" label="错误" width="80" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="10">
        <el-card shadow="never">
          <template #header>系统总览</template>
          <el-descriptions v-if="overview" :column="1" border size="small">
            <el-descriptions-item v-for="(v, k) in overview" :key="k" :label="k">{{ v }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="mt16">
      <template #header>
        <div class="flex-between">
          <span>推荐效果监控（离线快照 + 在线反馈）</span>
          <el-select v-model="metricType" size="small" class="sel" @change="loadMetrics">
            <el-option label="HR@10" value="hr10" />
            <el-option label="NDCG@10" value="ndcg10" />
            <el-option label="MRR" value="mrr" />
          </el-select>
        </div>
      </template>
      <div ref="metricEl" class="chart" />
    </el-card>

    <el-card shadow="never" class="mt16">
      <template #header>冷启动观察（候选池覆盖率 / 新番曝光）</template>
      <el-descriptions :column="2" border size="small">
        <el-descriptions-item v-for="(v, k) in coldStart" :key="k" :label="k">{{ v }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as echarts from 'echarts'
import { api } from '../../api/client'

const busy = ref(false)
const agents = ref([]); const overview = ref(null); const coldStart = ref({})
const metricType = ref('hr10')
const metricEl = ref(null)
let chart = null

function drawMetric (d) {
  chart = chart || echarts.init(metricEl.value)
  // 兼容两种形状：{series: [{name, points: [{date, value}]}]} 或 {list: [...]}
  const seriesSrc = d.series || []
  const dates = [...new Set(seriesSrc.flatMap((s) => (s.points || []).map((p) => p.date)))]
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    grid: { left: 50, right: 20, top: 20, bottom: 46 },
    xAxis: { type: 'category', data: dates },
    yAxis: { type: 'value' },
    series: seriesSrc.map((s) => ({
      name: s.name, type: 'line', smooth: true,
      data: dates.map((dt) => (s.points || []).find((p) => p.date === dt)?.value ?? null),
    })),
  }, true)
}

async function loadMetrics () {
  const d = await api.get('/api/v1/admin/metrics', { metric_type: metricType.value, days: 30 })
  drawMetric(d)
}

onMounted(async () => {
  busy.value = true
  try {
    const h = await api.get('/api/v1/admin/agents/health')
    agents.value = h.agents || h.list || []
    overview.value = h.overview || { agent_count: agents.value.length }
    coldStart.value = await api.get('/api/v1/admin/cold-start')
    await loadMetrics()
  } finally { busy.value = false }
})

onBeforeUnmount(() => chart?.dispose())
</script>

<style scoped>
.mt16 { margin-top: 16px; }
.chart { height: 300px; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.sel { width: 130px; }
</style>
