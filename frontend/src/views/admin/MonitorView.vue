<template>
  <div v-loading="busy">
    <el-row :gutter="16">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>Agent 健康状态（调用量 / p95 耗时 / 熔断）</template>
          <el-table :data="agents" stripe size="small">
            <el-table-column label="Agent" min-width="150">
              <template #default="{ row }">
                <b>{{ row.agent_id }}</b>
                <span class="dim"> · {{ row.name }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="status_label" label="状态" width="84">
              <template #default="{ row }">
                <el-tag size="small" :type="row.health_score >= 1 ? 'success' : row.health_score >= 0.5 ? 'warning' : 'danger'">
                  {{ row.status_label }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="calls" label="调用" width="80" />
            <el-table-column label="p95 (ms)" width="90">
              <template #default="{ row }">{{ row.p95_elapsed_ms ?? '—' }}</template>
            </el-table-column>
            <el-table-column prop="fail_cnt" label="失败" width="70" />
            <el-table-column prop="degrade_cnt" label="降级" width="70" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="10">
        <el-card shadow="never">
          <template #header>冷启动观察（候选池覆盖率 / 新番曝光）</template>
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="候选池容量">{{ cs.pool_size ?? '—' }}</el-descriptions-item>
            <el-descriptions-item label="池内 CTR">{{ fmtPct(cs.ctr) }}</el-descriptions-item>
            <el-descriptions-item label="曝光">{{ cs.funnel?.exposure ?? 0 }}</el-descriptions-item>
            <el-descriptions-item label="点击">{{ cs.funnel?.click ?? 0 }}</el-descriptions-item>
            <el-descriptions-item label="收藏">{{ cs.funnel?.fav ?? 0 }}</el-descriptions-item>
            <el-descriptions-item label="收藏率">{{ fmtPct(cs.fav_rate) }}</el-descriptions-item>
          </el-descriptions>
          <div v-if="cs.note" class="note">ℹ️ {{ cs.note }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="mt16">
      <template #header>
        <div class="flex-between">
          <span>离线指标（metric_snapshot，按模型版本）</span>
          <el-select v-model="metricType" size="small" class="sel" @change="loadMetrics">
            <el-option label="HR@10" value="hr10" />
            <el-option label="NDCG@10" value="ndcg10" />
            <el-option label="MRR" value="mrr" />
          </el-select>
        </div>
      </template>
      <div v-show="hasSeries" ref="metricEl" class="chart" />
      <el-empty v-if="!hasSeries && !busy" :description="metricNote" />
      <div v-if="latestModelVer" class="note">当前最新模型版本：<b>{{ latestModelVer }}</b></div>
    </el-card>

    <el-card shadow="never" class="mt16">
      <template #header>在线反馈（近 30 天逐日转化）</template>
      <el-table v-if="daily.length" :data="daily" stripe size="small">
        <el-table-column prop="date" label="日期" width="120" />
        <el-table-column prop="exposure" label="曝光" width="90" />
        <el-table-column prop="click" label="点击" width="90" />
        <el-table-column label="CTR" width="90">
          <template #default="{ row }">{{ fmtPct(row.ctr) }}</template>
        </el-table-column>
        <el-table-column prop="fav" label="收藏" width="90" />
        <el-table-column label="CVR">
          <template #default="{ row }">{{ fmtPct(row.cvr) }}</template>
        </el-table-column>
      </el-table>
      <el-empty v-else :description="'暂无在线反馈记录 —— 去首页点几次「加入追番」就有了'" />
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as echarts from 'echarts'
import { api } from '../../api/client'

/**
 * 形状来源（2026-09-15 与真接口逐一核对，勿凭想象改回）：
 * - /admin/agents/health → {agents:[{agent_id,name,status_label,health_score,calls,
 *   p95_elapsed_ms,fail_cnt,degrade_cnt,...}]}
 * - /admin/metrics → {metric_type, series:[{metric_date,metric_value,model_ver,...}],
 *   latest_model_ver, online:{counts}, daily:[{date,exposure,click,ctr,cvr,fav}], note}
 *   series 是**按行的平铺快照**（每行一个日期×场景×模型版本），不是 {name,points} 嵌套；
 *   空数组 = 尚未跑 run_experiments.py，必须显示 note 而不是一张空白图。
 * - /admin/cold-start → {pool_size,funnel:{exposure,click,fav},ctr,fav_rate,
 *   by_season:[...],note,pool_items}
 */
const busy = ref(false)
const agents = ref([])
const cs = ref({})
const metricType = ref('hr10')
const metricEl = ref(null)
const hasSeries = ref(false)
const metricNote = ref('')
const latestModelVer = ref(null)
const daily = ref([])
let chart = null

const fmtPct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—')

function drawMetric (rows) {
  // 按 model_ver 分组各画一条线（同版本内按日期）；无版本号的归入「未标注」
  const groups = new Map()
  for (const r of rows) {
    const name = r.model_ver || '未标注版本'
    if (!groups.has(name)) groups.set(name, new Map())
    groups.get(name).set(r.metric_date, r.metric_value)
  }
  const dates = [...new Set(rows.map((r) => r.metric_date))].sort()
  chart = chart || echarts.init(metricEl.value)
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    grid: { left: 60, right: 20, top: 20, bottom: 46 },
    xAxis: { type: 'category', data: dates },
    yAxis: { type: 'value' },
    series: [...groups.entries()].map(([name, byDate]) => ({
      name, type: 'line', smooth: true,
      data: dates.map((dt) => byDate.get(dt) ?? null),
    })),
  }, true)
}

async function loadMetrics () {
  const d = await api.get('/api/v1/admin/metrics', { metric_type: metricType.value, days: 30 })
  hasSeries.value = (d.series || []).length > 0
  metricNote.value = d.note || '暂无离线指标快照'
  latestModelVer.value = d.latest_model_ver || null
  daily.value = d.daily || []
  if (hasSeries.value) drawMetric(d.series)
}

onMounted(async () => {
  busy.value = true
  try {
    const h = await api.get('/api/v1/admin/agents/health')
    agents.value = h.agents || []
    cs.value = await api.get('/api/v1/admin/cold-start')
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
.dim { color: #909399; font-size: 12px; }
.note { margin-top: 8px; color: #909399; font-size: 12px; }
</style>
