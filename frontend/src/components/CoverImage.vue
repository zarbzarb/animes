<template>
  <div class="cover" :style="{ width: w, height: h }">
    <el-image
      :src="src"
      fit="cover"
      lazy
      class="img"
      @error="failed = true"
    >
      <template #placeholder>
        <div class="ph loading"><el-icon><Loading /></el-icon></div>
      </template>
      <template #error>
        <div class="ph failed" :style="grad">
          <span class="initial">{{ initial }}</span>
          <span v-if="showTitle" class="t">{{ title }}</span>
        </div>
      </template>
    </el-image>
    <!-- src 为空时 el-image 不渲染 error 槽，自己兜底 -->
    <div v-if="!src" class="ph failed" :style="grad">
      <span class="initial">{{ initial }}</span>
      <span v-if="showTitle" class="t">{{ title }}</span>
    </div>
  </div>
</template>

<script setup>
/**
 * 封面图 + 加载失败降级。
 *
 * 为什么必须有降级：库里 14,136 条番剧全部指向 cdn.myanimelist.net（MAL CDN），
 * 该域名在国内网络环境下经常加载失败/超时 —— 不做降级的话页面会大面积破图。
 * 降级形态：按标题哈希选一对渐变色 + 标题首字，保证同一番剧每次降级长相一致。
 */
import { computed, ref } from 'vue'
import { Loading } from '@element-plus/icons-vue'

const props = defineProps({
  src: { type: String, default: '' },
  title: { type: String, default: '' },
  w: { type: String, default: '64px' },
  h: { type: String, default: '88px' },
  showTitle: { type: Boolean, default: false },
})

const failed = ref(false)
const initial = computed(() => (props.title || '番').trim().charAt(0).toUpperCase())
const grad = computed(() => {
  // 稳定哈希：同一标题永远落到同一对颜色
  let h = 0
  for (const ch of props.title || '') h = (h * 31 + ch.charCodeAt(0)) >>> 0
  const pairs = [
    ['#f78fb3', '#c44569'], ['#7bed9f', '#2ed573'], ['#70a1ff', '#3742fa'],
    ['#ffa502', '#ff6348'], ['#a29bfe', '#6c5ce7'], ['#4bcffa', '#0abde3'],
  ]
  const [a, b] = pairs[h % pairs.length]
  return { background: `linear-gradient(135deg, ${a}, ${b})` }
})
</script>

<style scoped>
.cover { position: relative; flex: none; border-radius: 8px; overflow: hidden; background: #f0f1f5; }
.img { width: 100%; height: 100%; display: block; }
.ph {
  width: 100%; height: 100%;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 4px; color: #fff; user-select: none;
}
.ph.loading { color: #c0c4cc; font-size: 18px; }
.ph.failed .initial { font-size: 22px; font-weight: 700; line-height: 1; }
.ph.failed .t {
  font-size: 11px; max-width: 92%; line-height: 1.25;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
</style>
