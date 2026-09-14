<template>
  <div class="chat-page">
    <div ref="boxEl" class="msgs">
      <div v-for="(m, i) in msgs" :key="i" :class="['msg', m.role]">
        <div class="bubble">
          <div v-for="(seg, j) in m.segs" :key="j">
            <template v-if="seg.t === 'text'">{{ seg.v }}</template>
            <template v-else-if="seg.t === 'tool'">
              <div class="tool">🔧 {{ seg.v }}</div>
            </template>
            <div v-else class="cards">
              <div v-for="c in seg.v" :key="c.anime_id" class="mini-card"
                   @click="$router.push('/new')">
                <b>{{ c.title }}</b>
                <span class="dim">{{ c.year || '' }} {{ c.score != null ? '★' + c.score : '' }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div v-if="streaming" class="msg assistant"><div class="bubble typing">…</div></div>
    </div>
    <div class="input-bar">
      <el-input v-model="text" placeholder="想看热血战斗番，最近有什么新番？" size="large"
        :disabled="streaming" @keyup.enter="send" />
      <el-button type="primary" size="large" :loading="streaming" @click="send">发送</el-button>
    </div>
  </div>
</template>

<script setup>
import { nextTick, onMounted, ref } from 'vue'
import { auth } from '../stores/auth'

const msgs = ref([{ role: 'assistant', segs: [{ t: 'text', v: '你好！我是追番助手，可以直接告诉我你的口味～' }] }])
const text = ref(''); const streaming = ref(false); const boxEl = ref(null)
let sessionId = ''

function push (role, segs) { msgs.value.push({ role, segs }); scroll() }
function scroll () { nextTick(() => { boxEl.value?.scrollTo({ top: 1e9, behavior: 'smooth' }) }) }

async function send () {
  const q = text.value.trim()
  if (!q || streaming.value) return
  text.value = ''
  push('user', [{ t: 'text', v: q }])
  const segs = []; const reply = { role: 'assistant', segs }
  push('assistant', segs)   // 占位，SSE 逐步填充
  streaming.value = true

  // 刻意不带 session_id（首条消息）：后端会创建会话；之后带上以延续上下文
  const body = { message: q }
  if (sessionId) body.session_id = sessionId

  try {
    const resp = await fetch('/api/v1/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${auth.token}` },
      body: JSON.stringify(body),
    })
    const reader = resp.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2)
        const line = frame.split('\n').find((l) => l.startsWith('data:'))
        if (!line) continue
        let ev; try { ev = JSON.parse(line.slice(5).trim()) } catch { continue }
        handle(ev, segs)
      }
    }
  } catch (e) {
    segs.push({ t: 'text', v: `（连接中断：${e.message}）` })
  } finally {
    streaming.value = false
    if (!segs.length) segs.push({ t: 'text', v: '（无回复）' })
    scroll()
  }
}

function handle (ev, segs) {
  if (ev.type === 'session' && ev.session_id) { sessionId = ev.session_id; return }
  if (ev.type === 'tool_call') { segs.push({ t: 'tool', v: `调用 ${ev.name}…` }); scroll(); return }
  if (ev.type === 'tool_result') {
    const last = segs[segs.length - 1]
    if (last?.t === 'tool') last.v = `✓ ${ev.name}`
    return
  }
  if (ev.type === 'chunk') {
    const last = segs[segs.length - 1]
    if (last?.t === 'text') last.v += ev.text || ''
    else segs.push({ t: 'text', v: ev.text || '' })
    scroll(); return
  }
  if (ev.type === 'cards') { segs.push({ t: 'cards', v: ev.cards || [] }); scroll(); return }
  if (ev.type === 'notice') { segs.push({ t: 'text', v: `⚠ ${ev.message || ''}` }); return }
  // done / error：什么都不加，结束即可
}

onMounted(scroll)
</script>

<style scoped>
.chat-page { display: flex; flex-direction: column; height: calc(100vh - 130px); }
.msgs { flex: 1; overflow-y: auto; padding: 4px 2px; }
.msg { display: flex; margin-bottom: 10px; }
.msg.user { justify-content: flex-end; }
.bubble { max-width: 72%; background: #fff; border-radius: 10px; padding: 10px 14px; box-shadow: 0 1px 2px rgba(0,0,0,.06); white-space: pre-wrap; }
.msg.user .bubble { background: var(--brand); color: #fff; }
.msg.assistant .bubble { border: 1px solid #ebeef5; }
.typing { animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: .4; } }
.tool { font-size: 12px; color: #909399; background: #f4f4f5; border-radius: 4px; padding: 2px 8px; display: inline-block; margin: 2px 0; }
.cards { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.mini-card { border: 1px solid #ebeef5; border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer; display: flex; flex-direction: column; }
.mini-card:hover { border-color: var(--brand); }
.dim { color: #909399; font-size: 12px; }
.input-bar { display: flex; gap: 10px; padding-top: 10px; }
</style>
