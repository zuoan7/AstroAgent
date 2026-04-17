<template>
  <main class="app-shell">
    <WorkbenchHeader
      :user-id="session.userId"
      :disable-long-term-memory="session.disableLongTermMemory"
      :is-streaming="session.isStreaming"
      @update:user-id="handleUserIdChange"
      @toggle-memory="(value) => (session.disableLongTermMemory = value)"
    />

    <QueryConsole
      v-model="session.queryInput"
      :disabled="session.isStreaming"
      :error="session.lastError"
      @submit="handleSubmit"
      @refresh="refreshMemory"
      @clear-session="clearAllMemory"
    />

    <section class="content-grid">
      <PlanPanel :steps="plan.steps" :overview="trace.overview" />
      <TracePanel
        :items="trace.items"
        :reasoning-summary="trace.reasoningSummary"
        :overview="trace.overview"
      />
      <EvidencePanel :items="evidence.items" />
      <ChatPanel
        :messages="chat.messages"
        :streaming-answer="chat.streamingAnswer"
        :final-answer="chat.finalAnswer"
      />
      <MemoryPanel
        :profile="memory.profile"
        :stats="memory.stats"
        :memories="memory.memories"
        :candidates="memory.candidates"
        :confirmations="memory.confirmations"
        :memory-hits="memory.memoryHits"
        :short-term="memory.shortTerm"
        :loading="memory.loading"
        @confirm-memory="(id) => memory.confirmMemory(session.userId, id)"
        @archive-memory="(id) => memory.archiveMemory(session.userId, id)"
        @delete-memory="(id) => memory.removeMemory(session.userId, id)"
        @accept-candidate="(id) => memory.acceptCandidate(session.userId, id)"
        @reject-candidate="(id) => memory.discardCandidate(session.userId, id)"
        @resolve-confirmation="(id, status) => memory.resolvePending(session.userId, id, status)"
      />
    </section>
  </main>
</template>

<script setup>
import { onMounted } from 'vue'
import WorkbenchHeader from './components/WorkbenchHeader.vue'
import QueryConsole from './components/QueryConsole.vue'
import PlanPanel from './components/PlanPanel.vue'
import TracePanel from './components/TracePanel.vue'
import EvidencePanel from './components/EvidencePanel.vue'
import MemoryPanel from './components/MemoryPanel.vue'
import ChatPanel from './components/ChatPanel.vue'
import { streamQuery } from './lib/api'
import { useSessionStore } from './stores/session'
import { usePlanStore } from './stores/plan'
import { useTraceStore } from './stores/trace'
import { useEvidenceStore } from './stores/evidence'
import { useMemoryStore } from './stores/memory'
import { useChatStore } from './stores/chat'

const session = useSessionStore()
const plan = usePlanStore()
const trace = useTraceStore()
const evidence = useEvidenceStore()
const memory = useMemoryStore()
const chat = useChatStore()

async function refreshMemory() {
  try {
    await memory.refresh(session.userId)
  } catch (error) {
    session.setError(String(error))
  }
}

function resetRealtimeStores(query) {
  plan.reset()
  trace.reset()
  trace.startRun()
  evidence.reset()
  memory.setMemoryHits([])
  chat.resetForNewQuery(query)
}

function handleEvent(event) {
  switch (event.type) {
    case 'plan_update':
      plan.applyPlanUpdate(event.steps)
      break
    case 'step_start':
      plan.updateStep(event.step_id, 'running')
      break
    case 'step_end':
      plan.updateStep(event.step_id, event.status || 'done')
      break
    case 'tool_start':
      trace.startTool(event)
      break
    case 'tool_end':
      trace.endTool(event)
      break
    case 'evidence_found':
      evidence.addEvidence(event)
      break
    case 'memory_hit':
      memory.setMemoryHits([event, ...memory.memoryHits])
      evidence.addEvidence({
        source_id: event.memory_id,
        kind: 'memory',
        title: `${event.memory_type}.${event.key}`,
        snippet: typeof event.value === 'string' ? event.value : JSON.stringify(event.value),
        reason: event.reason,
      })
      break
    case 'text':
      chat.appendText(event.content)
      break
    case 'reasoning_summary':
      trace.reasoningSummary = event.summary || event.content || ''
      break
    case 'final_answer':
      trace.finishRun(event)
      chat.commitFinalAnswer(event)
      memory.setMemoryHits(event.memory_hits || [])
      evidence.finalSources = event.sources || []
      break
    default:
      break
  }
}

async function handleSubmit() {
  const query = session.queryInput.trim()
  if (!query || session.isStreaming) {
    return
  }

  session.resetRun()
  session.setStreaming(true)
  resetRealtimeStores(query)

  try {
    await streamQuery({
      query,
      userId: session.userId,
      disableLongTermMemory: session.disableLongTermMemory,
      onEvent: handleEvent,
    })
    await refreshMemory()
  } catch (error) {
    session.setError(String(error))
  } finally {
    session.setStreaming(false)
  }
}

async function clearAllMemory() {
  try {
    await memory.clearAll(session.userId, 'all')
  } catch (error) {
    session.setError(String(error))
  }
}

async function handleUserIdChange(value) {
  session.setUserId(value)
  await refreshMemory()
}

onMounted(refreshMemory)
</script>
