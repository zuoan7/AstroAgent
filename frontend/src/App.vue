<template>
  <main class="app-shell">
    <SurfaceSwitcher :mode="session.uiMode" @change="handleModeChange" />

    <CustomerConsole
      v-if="session.uiMode === 'customer'"
      v-model="session.queryInput"
      :accounts="session.accounts"
      :active-account-id="session.activeAccountId"
      :active-account-name="session.activeAccount?.name || 'Demo Account'"
      :conversations="session.conversations"
      :active-conversation-id="session.sessionId"
      :active-conversation-title="session.activeConversation?.title || '会话'"
      :user-id="session.userId"
      :session-id="session.sessionId"
      :model-options="modelOptions"
      :selected-model-key="selectedModelKey"
      :current-model-label="session.modelLabel"
      :messages="chat.messages"
      :streaming-answer="chat.streamingAnswer"
      :disabled="session.isStreaming"
      :is-streaming="session.isStreaming"
      :error="session.lastError"
      :image-file-name="pendingImageFileName"
      :audio-file-name="pendingAudioFileName"
      :image-preview-url="pendingImagePreviewUrl"
      :upload-state="uploadState"
      :is-recording="isRecording"
      :can-record="canRecord"
      :recording-duration-sec="recordingDurationSec"
      @submit="handleSubmit"
      @add-account="handleAddAccount"
      @select-account="handleSelectAccount"
      @add-conversation="handleAddConversation"
      @select-conversation="handleSelectConversation"
      @remove-conversation="handleRemoveConversation"
      @clear-session="clearCurrentSession"
      @change-model="handleModelChange"
      @select-image="handleImageSelected"
      @select-audio="handleAudioSelected"
      @clear-image="clearPendingImage"
      @clear-audio="clearPendingAudio"
      @toggle-recording="toggleRecording"
    />

    <template v-else>
      <WorkbenchHeader
        :account-name="session.activeAccount?.name || 'Demo Account'"
        :user-id="session.userId"
        :session-id="session.sessionId"
        :current-model-label="session.modelLabel"
        :disable-long-term-memory="session.disableLongTermMemory"
        :is-streaming="session.isStreaming"
        @toggle-memory="(value) => (session.disableLongTermMemory = value)"
      />

      <QueryConsole
        v-model="session.queryInput"
        :disabled="session.isStreaming"
        :error="session.lastError"
        :model-options="modelOptions"
        :selected-model-key="selectedModelKey"
        :current-model-label="session.modelLabel"
        :image-file-name="pendingImageFileName"
        :audio-file-name="pendingAudioFileName"
        :image-preview-url="pendingImagePreviewUrl"
        :upload-state="uploadState"
        :is-recording="isRecording"
        :can-record="canRecord"
        :recording-duration-sec="recordingDurationSec"
        @submit="handleSubmit"
        @refresh="refreshMemory"
        @clear-session="clearCurrentSession"
        @change-model="handleModelChange"
        @select-image="handleImageSelected"
        @select-audio="handleAudioSelected"
        @clear-image="clearPendingImage"
        @clear-audio="clearPendingAudio"
        @toggle-recording="toggleRecording"
      />

      <section class="content-grid">
        <PlanPanel :steps="plan.steps" :overview="trace.overview" />
        <TracePanel
          :items="trace.items"
          :events="trace.events"
          :event-type-counts="trace.eventTypeCounts"
          :errors="trace.errors"
          :final-report="trace.finalReport"
          :reasoning-summary="trace.reasoningSummary"
          :overview="trace.overview"
          :latency-metrics="trace.latencyMetrics"
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
          @confirm-memory="(id) => memory.confirmMemory(session.userId, session.sessionId, id)"
          @archive-memory="(id) => memory.archiveMemory(session.userId, session.sessionId, id)"
          @delete-memory="(id) => memory.removeMemory(session.userId, session.sessionId, id)"
          @accept-candidate="(id) => memory.acceptCandidate(session.userId, session.sessionId, id)"
          @reject-candidate="(id) => memory.discardCandidate(session.userId, session.sessionId, id)"
          @resolve-confirmation="(id, status) => memory.resolvePending(session.userId, session.sessionId, id, status)"
        />
      </section>
    </template>
  </main>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import SurfaceSwitcher from './components/SurfaceSwitcher.vue'
import CustomerConsole from './components/CustomerConsole.vue'
import WorkbenchHeader from './components/WorkbenchHeader.vue'
import QueryConsole from './components/QueryConsole.vue'
import PlanPanel from './components/PlanPanel.vue'
import TracePanel from './components/TracePanel.vue'
import EvidencePanel from './components/EvidencePanel.vue'
import MemoryPanel from './components/MemoryPanel.vue'
import ChatPanel from './components/ChatPanel.vue'
import {
  fetchAvailableModels,
  fetchSessionModel,
  resolveAssetUrl,
  streamAudioQuery,
  streamImageQuery,
  streamQuery,
  switchSessionModel,
} from './lib/api'
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
const pendingImageFile = ref(null)
const pendingAudioFile = ref(null)
const pendingImagePreviewUrl = ref('')
const uploadState = ref('idle')
const isRecording = ref(false)
const recordingDurationSec = ref(0)
const mediaRecorder = ref(null)
const mediaStream = ref(null)
const recordingChunks = ref([])
const recordingTimer = ref(null)

const pendingImageFileName = computed(() => pendingImageFile.value?.name || '')
const pendingAudioFileName = computed(() => pendingAudioFile.value?.name || '')
const canRecord = computed(
  () => typeof navigator !== 'undefined'
    && typeof navigator.mediaDevices?.getUserMedia === 'function'
    && typeof window !== 'undefined'
    && typeof window.MediaRecorder !== 'undefined'
)
const modelOptions = computed(() =>
  (session.availableModels || []).map((provider) => ({
    value: `${provider.provider}::${provider.default_model}`,
    label: `${provider.display_name} / ${provider.default_model}`,
    provider: provider.provider,
    modelName: provider.default_model,
    configured: Boolean(provider.configured),
  }))
)
const selectedModelKey = computed(() => `${session.modelProvider}::${session.modelName}`)

function resetRealtimeStores(payload = '') {
  plan.reset()
  trace.reset()
  trace.startRun()
  evidence.reset()
  memory.setMemoryHits([])
  if (payload) {
    chat.resetForNewQuery(payload)
  }
}

async function refreshMemory(options = {}) {
  const { hydrateChat = true } = options
  try {
    await memory.refresh(session.userId, session.sessionId)
    if (hydrateChat) {
      chat.hydrateMessages(memory.shortTerm?.messages || [])
    }
  } catch (error) {
    session.setError(String(error))
  }
}

async function initializeModels() {
  try {
    const data = await fetchAvailableModels()
    session.setAvailableModels(data.providers || [])
    const current = await fetchSessionModel(session.userId, session.sessionId)
    session.setConversationModel({
      provider: current.model_provider,
      modelName: current.model_name,
      modelLabel: current.model_label,
    })
  } catch (error) {
    session.setError(String(error))
  }
}

async function syncCurrentConversationModel() {
  try {
    const response = await switchSessionModel({
      userId: session.userId,
      sessionId: session.sessionId,
      modelProvider: session.modelProvider,
      modelName: session.modelName,
    })
    session.setConversationModel({
      provider: response.model_provider,
      modelName: response.model_name,
      modelLabel: response.model_label,
    })
  } catch (error) {
    session.setError(String(error))
  }
}

async function handleModelChange(value) {
  const [provider, modelName] = String(value || '').split('::')
  const previous = {
    provider: session.modelProvider,
    modelName: session.modelName,
    modelLabel: session.modelLabel,
  }
  session.setConversationModel({
    provider,
    modelName,
    modelLabel: `${provider} / ${modelName}`,
  })
  try {
    const response = await switchSessionModel({
      userId: session.userId,
      sessionId: session.sessionId,
      modelProvider: provider,
      modelName,
    })
    session.setConversationModel({
      provider: response.model_provider,
      modelName: response.model_name,
      modelLabel: response.model_label,
    })
  } catch (error) {
    session.setConversationModel(previous)
    session.setError(String(error))
  }
}

function handleEvent(event) {
  trace.recordEvent?.(event)
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
    case 'route_decision':
      trace.setRoute(event)
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
    case 'thinking':
      break
    case 'warning':
    case 'error':
      session.setError(event.content || event.meta?.message || event.type)
      break
    case 'image':
      chat.attachImageToLatestUserMessage(resolveAssetUrl(event.url))
      uploadState.value = 'sent'
      break
    case 'transcription':
      chat.attachTranscriptionToLatestUserMessage(event.text || '')
      uploadState.value = 'sent'
      break
    case 'reasoning_summary':
      trace.reasoningSummary = event.summary || event.content || ''
      break
    case 'latency_metrics':
      trace.setLatency(event)
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

function clearPendingImage() {
  if (pendingImagePreviewUrl.value) {
    URL.revokeObjectURL(pendingImagePreviewUrl.value)
  }
  pendingImageFile.value = null
  pendingImagePreviewUrl.value = ''
  if (!pendingAudioFile.value && !isRecording.value) {
    uploadState.value = 'idle'
  }
}

function clearPendingAudio() {
  pendingAudioFile.value = null
  if (!pendingImageFile.value && !isRecording.value) {
    uploadState.value = 'idle'
  }
}

function clearPendingUploads() {
  clearPendingImage()
  clearPendingAudio()
}

function handleImageSelected(file) {
  if (!file) {
    clearPendingImage()
    return
  }
  clearPendingImage()
  pendingImageFile.value = file
  pendingImagePreviewUrl.value = URL.createObjectURL(file)
  uploadState.value = 'preview'
  clearPendingAudio()
}

function handleAudioSelected(file) {
  if (!file) {
    clearPendingAudio()
    return
  }
  pendingAudioFile.value = file
  uploadState.value = 'ready'
  clearPendingImage()
}

function stopRecordingTimer() {
  if (recordingTimer.value) {
    window.clearInterval(recordingTimer.value)
    recordingTimer.value = null
  }
}

function releaseMediaStream() {
  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach((track) => track.stop())
    mediaStream.value = null
  }
}

async function toggleRecording() {
  if (isRecording.value) {
    mediaRecorder.value?.stop()
    return
  }
  if (!canRecord.value) {
    session.setError('当前浏览器不支持录音。')
    return
  }

  try {
    session.setError('')
    clearPendingUploads()
    recordingChunks.value = []
    mediaStream.value = await navigator.mediaDevices.getUserMedia({ audio: true })
    const recorder = new window.MediaRecorder(mediaStream.value)
    mediaRecorder.value = recorder
    recorder.ondataavailable = (event) => {
      if (event.data?.size) {
        recordingChunks.value.push(event.data)
      }
    }
    recorder.onstop = () => {
      const blob = new Blob(recordingChunks.value, { type: recorder.mimeType || 'audio/webm' })
      const extension = blob.type.includes('ogg') ? 'ogg' : blob.type.includes('mp4') ? 'm4a' : 'webm'
      const file = new File([blob], `recording-${Date.now()}.${extension}`, { type: blob.type || 'audio/webm' })
      isRecording.value = false
      recordingDurationSec.value = 0
      stopRecordingTimer()
      releaseMediaStream()
      mediaRecorder.value = null
      recordingChunks.value = []
      handleAudioSelected(file)
    }
    recorder.onerror = () => {
      session.setError('录音失败，请重试。')
      isRecording.value = false
      recordingDurationSec.value = 0
      stopRecordingTimer()
      releaseMediaStream()
      mediaRecorder.value = null
      recordingChunks.value = []
      uploadState.value = 'idle'
    }
    recorder.start()
    isRecording.value = true
    recordingDurationSec.value = 0
    uploadState.value = 'recording'
    recordingTimer.value = window.setInterval(() => {
      recordingDurationSec.value += 1
    }, 1000)
  } catch {
    session.setError('无法访问麦克风，请检查浏览器权限。')
    stopRecordingTimer()
    releaseMediaStream()
    mediaRecorder.value = null
    uploadState.value = 'idle'
  }
}

async function handleSubmit() {
  const query = session.queryInput.trim()
  const hasImage = Boolean(pendingImageFile.value)
  const hasAudio = Boolean(pendingAudioFile.value)

  if (session.isStreaming || (!query && !hasAudio)) {
    return
  }
  if (hasImage && !query) {
    session.setError('图片提问需要同时输入文本问题。')
    return
  }

  session.resetRun()
  session.setStreaming(true)
  resetRealtimeStores({
    query: query || (hasAudio ? '已发送语音' : ''),
    attachment: hasImage
      ? {
          kind: 'image',
          name: pendingImageFile.value?.name || '',
          imageUrl: pendingImagePreviewUrl.value,
          localPreviewUrl: pendingImagePreviewUrl.value,
          status: 'uploading',
        }
      : hasAudio
        ? {
          kind: 'audio',
          name: pendingAudioFile.value?.name || '',
          status: 'uploading',
        }
        : null,
  })

  try {
    if (hasImage) {
      uploadState.value = 'uploading'
      await streamImageQuery({
        query,
        image: pendingImageFile.value,
        userId: session.userId,
        sessionId: session.sessionId,
        modelProvider: session.modelProvider,
        modelName: session.modelName,
        onEvent: handleEvent,
      })
    } else if (hasAudio) {
      uploadState.value = 'uploading'
      await streamAudioQuery({
        query,
        audio: pendingAudioFile.value,
        userId: session.userId,
        sessionId: session.sessionId,
        modelProvider: session.modelProvider,
        modelName: session.modelName,
        onEvent: handleEvent,
      })
    } else {
      await streamQuery({
        query,
        userId: session.userId,
        sessionId: session.sessionId,
        disableLongTermMemory: session.disableLongTermMemory,
        modelProvider: session.modelProvider,
        modelName: session.modelName,
        onEvent: handleEvent,
      })
    }
    await refreshMemory({ hydrateChat: !hasImage && !hasAudio })
    clearPendingUploads()
    session.queryInput = ''
  } catch (error) {
    chat.markLatestUserAttachmentFailed()
    uploadState.value = 'failed'
    session.setError(String(error))
  } finally {
    session.setStreaming(false)
  }
}

async function clearCurrentSession() {
  try {
    await memory.clearAll(session.userId, session.sessionId, 'session')
    resetRealtimeStores()
    chat.clearConversation()
    clearPendingUploads()
  } catch (error) {
    session.setError(String(error))
  }
}

async function handleAddAccount(name) {
  session.addAccount(name)
  resetRealtimeStores()
  chat.clearConversation()
  clearPendingUploads()
  await syncCurrentConversationModel()
  await refreshMemory()
}

async function handleSelectAccount(accountId) {
  session.setActiveAccount(accountId)
  resetRealtimeStores()
  chat.clearConversation()
  clearPendingUploads()
  await syncCurrentConversationModel()
  await refreshMemory()
}

async function handleAddConversation(title) {
  session.addConversation(title)
  resetRealtimeStores()
  chat.clearConversation()
  clearPendingUploads()
  await syncCurrentConversationModel()
  await refreshMemory()
}

async function handleSelectConversation(conversationId) {
  session.setActiveConversation(conversationId)
  resetRealtimeStores()
  chat.clearConversation()
  clearPendingUploads()
  await syncCurrentConversationModel()
  await refreshMemory()
}

async function handleRemoveConversation(conversationId) {
  const target = session.conversations.find((item) => item.id === conversationId)
  if (!target) {
    return
  }
  const confirmed = window.confirm(`确认删除会话“${target.title}”吗？`)
  if (!confirmed) {
    return
  }

  try {
    await memory.clearAll(session.userId, conversationId, 'session')
  } catch (error) {
    session.setError(String(error))
    return
  }

  session.removeConversation(conversationId)
  resetRealtimeStores()
  chat.clearConversation()
  clearPendingUploads()
  await syncCurrentConversationModel()
  await refreshMemory()
}

function handleModeChange(mode) {
  session.setMode(mode)
}

onMounted(async () => {
  await initializeModels()
  await syncCurrentConversationModel()
  await refreshMemory()
})

onBeforeUnmount(() => {
  stopRecordingTimer()
  if (isRecording.value) {
    mediaRecorder.value?.stop()
  }
  releaseMediaStream()
  clearPendingImage()
})
</script>
