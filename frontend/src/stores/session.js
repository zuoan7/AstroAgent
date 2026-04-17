import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export const useSessionStore = defineStore('session', () => {
  const storedUserId = localStorage.getItem('astroagent.user_id') || 'demo-user'
  const userId = ref(storedUserId)
  const disableLongTermMemory = ref(false)
  const isStreaming = ref(false)
  const queryInput = ref('')
  const lastError = ref('')
  const currentRunId = ref('')

  function setUserId(value) {
    userId.value = value || 'demo-user'
    localStorage.setItem('astroagent.user_id', userId.value)
  }

  function setStreaming(value) {
    isStreaming.value = value
  }

  function setError(message) {
    lastError.value = message || ''
  }

  function resetRun() {
    currentRunId.value = `${Date.now()}`
    lastError.value = ''
  }

  const requestMeta = computed(() => ({
    userId: userId.value,
    disableLongTermMemory: disableLongTermMemory.value,
  }))

  return {
    userId,
    disableLongTermMemory,
    isStreaming,
    queryInput,
    lastError,
    currentRunId,
    requestMeta,
    setUserId,
    setStreaming,
    setError,
    resetRun,
  }
})
