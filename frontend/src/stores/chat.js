import { ref } from 'vue'
import { defineStore } from 'pinia'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const streamingAnswer = ref('')
  const finalAnswer = ref(null)

  function resetForNewQuery(query) {
    streamingAnswer.value = ''
    finalAnswer.value = null
    messages.value.push({
      id: `${Date.now()}-user`,
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    })
  }

  function appendText(content) {
    streamingAnswer.value += content || ''
  }

  function commitFinalAnswer(payload) {
    finalAnswer.value = payload
    streamingAnswer.value = ''
    messages.value.push({
      id: `${Date.now()}-assistant`,
      role: 'assistant',
      content: payload.final_answer,
      timestamp: new Date().toISOString(),
    })
  }

  return {
    messages,
    streamingAnswer,
    finalAnswer,
    resetForNewQuery,
    appendText,
    commitFinalAnswer,
  }
})
