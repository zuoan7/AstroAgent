import { ref } from 'vue'
import { defineStore } from 'pinia'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const streamingAnswer = ref('')
  const finalAnswer = ref(null)

  function normalizeMessage(message, index) {
    return {
      id: message.id || `${message.timestamp || Date.now()}-${index}-${message.role || 'message'}`,
      role: message.role || 'assistant',
      content: message.content || '',
      timestamp: message.timestamp || new Date().toISOString(),
      kind: message.kind || 'text',
      imageUrl: message.imageUrl || '',
      localPreviewUrl: message.localPreviewUrl || '',
      attachmentName: message.attachmentName || '',
      transcription: message.transcription || '',
      status: message.status || 'sent',
    }
  }

  function resetForNewQuery(payload) {
    const nextQuery = typeof payload === 'string' ? payload : payload?.query || ''
    const nextAttachment = typeof payload === 'string' ? null : payload?.attachment || null

    streamingAnswer.value = ''
    finalAnswer.value = null
    messages.value.push({
      id: `${Date.now()}-user`,
      role: 'user',
      content: nextQuery,
      timestamp: new Date().toISOString(),
      kind: nextAttachment?.kind || 'text',
      imageUrl: nextAttachment?.imageUrl || '',
      localPreviewUrl: nextAttachment?.localPreviewUrl || '',
      attachmentName: nextAttachment?.name || '',
      transcription: nextAttachment?.transcription || '',
      status: nextAttachment?.status || 'sent',
    })
  }

  function appendText(content) {
    streamingAnswer.value += content || ''
  }

  function hydrateMessages(items) {
    messages.value = (items || []).map((message, index) => normalizeMessage(message, index))
    streamingAnswer.value = ''
    finalAnswer.value = null
  }

  function clearConversation() {
    messages.value = []
    streamingAnswer.value = ''
    finalAnswer.value = null
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

  function attachImageToLatestUserMessage(imageUrl) {
    for (let index = messages.value.length - 1; index >= 0; index -= 1) {
      const message = messages.value[index]
      if (message.role === 'user') {
        message.kind = 'image'
        message.imageUrl = imageUrl
        message.status = 'sent'
        break
      }
    }
  }

  function attachTranscriptionToLatestUserMessage(text) {
    for (let index = messages.value.length - 1; index >= 0; index -= 1) {
      const message = messages.value[index]
      if (message.role === 'user') {
        message.transcription = text || ''
        message.status = 'sent'
        if (!message.content) {
          message.content = '已发送语音'
        }
        break
      }
    }
  }

  function markLatestUserAttachmentFailed() {
    for (let index = messages.value.length - 1; index >= 0; index -= 1) {
      const message = messages.value[index]
      if (message.role === 'user' && message.status === 'uploading') {
        message.status = 'failed'
        break
      }
    }
  }

  return {
    messages,
    streamingAnswer,
    finalAnswer,
    resetForNewQuery,
    appendText,
    hydrateMessages,
    clearConversation,
    commitFinalAnswer,
    attachImageToLatestUserMessage,
    attachTranscriptionToLatestUserMessage,
    markLatestUserAttachmentFailed,
  }
})
