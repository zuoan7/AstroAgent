const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

function buildUrl(path, searchParams) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  const isAbsoluteBase = /^https?:\/\//i.test(API_BASE_URL)
  const url = isAbsoluteBase
    ? new URL(normalizedPath, API_BASE_URL)
    : new URL(`${API_BASE_URL.replace(/\/$/, '')}${normalizedPath}`, window.location.origin)
  if (searchParams) {
    Object.entries(searchParams).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, value)
      }
    })
  }
  return isAbsoluteBase ? url : `${url.pathname}${url.search}${url.hash}`
}

async function request(path, options = {}) {
  const url = buildUrl(path, options.searchParams)
  const response = await fetch(url, {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })
  if (!response.ok) {
    const message = await extractErrorMessage(response)
    throw new Error(message || `HTTP ${response.status} (${url})`)
  }
  return response.json()
}

async function extractErrorMessage(response) {
  const text = await response.text()
  if (!text) {
    return ''
  }
  try {
    const data = JSON.parse(text)
    if (data?.message) {
      return data.message
    }
    return text
  } catch {
    return text
  }
}

export function getApiBaseUrl() {
  return API_BASE_URL
}

export function resolveAssetUrl(path) {
  if (!path) {
    return ''
  }
  if (/^https?:\/\//i.test(path)) {
    return path
  }
  return buildUrl(path).toString()
}

export function fetchMemoryOverview(userId, sessionId) {
  return request('/memory/overview', {
    searchParams: { user_id: userId, session_id: sessionId },
  })
}

export function fetchSessionContext(userId, sessionId) {
  return request('/session/context', {
    searchParams: { user_id: userId, session_id: sessionId },
  })
}

export function fetchAvailableModels() {
  return request('/models')
}

export function fetchSessionModel(userId, sessionId) {
  return request('/session/model', {
    searchParams: { user_id: userId, session_id: sessionId },
  })
}

export function switchSessionModel({ userId, sessionId, modelProvider, modelName }) {
  return request('/session/model', {
    method: 'POST',
    body: {
      user_id: userId,
      session_id: sessionId,
      model_provider: modelProvider,
      model_name: modelName,
    },
  })
}

export function clearMemory(userId, scope = 'all', sessionId) {
  return request('/clear_memory', {
    method: 'POST',
    searchParams: { user_id: userId, scope, session_id: sessionId },
  })
}

export function updateMemory(memoryId, payload) {
  return request(`/memories/${memoryId}`, {
    method: 'PUT',
    body: payload,
  })
}

export function deleteMemory(memoryId, userId) {
  return request(`/memories/${memoryId}`, {
    method: 'DELETE',
    searchParams: { user_id: userId },
  })
}

export function promoteCandidate(candidateId) {
  return request(`/candidates/${candidateId}/promote`, { method: 'POST' })
}

export function rejectCandidate(candidateId, reason = '') {
  return request(`/candidates/${candidateId}/reject`, {
    method: 'POST',
    searchParams: { reason },
  })
}

export function resolveConfirmation(confirmationId, status) {
  return request(`/confirmations/${confirmationId}/resolve`, {
    method: 'POST',
    body: { status },
  })
}

async function consumeEventStream(response, onEvent) {
  if (!response.ok || !response.body) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status} (${response.url})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const part of parts) {
      const lines = part.split('\n').filter((line) => line.startsWith('data: '))
      for (const line of lines) {
        const payload = line.slice(6)
        if (!payload) {
          continue
        }
        onEvent(JSON.parse(payload))
      }
    }
  }
}

export async function streamQuery({
  query,
  userId,
  sessionId,
  disableLongTermMemory,
  modelProvider,
  modelName,
  onEvent,
}) {
  const url = buildUrl('/query')
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      user_id: userId,
      session_id: sessionId,
      disable_long_term_memory: disableLongTermMemory,
      model_provider: modelProvider,
      model_name: modelName,
    }),
  })

  await consumeEventStream(response, onEvent)
}

export async function streamImageQuery({
  query,
  image,
  userId,
  sessionId,
  modelProvider,
  modelName,
  onEvent,
}) {
  const formData = new FormData()
  formData.append('query', query)
  formData.append('image', image)
  formData.append('user_id', userId)
  formData.append('session_id', sessionId)
  formData.append('model_provider', modelProvider || '')
  formData.append('model_name', modelName || '')

  const response = await fetch(buildUrl('/query_with_image'), {
    method: 'POST',
    body: formData,
  })

  await consumeEventStream(response, onEvent)
}

export async function streamAudioQuery({
  query,
  audio,
  userId,
  sessionId,
  modelProvider,
  modelName,
  onEvent,
}) {
  const formData = new FormData()
  formData.append('query', query || '')
  formData.append('audio', audio)
  formData.append('user_id', userId)
  formData.append('session_id', sessionId)
  formData.append('model_provider', modelProvider || '')
  formData.append('model_name', modelName || '')

  const response = await fetch(buildUrl('/query_with_audio'), {
    method: 'POST',
    body: formData,
  })

  await consumeEventStream(response, onEvent)
}
