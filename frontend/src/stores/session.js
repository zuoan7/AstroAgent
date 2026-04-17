import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'

const MODE_KEY = 'astroagent.ui_mode'
const ACCOUNTS_KEY = 'astroagent.accounts'
const ACCOUNT_KEY = 'astroagent.active_account'
const CONVERSATIONS_KEY = 'astroagent.conversations'
const CONVERSATION_KEY = 'astroagent.active_conversation'

function safeRead(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function slugify(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function createId(prefix) {
  return `${prefix}-${Date.now().toString(36)}`
}

function createConversation(title = '新会话') {
  const now = new Date().toISOString()
  return {
    id: createId('session'),
    title,
    createdAt: now,
    updatedAt: now,
  }
}

function createAccount(name = 'Demo Account') {
  const base = slugify(name) || createId('account')
  return {
    id: base,
    name,
  }
}

export const useSessionStore = defineStore('session', () => {
  const storedAccounts = safeRead(ACCOUNTS_KEY, null)
  const accounts = ref(
    Array.isArray(storedAccounts) && storedAccounts.length
      ? storedAccounts
      : [createAccount('Demo Account'), createAccount('Night Sky Club')]
  )

  const conversationsByAccount = ref(safeRead(CONVERSATIONS_KEY, {}))
  const uiMode = ref(localStorage.getItem(MODE_KEY) || 'customer')
  const activeAccountId = ref(localStorage.getItem(ACCOUNT_KEY) || accounts.value[0].id)
  const activeConversationId = ref(localStorage.getItem(CONVERSATION_KEY) || '')
  const disableLongTermMemory = ref(false)
  const isStreaming = ref(false)
  const queryInput = ref('')
  const lastError = ref('')
  const currentRunId = ref('')

  function ensureAccountConversations(accountId) {
    if (!conversationsByAccount.value[accountId] || !conversationsByAccount.value[accountId].length) {
      conversationsByAccount.value[accountId] = [createConversation('默认会话')]
    }
    if (
      activeAccountId.value === accountId &&
      !conversationsByAccount.value[accountId].some((item) => item.id === activeConversationId.value)
    ) {
      activeConversationId.value = conversationsByAccount.value[accountId][0].id
    }
  }

  accounts.value.forEach((account) => ensureAccountConversations(account.id))

  if (!accounts.value.some((account) => account.id === activeAccountId.value)) {
    activeAccountId.value = accounts.value[0].id
  }
  ensureAccountConversations(activeAccountId.value)

  const accountOptions = computed(() => accounts.value)
  const activeAccount = computed(
    () => accounts.value.find((account) => account.id === activeAccountId.value) || accounts.value[0]
  )
  const conversations = computed(
    () => conversationsByAccount.value[activeAccountId.value] || []
  )
  const activeConversation = computed(
    () => conversations.value.find((item) => item.id === activeConversationId.value) || conversations.value[0]
  )
  const userId = computed(() => activeAccount.value?.id || 'demo-account')
  const sessionId = computed(() => activeConversation.value?.id || 'default')

  function persist() {
    localStorage.setItem(MODE_KEY, uiMode.value)
    localStorage.setItem(ACCOUNTS_KEY, JSON.stringify(accounts.value))
    localStorage.setItem(ACCOUNT_KEY, activeAccountId.value)
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversationsByAccount.value))
    localStorage.setItem(CONVERSATION_KEY, activeConversationId.value)
  }

  function setMode(value) {
    uiMode.value = value === 'workbench' ? 'workbench' : 'customer'
  }

  function setActiveAccount(accountId) {
    if (!accounts.value.some((account) => account.id === accountId)) {
      return
    }
    activeAccountId.value = accountId
    ensureAccountConversations(accountId)
    queryInput.value = ''
    lastError.value = ''
  }

  function setActiveConversation(conversationId) {
    if (!conversations.value.some((item) => item.id === conversationId)) {
      return
    }
    activeConversationId.value = conversationId
    queryInput.value = ''
    lastError.value = ''
  }

  function addAccount(name) {
    const trimmed = String(name || '').trim()
    if (!trimmed) {
      return null
    }
    let baseId = slugify(trimmed) || createId('account')
    let nextId = baseId
    let index = 2
    while (accounts.value.some((account) => account.id === nextId)) {
      nextId = `${baseId}-${index}`
      index += 1
    }
    const account = { id: nextId, name: trimmed }
    accounts.value.push(account)
    conversationsByAccount.value[nextId] = [createConversation('首次会话')]
    setActiveAccount(nextId)
    return account
  }

  function addConversation(title) {
    const conversation = createConversation(String(title || '').trim() || '新会话')
    conversationsByAccount.value[activeAccountId.value] = [
      conversation,
      ...(conversationsByAccount.value[activeAccountId.value] || []),
    ]
    setActiveConversation(conversation.id)
    return conversation
  }

  function removeConversation(conversationId) {
    const nextItems = conversations.value.filter((item) => item.id !== conversationId)
    conversationsByAccount.value[activeAccountId.value] = nextItems.length
      ? nextItems
      : [createConversation('默认会话')]
    if (activeConversationId.value === conversationId) {
      activeConversationId.value = conversationsByAccount.value[activeAccountId.value][0].id
    }
  }

  function touchConversation(titleHint) {
    const current = activeConversation.value
    if (!current) {
      return
    }
    current.updatedAt = new Date().toISOString()
    if (titleHint && current.title === '新会话') {
      current.title = String(titleHint).slice(0, 24)
    }
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
    touchConversation(queryInput.value)
  }

  const requestMeta = computed(() => ({
    userId: userId.value,
    sessionId: sessionId.value,
    disableLongTermMemory: disableLongTermMemory.value,
  }))

  watch(
    [uiMode, accounts, conversationsByAccount, activeAccountId, activeConversationId],
    persist,
    { deep: true }
  )

  return {
    uiMode,
    accounts: accountOptions,
    activeAccount,
    activeAccountId,
    conversations,
    activeConversation,
    activeConversationId,
    userId,
    sessionId,
    disableLongTermMemory,
    isStreaming,
    queryInput,
    lastError,
    currentRunId,
    requestMeta,
    setMode,
    setActiveAccount,
    setActiveConversation,
    addAccount,
    addConversation,
    removeConversation,
    touchConversation,
    setStreaming,
    setError,
    resetRun,
  }
})
