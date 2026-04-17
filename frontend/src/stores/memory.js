import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import {
  fetchMemoryOverview,
  fetchSessionContext,
  updateMemory,
  deleteMemory,
  promoteCandidate,
  rejectCandidate,
  resolveConfirmation,
  clearMemory,
} from '../lib/api'

export const useMemoryStore = defineStore('memory', () => {
  const profile = ref(null)
  const stats = ref({})
  const memories = ref([])
  const candidates = ref([])
  const confirmations = ref([])
  const recentEvents = ref([])
  const shortTerm = ref(null)
  const context = ref(null)
  const memoryHits = ref([])
  const loading = ref(false)

  const activeMemories = computed(() =>
    memories.value.filter((item) => item.status === 'active')
  )

  async function refresh(userId) {
    loading.value = true
    try {
      const [overview, session] = await Promise.all([
        fetchMemoryOverview(userId),
        fetchSessionContext(userId),
      ])
      profile.value = overview.profile
      stats.value = overview.stats
      memories.value = overview.memories
      candidates.value = overview.candidates
      confirmations.value = overview.confirmations
      recentEvents.value = overview.recent_events
      shortTerm.value = overview.short_term
      context.value = session.context
    } finally {
      loading.value = false
    }
  }

  function setMemoryHits(items) {
    memoryHits.value = items || []
  }

  async function archiveMemory(userId, memoryId) {
    await updateMemory(memoryId, { user_id: userId, status: 'archived' })
    await refresh(userId)
  }

  async function confirmMemory(userId, memoryId) {
    await updateMemory(memoryId, {
      user_id: userId,
      confirmed_by_user: true,
      metadata: { confirmed_from_workbench: true },
      confidence: 1,
    })
    await refresh(userId)
  }

  async function removeMemory(userId, memoryId) {
    await deleteMemory(memoryId, userId)
    await refresh(userId)
  }

  async function acceptCandidate(userId, candidateId) {
    await promoteCandidate(candidateId)
    await refresh(userId)
  }

  async function discardCandidate(userId, candidateId) {
    await rejectCandidate(candidateId, 'Rejected from workbench')
    await refresh(userId)
  }

  async function resolvePending(userId, confirmationId, status) {
    await resolveConfirmation(confirmationId, status)
    await refresh(userId)
  }

  async function clearAll(userId, scope) {
    await clearMemory(userId, scope)
    await refresh(userId)
  }

  return {
    profile,
    stats,
    memories,
    candidates,
    confirmations,
    recentEvents,
    shortTerm,
    context,
    memoryHits,
    loading,
    activeMemories,
    refresh,
    setMemoryHits,
    archiveMemory,
    confirmMemory,
    removeMemory,
    acceptCandidate,
    discardCandidate,
    resolvePending,
    clearAll,
  }
})
