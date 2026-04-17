import { ref } from 'vue'
import { defineStore } from 'pinia'

export const useEvidenceStore = defineStore('evidence', () => {
  const items = ref([])
  const finalSources = ref([])

  function reset() {
    items.value = []
    finalSources.value = []
  }

  function addEvidence(item) {
    items.value.unshift(item)
  }

  return {
    items,
    finalSources,
    reset,
    addEvidence,
  }
})
