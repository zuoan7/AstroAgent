import { ref } from 'vue'
import { defineStore } from 'pinia'

export const usePlanStore = defineStore('plan', () => {
  const steps = ref([])

  function reset() {
    steps.value = []
  }

  function applyPlanUpdate(nextSteps) {
    steps.value = nextSteps || []
  }

  function updateStep(stepId, status) {
    steps.value = steps.value.map((step) =>
      step.id === stepId ? { ...step, status } : step
    )
  }

  return {
    steps,
    reset,
    applyPlanUpdate,
    updateStep,
  }
})
