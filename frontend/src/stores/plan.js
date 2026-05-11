import { ref } from 'vue'
import { defineStore } from 'pinia'

export const usePlanStore = defineStore('plan', () => {
  const steps = ref([])

  function reset() {
    steps.value = []
  }

  function applyPlanUpdate(nextSteps) {
    const currentById = new Map(steps.value.map((step) => [step.id, step]))
    steps.value = (nextSteps || []).map((step) => {
      const current = currentById.get(step.id) || {}
      const statusChanged = current.status && current.status !== step.status
      return {
        ...current,
        ...step,
        startedAt: current.startedAt || (step.status === 'running' ? Date.now() : null),
        finishedAt:
          current.finishedAt ||
          (['done', 'success', 'error', 'failed'].includes(step.status) ? Date.now() : null),
        statusHistory: [
          ...(current.statusHistory || []),
          ...(statusChanged || !current.status
            ? [{ status: step.status, timestamp: Date.now() }]
            : []),
        ],
      }
    })
  }

  function updateStep(stepId, status) {
    steps.value = steps.value.map((step) =>
      step.id === stepId
        ? {
            ...step,
            status,
            startedAt: step.startedAt || (status === 'running' ? Date.now() : null),
            finishedAt:
              step.finishedAt ||
              (['done', 'success', 'error', 'failed'].includes(status) ? Date.now() : null),
            statusHistory: [
              ...(step.statusHistory || []),
              ...(step.status === status ? [] : [{ status, timestamp: Date.now() }]),
            ],
          }
        : step
    )
  }

  return {
    steps,
    reset,
    applyPlanUpdate,
    updateStep,
  }
})
